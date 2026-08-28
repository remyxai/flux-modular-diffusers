# DyPE for FLUX — training-free ultra-high-resolution as a Modular Diffusers community block.
#
# DyPE ("Dynamic Position Extrapolation", arXiv:2510.20766, MIT ref https://github.com/guyyariv/DyPE):
# a timestep-dynamic YaRN / NTK-by-parts RoPE schedule (kappa = t^2) applied at inference, so an
# off-the-shelf FLUX.1 checkpoint renders coherently at up to 4096x4096 with no fine-tuning.
# Optional method="spectral" adds SEGA (arXiv:2605.22668, Apache ref wildminder/ComfyUI-DyPE): a
# per-RoPE-dimension attention temperature from the latent's Fourier spectrum, reducing high-freq speckle.
# AI-assisted port (disclosed in the repo README); DyPE math is the bit-exact validated hook implementation.
#
# Modular form: DyPE overrides the transformer's `pos_embed` and feeds it the current timestep (+ spectral
# profiles) via a native forward-pre-hook that reads the kwargs the standard FLUX denoise already passes,
# so this block needs no custom attention processor. The >2K flow-match "mu-cap" (base_shift == max_shift)
# is applied to the scheduler here (else a direct high-res denoise collapses to a woven blob).

import math

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device, randn_tensor
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    # Inverse dim formula to find the dimension index of a given number of rotations
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    """
    Find the correction range for NTK-by-parts interpolation.
    """
    low = math.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = math.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base, scale):
    """
    Calculate the new base for NTK-aware scaling.
    """
    return base * (scale ** (dim / (dim - 2)))


# ---------------------------------------------------------------------------
# SEGA: spectral-energy helpers (method="spectral")
# ---------------------------------------------------------------------------


def compute_base_mscale(target_res: int, training_res: int, coefficient: float = 0.08) -> float:
    r"""Reference magnitude ``m_ref = (R_target / R_train) ** kappa`` (>= 1.0)."""
    s = max(float(target_res) / float(training_res), 1.0)
    return s**coefficient


@torch.no_grad()
def compute_spectral_energy_profile(
    hidden_states: torch.Tensor, height: int, width: int, n_bins: int
) -> torch.Tensor:
    r"""
    Radial (isotropic) spectral energy profile ``E_iso``. Reshapes the leading ``height * width`` tokens into an
    ``H x W`` spatial map, averages over batch and channels, mean-centres, computes the 2D FFT power spectrum, and
    bins the power into ``n_bins`` concentric rings.
    """
    if hidden_states.dim() == 3:
        B, S, C = hidden_states.shape
        n_spatial = min(S, height * width)
        spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    elif hidden_states.dim() == 4:
        spatial = hidden_states
    else:
        raise ValueError(f"hidden_states must be 3-D or 4-D, got {hidden_states.dim()}-D")

    spatial_map = spatial.float().mean(dim=(0, -1))  # (H, W)
    spatial_map = spatial_map - spatial_map.mean()

    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().pow(2)

    cy, cx = height / 2.0, width / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(yy**2 + xx**2)
    radius = radius / (radius.max() + 1e-8)

    bin_idx = (radius * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()
    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    return energy_sum / (energy_cnt + 1e-8)


@torch.no_grad()
def compute_axis_spectral_profiles(
    hidden_states: torch.Tensor, height: int, width: int, n_bins_h: int, n_bins_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Per-axis (height, width) 1-D spectral energy profiles, so horizontal and vertical RoPE dimensions can be adjusted
    independently.
    """
    if hidden_states.dim() == 3:
        B, S, C = hidden_states.shape
        n_spatial = min(S, height * width)
        spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    elif hidden_states.dim() == 4:
        spatial = hidden_states
    else:
        raise ValueError(f"hidden_states must be 3-D or 4-D, got {hidden_states.dim()}-D")

    sm = spatial.float().mean(dim=(0, -1))  # (H, W)
    sm = sm - sm.mean()

    def _axis_profile(sm: torch.Tensor, axis: int, n_bins: int, length: int) -> torch.Tensor:
        fft = torch.fft.fft(sm, dim=axis)
        power = fft.abs().pow(2).mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx = (freq_norm * n_bins).long().clamp(0, n_bins - 1)
        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    return (
        _axis_profile(sm, axis=0, n_bins=n_bins_h, length=height),
        _axis_profile(sm, axis=1, n_bins=n_bins_w, length=width),
    )


@torch.no_grad()
def compute_dynamic_spread(
    energy_profile: torch.Tensor, spread_min: float = 0.0, spread_max: float = 1.0, alpha: float = 1.5
) -> float:
    r"""
    Spectral-flatness-driven spread in ``[spread_min, spread_max]``. Flat (noise-like) spectrum -> ``spread_min``;
    concentrated (structured) spectrum -> ``spread_max``.
    """
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)


@torch.no_grad()
def compute_spectral_allocation(
    energy_profile: torch.Tensor,
    freqs: torch.Tensor,
    base_mscale: float,
    spread: float,
    alpha: float = 0.15,
    beta: float = 1.5,
    min_mscale: float = 1.0,
) -> torch.Tensor:
    r"""
    Per-RoPE-dimension mscale from a spectral energy profile (SEGA). High spectral-energy dimensions get a *lower*
    ``m_k`` (sharpness-biased) while low-energy dimensions get a *higher* ``m_k``; the zero-mean constraint on the
    correction redistributes rather than shifts the magnitude.
    """
    D_half = freqs.shape[0]
    eps = 1e-8

    # Degenerate cases -> uniform base_mscale
    if spread <= 0.0 or alpha <= 0.0:
        return torch.full((D_half,), float(base_mscale), device=freqs.device, dtype=torch.float32)

    # Map each RoPE dim to its FFT bin via log-period
    periods = 2.0 * math.pi / freqs.clamp(min=eps)
    log_periods = torch.log(periods)
    min_lp, max_lp = log_periods.min(), log_periods.max()
    if (max_lp - min_lp).item() > 1e-6:
        lp_norm = (log_periods - min_lp) / (max_lp - min_lp)  # 0=high-freq, 1=low-freq
    else:
        lp_norm = torch.zeros_like(log_periods)

    n_bins = energy_profile.shape[0]
    bin_pos = (1.0 - lp_norm) * (n_bins - 1)
    j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)

    E = energy_profile.to(freqs.device).clamp(min=eps)
    log_E = torch.log(E)
    raw = log_E[j_low] * (1.0 - frac) + log_E[j_high] * frac

    # Standardise + tanh + re-centre (zero-sum property)
    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    # Final per-dim mscale
    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    return m.clamp(min=float(min_mscale)).to(torch.float32)


def _dype_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  # torch.float32, torch.float64 (flux)
    yarn=False,
    max_pe_len=None,
    ori_max_pe_len=64,
    dype=False,
    current_timestep=1.0,
    spectral_mscale=None,
):
    r"""
    Precompute the frequency tensor for complex exponentials (cis) with RoPE. Supports YaRN interpolation (optionally
    modulated by the DyPE timestep schedule) and, via `spectral_mscale`, SEGA's per-dimension spectral attention
    temperature.

    Args:
        dim (`int`):
            Dimension of the frequency tensor.
        pos (`torch.Tensor`):
            Position indices for the frequency tensor. [S] or scalar.
        theta (`float`, *optional*, defaults to `10000.0`):
            Scaling factor for frequency computation.
        use_real (`bool`, *optional*, defaults to `False`):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to `1.0`):
            Scaling factor for linear interpolation.
        ntk_factor (`float`, *optional*, defaults to `1.0`):
            Scaling factor for NTK-Aware RoPE.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If True and use_real, real and imaginary parts are interleaved with themselves to reach `dim`. Otherwise,
            they are concatenated.
        freqs_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            Data type of the frequency tensor. `torch.float64` is used by models such as Flux.
        yarn (`bool`, *optional*, defaults to `False`):
            If True, use YaRN interpolation combining NTK, linear, and base methods.
        max_pe_len (`int` or `torch.Tensor`, *optional*):
            Maximum position encoding length (current patches per axis for vision models).
        ori_max_pe_len (`int`, *optional*, defaults to `64`):
            Original maximum position encoding length (base patches per axis, 1024 // 16 = 64 for Flux).
        dype (`bool`, *optional*, defaults to `False`):
            If True, enable DyPE (Dynamic Position Extrapolation) with timestep-aware scaling of the correction
            ranges (`kappa = current_timestep**2`).
        current_timestep (`float`, *optional*, defaults to `1.0`):
            Current timestep for DyPE, normalized to [0, 1] where 1 is pure noise.
        spectral_mscale (`torch.Tensor`, *optional*):
            Per-dimension SEGA attention temperature of shape `[dim // 2]`. When provided, it replaces the scalar YaRN
            attention temperature and is applied to the returned `cos`/`sin`.

    Returns:
        `torch.Tensor`: Precomputed frequency tensor for complex exponentials. [S, D/2]. If `use_real=True`, returns a
        tuple of `(cos, sin)` tensors.
    """
    assert dim % 2 == 0

    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0 = 1.25
        beta_1 = 0.75
        gamma_0 = 16
        gamma_1 = 2

        exponents = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
        freqs_base = 1.0 / (theta**exponents)
        # Position interpolation (PI) frequencies
        freqs_linear = 1.0 / (scale * theta**exponents)

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(new_base, exponents)
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        if dype:
            kappa = current_timestep**2.0  # kappa(t) = t^lambda_t, with lambda_t = 2
            beta_0 = beta_0 * kappa
            beta_1 = beta_1 * kappa

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        if dype:
            gamma_0 = gamma_0 * kappa
            gamma_1 = gamma_1 * kappa

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low = max(0, low)
        high = min(dim // 2, high)

        freqs_mask = 1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype)
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask
    else:
        theta_ntk = theta * ntk_factor
        exponents = torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim
        freqs = 1.0 / (theta_ntk**exponents) / linear_factor

    freqs = torch.outer(pos, freqs)

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]

        if spectral_mscale is not None:
            # SEGA per-RoPE-dimension attention temperature. Replaces the scalar YaRN temperature below.
            ms = spectral_mscale.to(device=freqs_cos.device, dtype=freqs_cos.dtype).repeat_interleave(2)  # [D]
            freqs_cos = freqs_cos * ms
            freqs_sin = freqs_sin * ms
        elif yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            # YaRN attention temperature. `torch.ones_like` is used instead of a plain `torch.tensor(1.0)` so the
            # constant is materialized on the same device/dtype as `scale`.
            mscale = torch.where(scale <= 1.0, torch.ones_like(scale), 0.1 * torch.log(scale) + 1.0)
            freqs_cos = freqs_cos * mscale
            freqs_sin = freqs_sin * mscale

        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


class _DyPEPosEmbed(torch.nn.Module):
    r"""
    Drop-in replacement for the positional embedding of `FluxTransformer2DModel` that applies the DyPE schedule to the
    spatial axes. The first axis (text positions) always uses plain RoPE, and the scheduled path only engages on a
    spatial axis when the number of patches on that axis (`max_pos + 1`) exceeds the number of patches at the trained
    resolution (`base_resolution // patch_size = 1024 // 16 = 64`). As a result, generation at or below the trained
    resolution is a no-op compared to the stock positional embedding.

    With `method="spectral"` the spatial axes use NTK-scaled frequencies together with SEGA's per-dimension spectral
    attention temperature, computed from the latent's Fourier spectrum (set per step via `set_spectral_data`).
    """

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        method: str = "yarn",
        dype: bool = True,
        spectral_alpha: float = 0.15,
        spectral_beta: float = 1.5,
        spectral_kappa: float = 0.08,
        spectral_min_mscale: float = 1.0,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.base_resolution = 1024
        self.patch_size = 16
        self.base_patches = self.base_resolution // self.patch_size
        self.training_res_pixels = self.base_resolution
        self.method = method
        self.dype = dype if method != "base" else False
        # SEGA parameters
        self.spectral_alpha = spectral_alpha
        self.spectral_beta = spectral_beta
        self.spectral_kappa = spectral_kappa
        self.spectral_min_mscale = spectral_min_mscale
        # DyPE runtime state
        self.current_timestep = 1.0
        # SEGA runtime state (set per step by the hook when method == "spectral")
        self._energy_profile_h = None
        self._energy_profile_w = None
        self._dynamic_spread = 0.0
        self._target_res_h = 0
        self._target_res_w = 0

    def set_timestep(self, timestep: float):
        """Set current timestep for DyPE. Timestep normalized to [0, 1] where 1 is pure noise."""
        self.current_timestep = timestep

    def set_spectral_data(self, energy_profile_h, energy_profile_w, dynamic_spread, target_res_h=0, target_res_w=0):
        """Set the per-step SEGA spectral data (called by the hook before each forward when `method == 'spectral'`)."""
        self._energy_profile_h = energy_profile_h
        self._energy_profile_w = energy_profile_w
        self._dynamic_spread = dynamic_spread
        self._target_res_h = target_res_h
        self._target_res_w = target_res_w

    def _compute_spectral_mscale(self, axis_idx, axis_dim, scale, device):
        # Per-dimension SEGA attention temperature for a spatial axis, or None to fall back to plain RoPE.
        energy_profile = self._energy_profile_h if axis_idx == 1 else self._energy_profile_w
        target_res = self._target_res_h if axis_idx == 1 else self._target_res_w

        m_ref = compute_base_mscale(
            target_res if target_res > 0 else 2 * self.training_res_pixels,
            self.training_res_pixels,
            coefficient=self.spectral_kappa,
        )
        if m_ref <= 1.0 + 1e-8:
            return None
        if energy_profile is None or self._dynamic_spread <= 0.0:
            # No spectral data available yet -> uniform reference magnitude.
            return torch.full((axis_dim // 2,), float(m_ref), device=device, dtype=torch.float32)

        exponents = torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim
        theta_ntk = self.theta * (scale ** (axis_dim / (axis_dim - 2)))
        freqs = 1.0 / (theta_ntk**exponents)
        return compute_spectral_allocation(
            energy_profile=energy_profile,
            freqs=freqs,
            base_mscale=m_ref,
            spread=self._dynamic_spread,
            alpha=self.spectral_alpha,
            beta=self.spectral_beta,
            min_mscale=self.spectral_min_mscale,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        freqs_dtype = maybe_adjust_dtype_for_device(torch.float64, ids.device)
        for i in range(n_axes):
            common_kwargs = {
                "dim": self.axes_dim[i],
                "pos": pos[:, i],
                "theta": self.theta,
                "repeat_interleave_real": True,
                "use_real": True,
                "freqs_dtype": freqs_dtype,
            }

            if i > 0:
                max_pos = pos[:, i].max().item()
                current_patches = max_pos + 1

                if current_patches > self.base_patches and self.method == "yarn":
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = _dype_rotary_pos_embed(
                        **common_kwargs,
                        yarn=True,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=self.base_patches,
                        dype=self.dype,
                        current_timestep=self.current_timestep,
                    )
                elif current_patches > self.base_patches and self.method == "spectral":
                    scale = current_patches / self.base_patches
                    ntk_factor = scale ** (self.axes_dim[i] / (self.axes_dim[i] - 2))
                    spectral_mscale = self._compute_spectral_mscale(i, self.axes_dim[i], scale, pos.device)
                    cos, sin = _dype_rotary_pos_embed(
                        **common_kwargs,
                        yarn=False,
                        ntk_factor=ntk_factor,
                        spectral_mscale=spectral_mscale,
                    )
                else:
                    cos, sin = _dype_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = _dype_rotary_pos_embed(**common_kwargs)

            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


# ----------------------------------------------------------------------------------------------
# Install DyPE on a transformer: swap pos_embed + a native forward-pre-hook feeding timestep(+spectral).
# (The concurrency-safe modular form of the validated DyPE hook; no HookRegistry dependency.)
# ----------------------------------------------------------------------------------------------
def _feed_spectral_data(pos_embed, hidden_states, img_ids):
    if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3 or not torch.is_tensor(img_ids):
        return
    h = int(img_ids[:, 1].max().item()) + 1
    w = int(img_ids[:, 2].max().item()) + 1
    if h * w > hidden_states.shape[1] or h < 2 or w < 2:
        return
    energy_h, energy_w = compute_axis_spectral_profiles(hidden_states, h, w, max(h // 2, 8), max(w // 2, 8))
    spread = compute_dynamic_spread(compute_spectral_energy_profile(hidden_states, h, w, max(h, w) // 2))
    pos_embed.set_spectral_data(energy_h, energy_w, spread,
                                target_res_h=h * pos_embed.patch_size, target_res_w=w * pos_embed.patch_size)


def _install_dype(transformer, method="yarn", dype=True, spectral_alpha=0.15, spectral_beta=1.5,
                  spectral_kappa=0.08, spectral_min_mscale=1.0):
    pe = transformer.pos_embed
    transformer.pos_embed = _DyPEPosEmbed(
        theta=pe.theta, axes_dim=pe.axes_dim, method=method, dype=dype,
        spectral_alpha=spectral_alpha, spectral_beta=spectral_beta,
        spectral_kappa=spectral_kappa, spectral_min_mscale=spectral_min_mscale)
    use_spectral = method == "spectral"

    def _feed(mod, args, kwargs):
        timestep = kwargs.get("timestep", None)
        hidden_states = kwargs.get("hidden_states", None)
        img_ids = kwargs.get("img_ids", None)
        if timestep is None and len(args) > 3:
            hidden_states = args[0] if hidden_states is None and len(args) > 0 else hidden_states
            timestep = args[3]
        if timestep is not None:
            if torch.is_tensor(timestep):
                timestep = timestep.flatten()[0]
            mod.pos_embed.set_timestep(float(timestep))
        if use_spectral and hidden_states is not None and img_ids is not None:
            _feed_spectral_data(mod.pos_embed, hidden_states, img_ids)

    handle = transformer.register_forward_pre_hook(_feed, with_kwargs=True)
    return pe, handle


# ----------------------------------------------------------------------------------------------
# DyPE community modular block — self-contained single-pass FLUX generation with DyPE + mu-cap.
# ----------------------------------------------------------------------------------------------
class DyPEBlock(ModularPipelineBlocks):
    """Training-free ultra-high-res text-to-image: installs DyPE on the transformer, caps the flow-match
    shift for >2K, and runs a single-pass FLUX denoise (DyPE fed per step via the native pre-hook)."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FLUX = "black-forest-labs/FLUX.1-Krea-dev"   # DyPE's validated checkpoint (realism-tuned)

    @property
    def expected_components(self):
        F_ = self._FLUX
        return [
            ComponentSpec("text_encoder", CLIPTextModel, pretrained_model_name_or_path=F_, subfolder="text_encoder"),
            ComponentSpec("tokenizer", CLIPTokenizer, pretrained_model_name_or_path=F_, subfolder="tokenizer"),
            ComponentSpec("text_encoder_2", T5EncoderModel, pretrained_model_name_or_path=F_, subfolder="text_encoder_2"),
            ComponentSpec("tokenizer_2", T5TokenizerFast, pretrained_model_name_or_path=F_, subfolder="tokenizer_2"),
            ComponentSpec("transformer", FluxTransformer2DModel, pretrained_model_name_or_path=F_, subfolder="transformer"),
            ComponentSpec("vae", AutoencoderKL, pretrained_model_name_or_path=F_, subfolder="vae"),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, pretrained_model_name_or_path=F_, subfolder="scheduler"),
        ]

    @property
    def inputs(self):
        return [
            InputParam("prompt", required=True),
            InputParam("prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=4.5),
            InputParam("method", default="yarn"),      # "yarn" (DyPE) or "spectral" (DyPE + SEGA)
            InputParam("dype", default=True),           # modulate YaRN by kappa=t^2
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    def _encode_prompt(self, components, prompt, prompt_2, max_sequence_length, device, dtype):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt if prompt_2 is None else ([prompt_2] if isinstance(prompt_2, str) else prompt_2)
        tok, te = components.tokenizer, components.text_encoder
        clip_ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True,
                       return_overflowing_tokens=False, return_length=False, return_tensors="pt").input_ids
        pooled = te(clip_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=te.dtype, device=device)
        tok2, te2 = components.tokenizer_2, components.text_encoder_2
        t5_ids = tok2(prompt_2, padding="max_length", max_length=max_sequence_length, truncation=True,
                      return_length=False, return_overflowing_tokens=False, return_tensors="pt").input_ids
        prompt_embeds = te2(t5_ids.to(device), output_hidden_states=False)[0].to(dtype=te2.dtype, device=device)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
        return prompt_embeds, pooled, text_ids

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)
        quant = vsf * 2
        num_channels_latents = tr.config.in_channels // 4
        height, width = int(bs.height), int(bs.width)
        if height % quant or width % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        nsteps = int(bs.num_inference_steps)
        guidance_embeds = tr.config.guidance_embeds
        batch_size = 1

        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, bs.prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = (torch.full([1], bs.guidance_scale, device=device, dtype=torch.float32).expand(batch_size)
                    if guidance_embeds else None)

        lh, lw = 2 * (height // quant), 2 * (width // quant)
        latents = randn_tensor((batch_size, num_channels_latents, lh, lw), generator=bs.generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, batch_size, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        # DyPE mu-cap: base_shift == max_shift so the >2K flow-match schedule does not collapse.
        max_shift = scheduler.config.get("max_shift", 1.15)
        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=max_shift)

        orig_pe, handle = _install_dype(tr, method=bs.method, dype=bool(bs.dype))
        vae.enable_tiling()   # required at 4K (the VAE encode/decode dominates memory otherwise)
        try:
            for t in timesteps:
                noise_pred = tr(hidden_states=latents, timestep=t.expand(latents.shape[0]).to(dtype) / 1000,
                                guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids, img_ids=img_ids, return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if bs.output_type == "latent":
                image = latents
            else:
                lat = FluxPipeline._unpack_latents(latents, height, width, vsf)
                lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
                from diffusers.image_processor import VaeImageProcessor
                image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)
        finally:
            handle.remove()
            tr.pos_embed = orig_pe

        bs.images = image
        self.set_block_state(state, bs)
        return components, state
