# MultiDiffusion panorama for FLUX — training-free ultra-wide generation as a Modular Diffusers block.
#
# Method: MultiDiffusion ("MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation",
# arXiv:2302.08113 — Bar-Tal, Yariv, Lipman, Dekel). A wide latent is denoised by windows: at every step
# the transformer runs on overlapping crops of the shared canvas and each window's prediction is averaged
# back into that canvas (per-pixel mean over the windows covering the pixel), so the fused image is one
# globally coherent scene instead of stitched tiles. Training-free; no extra weights.
#
# Clean-room: implemented from the paper's description only. The reference repo
# (omerbt/MultiDiffusion) is unlicensed, so none of its code was read or copied; the method itself is
# not copyrightable. Lineage: this is the same fusion rule the regional-prompting card cites.
#
# Why window (and not just a wide latent): every window stays at FLUX's NATIVE resolution, so the
# flow-match sigma schedule sees the token count it was trained on. Denoising one 3072-wide latent
# instead stretches position ids far out of distribution and produces the "woven blob".
#
# Modular-Diffusers adaptation: a custom denoise loop. No attention/processor/pos_embed swap — the
# transformer is never mutated, so (unlike the other custom-loop blocks) there is nothing to restore.
# Windows are cut on the FULL packed latent and given their own per-window img_ids offset to the
# window's origin in the canvas, so every window's local coordinates are native while the global
# geometry comes from the offsets. Predictions are accumulated into sum/coverage buffers in fp32 and
# divided once per step.
#
# STATUS: the loop, window math and id bookkeeping follow the proven house custom-denoise pattern; the
# seam this gates on (per-window img_ids + scatter-average fusion) is exercised by smoke.ipynb, whose
# single-window case must be BIT-EXACT stock FLUX (window == canvas -> one covering window -> the
# average is the identity).
# Authored with AI assistance (Claude), validated by the Remyx AI team; method credit to the
# MultiDiffusion authors. Uses FLUX.1-dev (non-commercial license).

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"


def _window_starts(size, window, stride):
    """Overlapping window origins covering [0, size); the last one is right-aligned so the edge is covered."""
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, stride))
    last = size - window
    if starts[-1] != last:
        starts.append(last)          # right-align the final window rather than zero-padding the canvas
    return starts


class PanoramaBlock(ModularPipelineBlocks):
    """Training-free ultra-wide / panoramic FLUX: denoise one shared canvas through overlapping
    native-resolution windows and average every window's prediction back into it (MultiDiffusion)."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FLUX = _FLUX

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
            InputParam("width", default=3072),          # ultra-wide: 2048-4096 (multiple of 16)
            InputParam("window", default=1024),          # native-resolution tile side (keep at FLUX's 1024)
            InputParam("stride", default=512),           # window step; overlap = window - stride (>= 16px required)
            InputParam("window_height", default=None),   # None -> window (square tiles); lower for short/wide bands
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=3.5),
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
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)      # 8
        quant = vsf * 2                                           # 16
        num_channels_latents = tr.config.in_channels // 4        # 16

        H, W = int(bs.height), int(bs.width)
        if H % quant or W % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        nsteps = int(bs.num_inference_steps)
        guidance_embeds = tr.config.guidance_embeds

        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, bs.prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = (torch.full([1], float(bs.guidance_scale), device=device, dtype=torch.float32)
                    if guidance_embeds else None)

        # --- wide packed latent: one shared canvas for the whole panorama ---
        lh, lw = 2 * (H // quant), 2 * (W // quant)               # latent grid, 3072x1024 px -> 192x128
        gh, gw = lh // 2, lw // 2                                 # packed token grid, 96x64
        latents = randn_tensor((1, num_channels_latents, lh, lw),
                               generator=bs.generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, 1, num_channels_latents, lh, lw)

        # --- window layout on the packed grid (1 packed token = a 2x2 latent patch = a 16px cell) ---
        win = int(bs.window)
        wh = int(bs.window_height) if bs.window_height is not None else win
        if win % quant or wh % quant:
            raise ValueError(f"window/window_height must be multiples of {quant}.")
        # window size in packed tokens, clamped to the canvas: a window at least as large as the canvas
        # IS the canvas (one covering pass, i.e. plain wide generation — the no-op/naive control)
        wgh, wgw = min(wh // quant, gh), min(win // quant, gw)
        s = max(1, int(bs.stride) // quant)
        ys = _window_starts(gh, wgh, s)
        xs = _window_starts(gw, wgw, s)
        # overlap is only load-bearing where windows actually abut: on an axis with a single window the
        # stride is irrelevant, so validate it per axis (barely-overlapping windows seam and repeat)
        for axis, starts, wtok in (("height", ys, wgh), ("width", xs, wgw)):
            if len(starts) > 1 and wtok - s < 1:
                raise ValueError(f"stride must leave at least {quant}px of window overlap on {axis} "
                                 f"(got {int(bs.stride) - win}px); barely-overlapping windows seam "
                                 "and repeat content.")
        windows = [(y, x) for y in ys for x in xs]

        # per-window img_ids: a native window's ids shifted to the window's origin. Local coordinates
        # stay native (what the transformer was trained on); the offsets carry the global geometry.
        # A single covering window reproduces the stock full-canvas ids exactly.
        win_ids = []
        for y, x in windows:
            ids = FluxPipeline._prepare_latent_image_ids(None, wgh, wgw, device, dtype)
            ids[:, 0] += y
            ids[:, 1] += x
            win_ids.append(ids)

        # flat packed-token indices of each window (row-major, matching _pack_latents) for slicing the
        # canvas and scattering the window's prediction back
        win_idx = [((y + torch.arange(wgh, device=device))[:, None] * gw
                    + (x + torch.arange(wgw, device=device))[None, :]).reshape(-1)
                   for y, x in windows]

        # fp32 sum + coverage buffers; each window contributes 1 to the coverage of its tokens, and the
        # mean is taken once per step (this is the MultiDiffusion fusion rule)
        canvas_v = torch.zeros_like(latents, dtype=torch.float32)
        coverage = torch.zeros(1, 1, gh * gw, 1, device=device, dtype=torch.float32)
        for m in win_idx:
            coverage.index_add_(2, m, torch.ones(1, 1, m.numel(), 1, device=device))

        # --- sigma schedule at the WINDOW token count, not the canvas token count: every window is
        # native-resolution, so mu is shifted by the window's sequence length. Using the full canvas
        # length here would push the schedule far out of distribution (the naive-wide-gen failure).
        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        cfg = scheduler.config
        mu = calculate_shift(wgh * wgw, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)

        # no try/finally: the transformer is NEVER mutated (no processor/hook/pos_embed swap), so there
        # is no mutation to unwind on error — the other custom-loop blocks restore one in `finally`.
        for t in timesteps:
            canvas_v.zero_()
            for ids, m in zip(win_ids, win_idx):
                v = tr(hidden_states=latents[:, :, m], timestep=t.expand(1).to(dtype) / 1000, guidance=guidance,
                       pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                       img_ids=ids, return_dict=False)[0]
                canvas_v.index_add_(2, m, v.to(torch.float32))
            noise_pred = (canvas_v / coverage).to(dtype)          # per-pixel mean over covering windows
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        if bs.output_type == "latent":
            image = latents
        else:
            vae.enable_tiling()                   # required for a 4K-class decode (DyPE's rule)
            lat = FluxPipeline._unpack_latents(latents, H, W, vsf)
            lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
            from diffusers.image_processor import VaeImageProcessor
            image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
