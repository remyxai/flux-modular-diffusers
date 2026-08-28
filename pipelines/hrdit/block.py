# HRDiT for FLUX — training-free high-resolution as a Modular Diffusers community block.
#
# Method: NTK-aware RoPE scaling + Spatial Position Alignment (SPA) + structure-guided resolution
# ladder. Ported from HRDiT, "Training-Free High-Resolution Image Generation with Off-the-Shelf
# Diffusion Transformer Models" (arXiv:2608.07003), MIT reference https://github.com/zylwithxy/HRDiT.
# AI-assisted port (disclosed in the repo README).
#
# This is the concurrency-safe modular form of the community pipeline (huggingface/diffusers#14480):
# the per-stage NTK/SPA rope is delivered through `joint_attention_kwargs['hrdit_rope']` rather than a
# module global. The ATTENTION path (single NTK rope AND SPA variant-list averaging) is validated
# bit-exact (max|Δ| = 0) vs the community pipeline. The LADDER (HRDiTLadderBlock) is a near-verbatim
# transcription of the pipeline's stage loop — logic proven, plumbing substitutions flagged [VERIFY].
#
# STATUS: attention seam PROVEN; ladder/structure UNTESTED end-to-end (no local GPU). Publish PRIVATE
# to remyxai first, then E2E + per-stage parity-gate on GPU before making public / showing maintainers.

import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import (
    FluxAttnProcessor,
    FluxTransformer2DModel,
    _get_qkv_projections,
)
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
# NOTE[VERIFY-import]: confirm these export paths against the pinned diffusers in _requirements.
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam

_TRAIN_SEQ_LEN = 64 ** 2 + 512
_ROPE_THETA_DEFAULT = 10000


# ----------------------------------------------------------------------------------------------
# Vendored HRDiT math + structure helpers (single source of truth = the community pipeline).
# ----------------------------------------------------------------------------------------------
def _phi(x: torch.Tensor, n1: int, size: int) -> torch.Tensor:
    return torch.where(x < n1, torch.zeros_like(x), (x + 1 - n1 + size - 1) // size)


def build_bundle_id_variants(img_ids: torch.Tensor, group_num: int) -> List[torch.Tensor]:
    if group_num < 2:
        raise ValueError(f"`group_num` must be >= 2 for SPA, got {group_num}.")
    rows, cols = img_ids[:, 1].long(), img_ids[:, 2].long()
    s_row = max(1, math.ceil(rows.max().item() / (group_num - 1)))
    s_col = max(1, math.ceil(cols.max().item() / (group_num - 1)))

    def variant(n1_row: int, n1_col: int) -> torch.Tensor:
        ids = img_ids.clone()
        ids[:, 1] = _phi(rows, n1_row, s_row).to(img_ids.dtype)
        ids[:, 2] = _phi(cols, n1_col, s_col).to(img_ids.dtype)
        return ids

    variants = [variant(s_row, s_col)]
    variants += [variant(n, s_col) for n in range(1, s_row)]
    variants += [variant(s_row, m) for m in range(1, s_col)]
    return variants


def flux_rope(ids: torch.Tensor, axes_dim, theta: float, ntk_factor: float = 1.0):
    scaled_theta = theta * ntk_factor
    cos_out, sin_out = [], []
    for i, dim in enumerate(axes_dim):
        pos = ids[:, i].to(torch.float64)
        exps = torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device)[: dim // 2] / dim
        freqs = torch.outer(pos, 1.0 / (scaled_theta ** exps))
        cos_out.append(freqs.cos().repeat_interleave(2, dim=1).float())
        sin_out.append(freqs.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos_out, dim=-1), torch.cat(sin_out, dim=-1)


def butterworth_low_pass_filter_2d(height: int, width: int, ratio: float, device, order: int = 4) -> torch.Tensor:
    if ratio <= 0:
        return torch.zeros(1, 1, height, width, device=device)
    yy = (2.0 * torch.arange(height, device=device) / height - 1.0).view(height, 1)
    xx = (2.0 * torch.arange(width, device=device) / width - 1.0).view(1, width)
    d_square = yy ** 2 + xx ** 2
    return (1.0 / (1.0 + (d_square / ratio ** 2) ** order)).view(1, 1, height, width)


def split_low_freq(x: torch.Tensor, freq_filter: torch.Tensor) -> torch.Tensor:
    x_freq = torch.fft.fftshift(torch.fft.fft2(x.to(freq_filter.dtype)))
    return torch.fft.ifft2(torch.fft.ifftshift(x_freq * freq_filter)).real


def sharpen(image: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0, alpha: float = 1.0) -> torch.Tensor:
    blurred = gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return (alpha + 1.0) * image - alpha * blurred


# ----------------------------------------------------------------------------------------------
# Concurrency-safe attention processor. PROVEN Δ=0 vs the community pipeline (NTK + SPA modes).
# ----------------------------------------------------------------------------------------------
class HRDiTFluxAttnProcessor(FluxAttnProcessor):
    """Flux attention with HRDiT's NTK RoPE + SPA, driven by `joint_attention_kwargs['hrdit_rope']`
    (a single (cos,sin) tuple, or a LIST of SPA variants averaged). None -> stock Flux attention."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, hrdit_rope=None):
        if hrdit_rope is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        query, key, value, e_q, e_k, e_v = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            e_q = e_q.unflatten(-1, (attn.heads, -1))
            e_k = e_k.unflatten(-1, (attn.heads, -1))
            e_v = e_v.unflatten(-1, (attn.heads, -1))
            e_q = attn.norm_added_q(e_q)
            e_k = attn.norm_added_k(e_k)
            query = torch.cat([e_q, query], dim=1)
            key = torch.cat([e_k, key], dim=1)
            value = torch.cat([e_v, value], dim=1)
        head_dim, seq_len = query.shape[-1], query.shape[1]
        scale = math.sqrt(math.log(seq_len, _TRAIN_SEQ_LEN) / head_dim) if seq_len > 1 else head_dim ** -0.5
        value_t = value.transpose(1, 2).contiguous()
        ropes = hrdit_rope if isinstance(hrdit_rope, list) else [hrdit_rope]  # tuple=1 rope; list=SPA variants
        acc = None
        for rope in ropes:
            q_v = apply_rotary_emb(query, rope, sequence_dim=1).transpose(1, 2).contiguous()
            k_v = apply_rotary_emb(key, rope, sequence_dim=1).transpose(1, 2).contiguous()
            out = F.scaled_dot_product_attention(q_v, k_v, value_t, dropout_p=0.0, is_causal=False, scale=scale)
            acc = out if acc is None else acc + out
        hidden_states = (acc / len(ropes)).transpose(1, 2).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            n_txt = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [n_txt, hidden_states.shape[1] - n_txt], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states


# ----------------------------------------------------------------------------------------------
# HRDiT ladder block — near-verbatim transcription of the community pipeline's stage loop.
# Consumes prompt encodings from upstream FLUX text-encoder blocks; owns the multi-stage generation.
# self.* -> components.* / FluxPipeline utilities; the `_SPA_STATE` global -> per-call
# joint_attention_kwargs['hrdit_rope'].  [VERIFY-*] = plumbing to confirm on the first GPU E2E.
# ----------------------------------------------------------------------------------------------
def _stage_value(schedule, j, fill):
    if schedule is None:
        return fill
    return schedule[j] if j < len(schedule) else schedule[-1]


def _resolution_ladder(height, width, resolutions):
    if resolutions is None:
        target = max(height, width)
        ladder, side = [], min(1024, target)
        while side < target:
            ladder.append(side)
            side = min(side * 2, target)
        ladder.append(target)
        return ladder
    ladder = [int(r) for r in resolutions]
    if not ladder or any(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1)):
        raise ValueError(f"`resolutions` must be non-empty, strictly increasing, got {resolutions}.")
    return ladder


def _stage_dimensions(height, width, target, side, quant):
    if side >= target:
        return height, width
    sh = max(quant, int(round(height * side / target)) // quant * quant)
    sw = max(quant, int(round(width * side / target)) // quant * quant)
    return sh, sw


class HRDiTLadderBlock(ModularPipelineBlocks):
    """Full HRDiT generation: base stage (stock FLUX) then structure-guided NTK/SPA upscale stages.

    Assemble as [FLUX text-encoder blocks] -> HRDiTLadderBlock (this block owns denoise + VAE decode,
    so it replaces the stock before_denoise/denoise/decoder sub-blocks). See build_hrdit_flux_blocks().
    """

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0", "torchvision": ">=0.19.0"}

    # Self-contained: all components sourced from the base FLUX repo (like the Florence template),
    # so the published repo is just block.py + modular_config.json (no weights, no modular_model_index).
    _FLUX = "black-forest-labs/FLUX.1-dev"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        F = self._FLUX
        return [
            ComponentSpec("text_encoder", CLIPTextModel, pretrained_model_name_or_path=F, subfolder="text_encoder"),
            ComponentSpec("tokenizer", CLIPTokenizer, pretrained_model_name_or_path=F, subfolder="tokenizer"),
            ComponentSpec("text_encoder_2", T5EncoderModel, pretrained_model_name_or_path=F, subfolder="text_encoder_2"),
            ComponentSpec("tokenizer_2", T5TokenizerFast, pretrained_model_name_or_path=F, subfolder="tokenizer_2"),
            ComponentSpec("transformer", FluxTransformer2DModel, pretrained_model_name_or_path=F, subfolder="transformer"),
            ComponentSpec("vae", AutoencoderKL, pretrained_model_name_or_path=F, subfolder="vae"),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, pretrained_model_name_or_path=F, subfolder="scheduler"),
        ]

    @property
    def inputs(self) -> List[InputParam]:
        return [
            InputParam("prompt", required=True),
            InputParam("prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("resolutions", default=None),
            InputParam("num_inference_steps", default=30),
            InputParam("num_inference_steps_highres", default=None),
            InputParam("guidance_scale", default=3.5),
            InputParam("guidance_scale_highres", default=None),
            InputParam("ntk_factor", default=None),
            InputParam("spa_steps", default=None),
            InputParam("group_num", default=80),
            InputParam("alphas", default=None),
            InputParam("betas", default=None),
            InputParam("filter_ratio", default=0.2),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self) -> List[OutputParam]:
        return [OutputParam("images")]

    def _encode_prompt(self, components, prompt, prompt_2, max_sequence_length, device, dtype):
        """Transcribed FluxPipeline.encode_prompt: CLIP pooled + T5 embeds + zeros txt_ids."""
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

    def _flowmatch_step(self, scheduler, vsf, model_output, timestep, sample, mo, *, structure_on=False,
                        pred_x0_dict=None, height_dict=None, width_dict=None, batch_size=None,
                        num_channels_latents=None, target_height=None, target_width=None,
                        filter_ratio=0.0, alpha=0.0, beta=0.0):
        out_dtype = sample.dtype                       # cast the scheduler output back (matches monolith)
        sample = sample.to(torch.float32)
        sigma = scheduler.sigmas[scheduler.index_for_timestep(timestep)]
        pred_x0 = sample - model_output.to(torch.float32) * sigma
        original_pred_x0 = pred_x0
        if structure_on:
            up = FluxPipeline._unpack_latents
            pk = FluxPipeline._pack_latents
            x0 = up(pred_x0, target_height, target_width, vsf).float()
            lh, lw = x0.shape[-2], x0.shape[-1]
            ref = up(pred_x0_dict[timestep.item()], height_dict[timestep.item()], width_dict[timestep.item()], vsf).float()
            ref = F.interpolate(ref, (lh, lw), mode="bicubic", align_corners=False)
            ff = butterworth_low_pass_filter_2d(lh, lw, filter_ratio, x0.device)
            x0 = x0 + alpha * (split_low_freq(ref, ff) - split_low_freq(x0, ff))
            x0 = pk(x0, batch_size, num_channels_latents, lh, lw)
            ref = pk(ref, batch_size, num_channels_latents, lh, lw)
            model_output = (sample - x0) / (sigma + 1e-6)
            model_output_ref = (sample - ref) / (sigma + 1e-6)
            if mo["high"] is not None:
                model_output = model_output + beta * (mo["high"] + model_output_ref - mo["ref"] - model_output)
            mo["high"], mo["ref"] = model_output, model_output_ref
        else:
            model_output = model_output.to(torch.float32)
        prev = scheduler.step(model_output, timestep, sample, return_dict=False)[0]
        return prev.to(out_dtype), original_pred_x0

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)          # [VERIFY-vsf]
        quant = vsf * 2
        num_channels_latents = tr.config.in_channels // 4
        axes_dim, rope_theta = tr.pos_embed.axes_dim, tr.pos_embed.theta
        guidance_embeds = tr.config.guidance_embeds
        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, bs.prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype
        )
        batch_size = prompt_embeds.shape[0]

        def _guidance(scale):
            if not guidance_embeds:
                return None
            return torch.full([1], scale, device=device, dtype=torch.float32).expand(batch_size)

        height, width = int(bs.height), int(bs.width)
        if height % quant or width % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        ladder = _resolution_ladder(height, width, bs.resolutions)
        target = max(height, width)
        ntk_sched = bs.ntk_factor if bs.ntk_factor is not None else [4.0, 10.0]
        spa_sched = bs.spa_steps if bs.spa_steps is not None else [3, 0]
        guid_hr = bs.guidance_scale_highres if bs.guidance_scale_highres is not None else [4.5, 6.0]
        steps_hr = bs.num_inference_steps_highres if bs.num_inference_steps_highres is not None else [17, 10]
        alpha_sched = bs.alphas if bs.alphas is not None else [1.0, 0.25]
        beta_sched = bs.betas if bs.betas is not None else [0.5, 0.5]
        nsteps = int(bs.num_inference_steps)

        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        base_grid = ladder[0] // quant
        mu = calculate_shift(
            base_grid * base_grid,
            scheduler.config.get("base_image_seq_len", 256),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.5),
            scheduler.config.get("max_shift", 1.15),
        )
        jak = {}  # base stage disarmed: no 'hrdit_rope' -> processor is stock FLUX
        original_procs = dict(tr.attn_processors)
        tr.set_attn_processor(HRDiTFluxAttnProcessor())
        pred_x0_dict, height_dict, width_dict = {}, {}, {}
        try:
            # --- base stage (stock RoPE) ---
            base_h = base_w = ladder[0]
            lh, lw = 2 * (base_h // quant), 2 * (base_w // quant)
            latents = randn_tensor((batch_size, num_channels_latents, lh, lw), generator=bs.generator, device=device, dtype=dtype)
            latents = FluxPipeline._pack_latents(latents, batch_size, num_channels_latents, lh, lw)
            image_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)
            timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)
            cur_h, cur_w = base_h, base_w
            for t in timesteps:
                noise_pred = tr(hidden_states=latents, timestep=t.expand(latents.shape[0]).to(latents.dtype) / 1000,
                                guidance=_guidance(bs.guidance_scale), pooled_projections=pooled,
                                encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=image_ids,
                                joint_attention_kwargs=jak, return_dict=False)[0]
                latents, pred_x0 = self._flowmatch_step(scheduler, vsf, noise_pred, t, latents, None)
                pred_x0_dict[t.item()], height_dict[t.item()], width_dict[t.item()] = pred_x0, cur_h, cur_w

            # --- structure-guided NTK/SPA upscale stages ---
            for stage in range(1, len(ladder)):
                j = stage - 1
                sh, sw = _stage_dimensions(height, width, target, ladder[stage], quant)
                gh, gw = sh // quant, sw // quant
                stage_ntk = float(_stage_value(ntk_sched, j, 1.0))
                stage_spa = int(_stage_value(spa_sched, j, 0))
                stage_guid = _guidance(float(_stage_value(guid_hr, j, bs.guidance_scale)))
                stage_steps = int(_stage_value(steps_hr, j, max(1, round(nsteps * 0.5))))
                a0 = float(_stage_value(alpha_sched, j, 0.0))
                b0 = float(_stage_value(beta_sched, j, 0.0))

                dec = FluxPipeline._unpack_latents(latents, cur_h, cur_w, vsf)
                dec = (dec / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(dec.to(vae.dtype), return_dict=False)[0]
                image = sharpen(F.interpolate(image, (sh, sw), mode="bicubic", align_corners=False))
                enc = vae.encode(image.to(vae.dtype).to(vae.device)).latent_dist.mode()
                enc = (enc - vae.config.shift_factor) * vae.config.scaling_factor
                latents = FluxPipeline._pack_latents(enc, batch_size, num_channels_latents, enc.shape[-2], enc.shape[-1]).to(dtype)
                image_ids = FluxPipeline._prepare_latent_image_ids(None, gh, gw, device, dtype)

                base_rope = flux_rope(torch.cat([text_ids, image_ids], 0), axes_dim, rope_theta, ntk_factor=stage_ntk)
                if stage_spa > 0:
                    variant_ropes = [flux_rope(torch.cat([text_ids, v], 0), axes_dim, rope_theta, ntk_factor=stage_ntk)
                                     for v in build_bundle_id_variants(image_ids, int(bs.group_num))]
                else:
                    variant_ropes = None

                retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)
                stage_ts = scheduler.timesteps[-stage_steps:]
                noise = randn_tensor(latents.shape, generator=bs.generator, device=device, dtype=latents.dtype)
                latents = scheduler.scale_noise(latents, stage_ts[:1], noise).to(dtype)
                mo = {"high": None, "ref": None}
                for i, t in enumerate(stage_ts):
                    rope = (variant_ropes if (stage_spa > 0 and i < stage_spa) else base_rope)
                    noise_pred = tr(hidden_states=latents, timestep=t.expand(latents.shape[0]).to(latents.dtype) / 1000,
                                    guidance=stage_guid, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                                    txt_ids=text_ids, img_ids=image_ids,
                                    joint_attention_kwargs={"hrdit_rope": rope}, return_dict=False)[0]
                    decay = (stage_steps - i) / stage_steps
                    latents, pred_x0 = self._flowmatch_step(
                        scheduler, vsf, noise_pred, t, latents, mo, structure_on=True,
                        pred_x0_dict=pred_x0_dict, height_dict=height_dict, width_dict=width_dict,
                        batch_size=batch_size, num_channels_latents=num_channels_latents,
                        target_height=sh, target_width=sw, filter_ratio=float(bs.filter_ratio),
                        alpha=a0 * decay, beta=b0 * decay)
                    pred_x0_dict[t.item()], height_dict[t.item()], width_dict[t.item()] = pred_x0, sh, sw
                cur_h, cur_w = sh, sw

            if bs.output_type == "latent":
                image = latents
            else:
                lat = FluxPipeline._unpack_latents(latents, cur_h, cur_w, vsf)
                lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
                # [VERIFY-postprocess]: use the pipeline's image_processor if exposed on components.
                from diffusers.image_processor import VaeImageProcessor
                image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)
        finally:
            tr.set_attn_processor(original_procs)

        bs.images = image
        self.set_block_state(state, bs)
        return components, state
