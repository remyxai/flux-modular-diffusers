# Tiled Creative Upscaler for FLUX — training-free detail-adding upscaling as a Modular Diffusers block.
#
# Method: tiled img2img refinement (no single paper — a workflow). Upscale with a classic resampler, cut
# overlapping tiles, refine each tile with FLUX img2img at low-moderate denoise, feather-blend the overlaps
# back into the canvas. The per-tile refine adds detail far beyond a plain resize, and tile-by-tile keeps
# peak activation memory flat. Reference: neuralwork/flux-tiled-upscaler (MIT); lineage: the Stable Diffusion
# tiled-upscaling workflow (diffusers/examples/community/tiled_upscaling.py) + SDEdit-style partial noising.
#
# Modular-Diffusers adaptation: a self-contained custom denoise loop (no attention/hooks/pos_embed swap —
# nothing is mutated on the transformer, so there is no seam to restore). Uses FLUX.1-dev
# (non-commercial license). Authored with AI assistance (Claude), validated by the Remyx AI team.

import math

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"

_DEFAULT_PROMPT = ("ultra detailed, sharp focus, high resolution, fine texture detail, photorealistic, "
                   "no blur, no compression artifacts")


def tile_grid(length, tile, overlap):
    """Tile origins (in px) covering [0, length), neighbouring tiles overlapping by >= `overlap`.

    Spreads (length - tile) evenly over the gaps instead of walking a fixed stride, and guarantees a
    real overlap between the last two tiles even when (length - tile) divides the stride evenly:
    2048px with 1024/128 tiles must not degrade to two merely *touching* tiles, since a touching seam
    gets no feathered blend at all, and that is precisely where seams come from.
    """
    if length <= tile:
        return [0]
    # Cap the overlap at half the tile: feather_weight cannot express more, and an overlap that large
    # would drive the stride toward 0 and explode the tile count for no blending benefit.
    overlap = min(int(overlap), max(tile // 2, 1))
    gaps = max(1, math.ceil(length / (tile - overlap)) - 1)     # gaps = tiles - 1
    stride = (length - tile) / gaps                             # spread the remainder evenly
    origins = [round(i * stride) for i in range(gaps + 1)]
    origins[-1] = length - tile                                 # last tile flush to the edge
    if len(origins) > 1 and origins[-1] - origins[-2] > tile - overlap - 1:
        origins[-2] = origins[-1] - (tile - overlap)            # guarantee a true overlap
    return origins


def feather_weight(size, overlap, device, dtype):
    """2-D separable ramp weight: 1 in the interior, ramping to 0 over `overlap` px on every edge.

    Ramps overlap on the two axes so the interior of a tile (where the tile is fully trustworthy)
    carries weight 1 and only the border fades — a seam between tiles lands where both are ~0.5.
    """
    o = min(int(overlap), size // 2)
    ramp = torch.ones(size, device=device, dtype=torch.float32)
    if o > 0:
        ramp[:o] = torch.linspace(0.0, 1.0, o, device=device)
        ramp[-o:] = torch.linspace(1.0, 0.0, o, device=device)
    return (ramp[:, None] * ramp[None, :]).to(dtype)


class TiledUpscalerBlock(ModularPipelineBlocks):
    """Training-free creative upscaling: resize, refine overlapping tiles with FLUX img2img, feather-blend."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FLUX = _FLUX
    _DEFAULT_PROMPT = _DEFAULT_PROMPT

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
            InputParam("image", required=True),            # low-res source: PIL / numpy RGB / path
            InputParam("prompt", default=None),            # detail prompt; None -> the built-in detail prompt
            InputParam("prompt_2", default=None),
            InputParam("scale", default=2),                # upscale factor (2 = 2x per side)
            InputParam("tile_size", default=1024),         # per-tile side, px (snapped to a /16 multiple)
            InputParam("tile_overlap", default=128),       # overlap between neighbouring tiles, px
            InputParam("denoise_strength", default=0.4),   # per-tile img2img strength (keep 0.3-0.5)
            InputParam("max_sequence_length", default=512),
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=3.5),
            InputParam("resample_filter", default="lanczos"),
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

    @staticmethod
    def _to_pil(x):
        from PIL import Image
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.fromarray(np.asarray(x)).convert("RGB")

    @staticmethod
    def _upscale(src, scale, quant, resample_filter):
        """Lanczos (default) resize by `scale`, then snap to a multiple of the VAE quantum (/16)."""
        resample = {"lanczos": 1, "bilinear": 2, "bicubic": 3}.get(str(resample_filter).lower(), 1)
        W, H = src.size
        new_w = max(quant, int(round(W * scale)) // quant * quant)
        new_h = max(quant, int(round(H * scale)) // quant * quant)
        return src.resize((new_w, new_h), resample=resample)

    def _refine_tile(self, tile_px, prompt_embeds, pooled, text_ids, guidance, timesteps, init_step,
                     tr, vae, scheduler, img_proc, vsf, num_channels_latents, device, dtype, generator):
        """One FLUX img2img pass over a single tile: encode -> partial noise -> denoise -> decode to pixels."""
        th, tw = tile_px.shape[-2], tile_px.shape[-1]
        lh, lw = th // vsf, tw // vsf
        x_init = vae.encode(tile_px.to(device=device, dtype=vae.dtype)).latent_dist.mode()
        x_init = ((x_init - vae.config.shift_factor) * vae.config.scaling_factor).to(dtype)
        latents = FluxPipeline._pack_latents(x_init, 1, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=dtype)
        latents = scheduler.scale_noise(latents, timesteps[init_step:init_step + 1], noise).to(dtype)

        for t in timesteps[init_step:]:
            noise_pred = tr(hidden_states=latents, timestep=t.expand(latents.shape[0]).to(dtype) / 1000,
                            guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids, img_ids=img_ids, joint_attention_kwargs=None,
                            return_dict=False)[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(dtype)

        lat = FluxPipeline._unpack_latents(latents, th, tw, vsf)
        lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
        pixels = vae.decode(lat.to(vae.dtype), return_dict=False)[0]        # [-1, 1]
        return img_proc.postprocess(pixels, output_type="pt")[0]            # [0, 1]

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)     # 8
        quant = vsf * 2                                          # 16
        num_channels_latents = tr.config.in_channels // 4       # 16
        guidance_embeds = tr.config.guidance_embeds

        # --- upscale the source ---
        src = self._to_pil(bs.image)
        scale = float(bs.scale)
        if scale <= 0:
            raise ValueError("scale must be > 0.")
        canvas_img = self._upscale(src, scale, quant, bs.resample_filter)
        H, W = canvas_img.height, canvas_img.width
        img_proc = VaeImageProcessor(vae_scale_factor=vsf)
        canvas_px = img_proc.preprocess(canvas_img, height=H, width=W)[0].to(device=device, dtype=torch.float32)

        # --- tiles (tile_size snapped down to a /16 multiple so encode/decode are exact; overlap is
        #     clamped to half the tile, which is the most feather_weight can express) ---
        tile = max(quant, int(bs.tile_size) // quant * quant)
        overlap = int(min(int(bs.tile_overlap), tile // 2))
        strength = float(min(max(float(bs.denoise_strength), 0.0), 1.0))
        n_steps = int(bs.num_inference_steps)
        init_step = int(round((1.0 - strength) * n_steps))
        if init_step >= n_steps:
            init_step = n_steps - 1                               # keep at least one denoise step
        rows = tile_grid(H, tile, overlap)
        cols = tile_grid(W, tile, overlap)

        # --- prompt (embeds are shared by every tile: encode once) ---
        prompt = bs.prompt if bs.prompt not in (None, "") else self._DEFAULT_PROMPT
        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = None
        if guidance_embeds:
            guidance = torch.full([1], float(bs.guidance_scale), device=device, dtype=torch.float32)

        # --- tile-by-tile img2img refine; feather-blend into the canvas (both accumulate, so overlap
        #     regions are the weight-normalized average of their tiles -> no seams, no halos) ---
        acc = torch.zeros(3, H, W, device=device, dtype=torch.float32)
        wsum = torch.zeros(1, H, W, device=device, dtype=torch.float32)
        n_tiles = len(rows) * len(cols)
        done = 0
        for y in rows:
            for x in cols:
                tile_px = canvas_px[:, y:y + tile, x:x + tile]
                scheduler.set_begin_index(0)
                sigmas = np.linspace(1.0, 1 / n_steps, n_steps)
                mu = calculate_shift((tile // 2) ** 2, scheduler.config.get("base_image_seq_len", 256),
                                     scheduler.config.get("max_image_seq_len", 4096),
                                     scheduler.config.get("base_shift", 0.5),
                                     scheduler.config.get("max_shift", 1.15))
                timesteps, _ = retrieve_timesteps(scheduler, n_steps, device, sigmas=sigmas, mu=mu)
                refined = self._refine_tile(
                    tile_px, prompt_embeds, pooled, text_ids, guidance, timesteps, init_step,
                    tr, vae, scheduler, img_proc, vsf, num_channels_latents, device, dtype, bs.generator)
                weight = feather_weight(tile, overlap, device, torch.float32).unsqueeze(0)
                acc[:, y:y + tile, x:x + tile] += refined.to(torch.float32) * weight
                wsum[:, y:y + tile, x:x + tile] += weight
                done += 1
                print(f"  [tile {done}/{n_tiles}] refined + blended")

        out_px = acc / wsum.clamp_min(1e-6)

        if bs.output_type == "latent":
            image = out_px
        else:
            image = img_proc.postprocess(out_px.unsqueeze(0), output_type=bs.output_type)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
