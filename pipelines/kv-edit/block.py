# KV-Edit for FLUX — training-free text editing with pixel-precise background preservation,
# as a Modular Diffusers block.
#
# Method: KV-Edit ("KV-Edit: Training-Free Image Editing for Precise Background Preservation",
# arXiv:2502.17363 — Zhu, Zhang, Shao, Tang). Reference: Xilluill/KV-Edit (Apache-2.0).
# The source image is rectified-flow INVERTED under the source prompt while every attention block
# caches the normalized K/V of the BACKGROUND image tokens (mask = keep region). The edit denoise
# then runs from the inverted latent under the target prompt with the cached background K/V
# substituted in place — mathematically identical to the reference's "concat background K/V memory
# with foreground content" (softmax over keys is permutation-invariant), so background tokens are
# PRESERVED rather than re-synthesized. Unlike FlowEdit (structure-preserving), KV-Edit keeps the
# unedited background pixel-precise.
#
# Modular-Diffusers adaptation: attention-processor swap threading the cached background K/V
# through `joint_attention_kwargs['kv_edit']` (the HRDiT seam — named kwarg, no module globals,
# concurrency-safe). Processors are restored in `finally`. With no background tokens (mask=None,
# i.e. whole-image edit) the seam is disarmed and attention is the bit-exact stock FLUX path.
#
# STATUS: follows the PROVEN HRDiT attention seam; the capture/substitute K/V path is gated by the
# e2e.ipynb spike (empty mask -> reconstructs the source). Cached K/V live on CPU (O(n_bg) per
# block per step) and are sliced back per denoise step.
# Authored with AI assistance (Claude), validated by the Remyx AI team; method credit to the
# KV-Edit authors. Uses FLUX.1-dev (non-commercial license).

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"


class KVEditFluxAttnProcessor(FluxAttnProcessor):
    """Flux attention with KV-Edit background K/V memory, driven by
    `joint_attention_kwargs['kv_edit']` (None -> stock Flux attention, bit-exact).

    kv_edit payload (per transformer call):
      mode:  "capture"    — inversion pass: record the bg-token K/V, then run STOCK attention
             "substitute" — edit denoise: swap the cached bg K/V in place of the current ones
      store: dict keyed (id(attn), step) -> (k_bg, v_bg), CPU tensors [B, n_bg, heads, head_dim]
      step:  scheduler step index (aligns the inversion capture with the denoise step)
      bg:    BoolTensor [n_img] — True = background token (K/V kept from the source image)
      n_txt: text tokens prepended inside single-stream blocks (encoder_hidden_states=None)

    Substitution is pre-RoPE on the normalized K/V; positions are identical between the inversion
    and denoise passes (same img_ids), and in-place substitution attends over the same key/value
    SET as the reference's concat form, so the two are equivalent.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, kv_edit=None):
        if kv_edit is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        query, key, value, e_q, e_k, e_v = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        # image-stream offset: joint blocks project the image stream alone (offset 0); single
        # blocks receive cat([text, image]) so the image tokens start after the text tokens.
        off = 0 if encoder_hidden_states is not None else kv_edit["n_txt"]
        idx = kv_edit["bg"].nonzero(as_tuple=False).squeeze(-1) + off
        if kv_edit["mode"] == "capture":
            kv_edit["store"][(id(attn), kv_edit["step"])] = (key[:, idx].to("cpu"), value[:, idx].to("cpu"))
            # stock math for the inversion trajectory itself (bit-exact; capture is read-only)
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        k_bg, v_bg = kv_edit["store"][(id(attn), kv_edit["step"])]
        key[:, idx] = k_bg.to(device=key.device, dtype=key.dtype)
        value[:, idx] = v_bg.to(device=value.device, dtype=value.dtype)
        if encoder_hidden_states is not None:
            e_q = attn.norm_added_q(e_q.unflatten(-1, (attn.heads, -1)))
            e_k = attn.norm_added_k(e_k.unflatten(-1, (attn.heads, -1)))
            e_v = e_v.unflatten(-1, (attn.heads, -1))
            query = torch.cat([e_q, query], dim=1)
            key = torch.cat([e_k, key], dim=1)
            value = torch.cat([e_v, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            n_txt = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [n_txt, hidden_states.shape[1] - n_txt], dim=1)
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        # single-stream blocks: out_dim=None -> to_out is Identity; proj_out lives in the block
        return hidden_states


class KVEditBlock(ModularPipelineBlocks):
    """Training-free text editing with pixel-precise background preservation: RF-invert the source
    under the source prompt while caching background-token K/V, then denoise under the target
    prompt with the cached background K/V substituted — only the masked edit region changes."""

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
            InputParam("image", required=True),            # source image: PIL / numpy RGB / path
            InputParam("prompt", required=True),           # TARGET prompt (the edit)
            InputParam("source_prompt", required=True),    # describes the source image
            InputParam("mask", default=None),              # edit region (white=edit, black=keep);
                                                           # None -> whole-image edit (seam disarmed)
            InputParam("prompt_2", default=None),
            InputParam("source_prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=None),            # default: source image size (snapped to /16)
            InputParam("width", default=None),
            InputParam("T_steps", default=28),
            InputParam("guidance_scale", default=3.5),     # target/denoise guidance
            InputParam("src_guidance_scale", default=1.0), # inversion guidance (low = faithful inversion)
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
        import numpy as _np
        return Image.fromarray(_np.asarray(x)).convert("RGB")

    @staticmethod
    def _to_mask(x):
        from PIL import Image
        if isinstance(x, str):
            return Image.open(x).convert("L")
        if isinstance(x, Image.Image):
            return x.convert("L")
        import numpy as _np
        return Image.fromarray(_np.asarray(x)).convert("L")

    def _calc_v(self, tr, latents, prompt_embeds, pooled, text_ids, img_ids, guidance, t, dtype, jak):
        return tr(hidden_states=latents, timestep=(t.expand(latents.shape[0]) / 1000).to(dtype),
                  guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                  txt_ids=text_ids, img_ids=img_ids, joint_attention_kwargs=jak, return_dict=False)[0]

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)     # 8
        quant = vsf * 2                                          # 16
        num_channels_latents = tr.config.in_channels // 4       # 16
        T_steps = int(bs.T_steps)
        guidance_embeds = tr.config.guidance_embeds

        # --- source image -> latents ---
        src = self._to_pil(bs.image)
        H = int(bs.height) if bs.height else (src.height // quant) * quant
        W = int(bs.width) if bs.width else (src.width // quant) * quant
        img_proc = VaeImageProcessor(vae_scale_factor=vsf)
        pixels = img_proc.preprocess(src, height=H, width=W).to(device=device, dtype=vae.dtype)
        x_src = vae.encode(pixels).latent_dist.sample(generator=bs.generator)
        x_src = (x_src - vae.config.shift_factor) * vae.config.scaling_factor
        x_src = x_src.to(dtype)
        batch = x_src.shape[0]
        lh, lw = x_src.shape[2], x_src.shape[3]
        x_src_packed = FluxPipeline._pack_latents(x_src, batch, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        # --- mask -> background token mask (True = keep). Each packed token = a 2x2 latent patch
        # = a 16x16 pixel cell, row-major over (lh//2, lw//2) (matches FluxPipeline._pack_latents).
        bg = None
        if bs.mask is not None:
            m = self._to_mask(bs.mask).resize((lw // 2, lh // 2))
            bg = torch.from_numpy(np.asarray(m) < 128).flatten().to(device)   # white=edit, black=keep
            if not bool(bg.any()):
                bg = None   # nothing to keep -> whole-image edit; seam stays disarmed (bit-exact)

        # --- prompts (source for inversion, target for the edit) ---
        src_embeds, src_pooled, src_text_ids = self._encode_prompt(
            components, bs.source_prompt, bs.source_prompt_2, int(bs.max_sequence_length), device, dtype)
        tar_embeds, tar_pooled, tar_text_ids = self._encode_prompt(
            components, bs.prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)

        if guidance_embeds:
            src_g = torch.full([1], float(bs.src_guidance_scale), device=device, dtype=torch.float32).expand(batch)
            tar_g = torch.full([1], float(bs.guidance_scale), device=device, dtype=torch.float32).expand(batch)
        else:
            src_g = tar_g = None

        # --- timesteps (capped mu, same schedule as generation) ---
        sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
        cfg = scheduler.config
        mu = calculate_shift(x_src_packed.shape[1], cfg.get("base_image_seq_len", 256),
                             cfg.get("max_image_seq_len", 4096), cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, sigmas=sigmas, mu=mu)

        def _jak(mode, store, step, n_txt):
            if bg is None:
                return None   # disarmed -> stock attention
            return {"kv_edit": {"mode": mode, "store": store, "step": step, "bg": bg, "n_txt": n_txt}}

        original_procs = dict(tr.attn_processors)
        tr.set_attn_processor(KVEditFluxAttnProcessor())
        store = {}
        try:
            # --- rectified-flow inversion (reference pass) under the source prompt, caching the
            # background-token K/V of every attention block at every step ---
            latents = x_src_packed.clone()
            for i in range(T_steps - 1, -1, -1):
                t = timesteps[i]
                scheduler._init_step_index(t)
                s_i = scheduler.sigmas[scheduler.step_index]
                s_ip1 = scheduler.sigmas[scheduler.step_index + 1]
                v = self._calc_v(tr, latents, src_embeds, src_pooled, src_text_ids, img_ids, src_g, t, dtype,
                                 _jak("capture", store, i, src_embeds.shape[1]))
                latents = (latents.to(torch.float32) + (s_i - s_ip1) * v.to(torch.float32)).to(dtype)

            # --- edit denoise under the target prompt, background K/V substituted per step ---
            for i, t in enumerate(timesteps):
                v = self._calc_v(tr, latents, tar_embeds, tar_pooled, tar_text_ids, img_ids, tar_g, t, dtype,
                                 _jak("substitute", store, i, tar_embeds.shape[1]))
                latents = scheduler.step(v, t, latents, return_dict=False)[0]
        finally:
            tr.set_attn_processor(original_procs)
            store.clear()

        # --- decode ---
        if bs.output_type == "latent":
            image = latents
        else:
            lat = FluxPipeline._unpack_latents(latents, H, W, vsf)
            lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
            image = img_proc.postprocess(image, output_type=bs.output_type)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
