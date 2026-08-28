# StyleAligned for FLUX — training-free style-consistent set generation as a Modular Diffusers block.
#
# Method: StyleAligned ("Style Aligned Image Generation via Shared Attention", arXiv:2312.02133,
# Hertz, Voynov, Fruchter, Cohen-Or). A batch of prompts is generated so the images share ONE style, by
# sharing attention across the batch: each item's Q/K are AdaIN-aligned to a reference (batch item 0) and
# each item additionally attends to the reference's K/V. Training-free; no extra weights.
#
# Modular-Diffusers adaptation: a FLUX joint-attention processor (delegates to stock when off = bit-exact
# no-op; the seam is validated in attn_spike.ipynb) installed for a batched denoise, threaded via
# joint_attention_kwargs. Authored with AI assistance (Claude), validated by Remyx AI; method credit to the
# StyleAligned authors. Uses FLUX.1-dev (non-commercial).

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"


def _adain(feat, ref, eps=1e-6):
    """Align feat's per-(batch,head,dim) statistics over the token axis to ref's."""
    fm, fs = feat.mean(dim=2, keepdim=True), feat.std(dim=2, keepdim=True)
    rm, rs = ref.mean(dim=2, keepdim=True), ref.std(dim=2, keepdim=True)
    return (feat - fm) / (fs + eps) * rs + rm


class StyleAlignedProcessor(FluxAttnProcessor):
    """FLUX joint attention with StyleAligned sharing: each item keeps its OWN query, AdaIN-aligns its keys to
    item-0, and attends to item-0's K/V (transfers style, not subject). Applied only in later layers
    (self.active). `style_share` off or inactive -> stock attention (bit-exact no-op)."""
    active = True

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, style_share=None):
        if not style_share or not self.active:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = q.unflatten(-1, (attn.heads, -1)); k = k.unflatten(-1, (attn.heads, -1)); v = v.unflatten(-1, (attn.heads, -1))
        q = attn.norm_q(q); k = attn.norm_k(k)
        if encoder_hidden_states is not None:
            eq = eq.unflatten(-1, (attn.heads, -1)); ek = ek.unflatten(-1, (attn.heads, -1)); ev = ev.unflatten(-1, (attn.heads, -1))
            eq = attn.norm_added_q(eq); ek = attn.norm_added_k(ek)
            q = torch.cat([eq, q], dim=1); k = torch.cat([ek, k], dim=1); v = torch.cat([ev, v], dim=1)
        q_r = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1).transpose(1, 2)   # (B,heads,S,D)
        k_r = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1).transpose(1, 2)
        v_t = v.transpose(1, 2)
        # StyleAligned v2: keep each item's OWN query (preserves subject); AdaIN only the keys to item-0's
        # statistics, then attend to own + item-0's K/V (transfers style, not content).
        k_r = _adain(k_r, k_r[:1].expand_as(k_r))
        share_k = k_r[:1].expand_as(k_r); share_v = v_t[:1].expand_as(v_t)
        out = F.scaled_dot_product_attention(q_r, torch.cat([k_r, share_k], 2), torch.cat([v_t, share_v], 2))
        hidden_states = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        if encoder_hidden_states is not None:
            n_txt = encoder_hidden_states.shape[1]
            enc, hidden_states = hidden_states.split_with_sizes([n_txt, hidden_states.shape[1] - n_txt], dim=1)
            hidden_states = attn.to_out[0](hidden_states.contiguous()); hidden_states = attn.to_out[1](hidden_states)
            enc = attn.to_add_out(enc.contiguous())
            return hidden_states, enc
        return hidden_states


def _install(transformer, proc_cls, skip_frac=0.0):
    attns = [(n, m) for n, m in transformer.named_modules() if n.endswith(".attn") and hasattr(m, "processor")]
    start = int(skip_frac * len(attns))            # skip early layers (they carry subject/structure)
    orig = {}
    for i, (name, mod) in enumerate(attns):
        orig[name] = mod.processor
        p = proc_cls(); p.active = (i >= start); mod.processor = p
    return orig


def _restore(transformer, orig):
    for name, mod in transformer.named_modules():
        if name in orig:
            mod.processor = orig[name]


class StyleAlignedFluxBlock(ModularPipelineBlocks):
    """Training-free style-consistent set generation: a batch of prompts share one style via StyleAligned
    shared attention on FLUX (installed for the denoise, restored after)."""

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
            InputParam("prompts", required=True),          # list of prompts sharing one style (item 0 = style anchor)
            InputParam("prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=3.5),
            InputParam("style_share", default=True),        # False = independent generations (no-op)
            InputParam("share_start_frac", default=0.8),     # share only the last ~20% of layers (style, not subject)
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
        H, W = int(bs.height), int(bs.width)
        if H % quant or W % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        prompts = bs.prompts if isinstance(bs.prompts, (list, tuple)) else [bs.prompts]
        batch = len(prompts)
        nsteps = int(bs.num_inference_steps)
        guidance_embeds = tr.config.guidance_embeds

        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, list(prompts), bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = (torch.full([1], bs.guidance_scale, device=device, dtype=torch.float32).expand(batch)
                    if guidance_embeds else None)

        lh, lw = 2 * (H // quant), 2 * (W // quant)
        latents = randn_tensor((batch, num_channels_latents, lh, lw), generator=bs.generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, batch, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        cfg = scheduler.config
        mu = calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)

        jkw = {"style_share": bool(bs.style_share)}
        orig = _install(tr, StyleAlignedProcessor, skip_frac=float(bs.share_start_frac))
        try:
            for t in timesteps:
                noise_pred = tr(hidden_states=latents, timestep=t.expand(batch).to(dtype) / 1000, guidance=guidance,
                                pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                                img_ids=img_ids, joint_attention_kwargs=jkw, return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if bs.output_type == "latent":
                image = latents
            else:
                lat = FluxPipeline._unpack_latents(latents, H, W, vsf)
                lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
                from diffusers.image_processor import VaeImageProcessor
                image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)
        finally:
            _restore(tr, orig)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
