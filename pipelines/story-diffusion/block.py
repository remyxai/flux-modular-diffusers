# StoryDiffusion for FLUX — training-free consistent-character generation as a Modular Diffusers block.
#
# Method: StoryDiffusion ("Consistent Self-Attention for Long-Range Image and Video Generation",
# arXiv:2405.01434). A batch of frames of the SAME character stays consistent because each frame's
# self-attention also attends to the OTHER frames' tokens (Consistent Self-Attention) — training-free.
# Scope: image consistency + a comic compositor. Out of scope: the video motion module; PhotoMaker (real-face) = a v2.
#
# Modular-Diffusers adaptation: a FLUX joint-attention processor (delegates to stock when off = bit-exact
# no-op; seam validated in attn_spike.ipynb) that shares K/V across the batch, threaded via
# joint_attention_kwargs. Authored with AI assistance (Claude), validated by Remyx AI. FLUX.1-dev (non-commercial).

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


class ConsistentSelfAttnProcessor(FluxAttnProcessor):
    """Consistent Self-Attention: each batch frame also attends to all frames' K/V. None -> stock (no-op)."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, story_share=None, share_ratio=1.0):
        if not story_share:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = q.unflatten(-1, (attn.heads, -1)); k = k.unflatten(-1, (attn.heads, -1)); v = v.unflatten(-1, (attn.heads, -1))
        q = attn.norm_q(q); k = attn.norm_k(k)
        if encoder_hidden_states is not None:
            eq = eq.unflatten(-1, (attn.heads, -1)); ek = ek.unflatten(-1, (attn.heads, -1)); ev = ev.unflatten(-1, (attn.heads, -1))
            eq = attn.norm_added_q(eq); ek = attn.norm_added_k(ek)
            q = torch.cat([eq, q], dim=1); k = torch.cat([ek, k], dim=1); v = torch.cat([ev, v], dim=1)
        q_r = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1).transpose(1, 2)   # (B,h,S,D)
        k_r = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1).transpose(1, 2)
        v_t = v.transpose(1, 2)
        B, h, S, D = k_r.shape
        if B > 1:                                          # own K/V + a SAMPLED pool of all frames' tokens
            pool_k = k_r.permute(1, 0, 2, 3).reshape(h, B * S, D)
            pool_v = v_t.permute(1, 0, 2, 3).reshape(h, B * S, D)
            if share_ratio < 1.0:                          # sample a fraction (keeps scene diversity)
                n = max(1, int(share_ratio * B * S))
                idx = torch.randperm(B * S, device=pool_k.device)[:n]
                pool_k = pool_k[:, idx]; pool_v = pool_v[:, idx]
            pk = pool_k.unsqueeze(0).expand(B, h, pool_k.shape[1], D)
            pv = pool_v.unsqueeze(0).expand(B, h, pool_v.shape[1], D)
            out = F.scaled_dot_product_attention(q_r, torch.cat([k_r, pk], 2), torch.cat([v_t, pv], 2))
        else:
            out = F.scaled_dot_product_attention(q_r, k_r, v_t)
        hidden_states = out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        if encoder_hidden_states is not None:
            n_txt = encoder_hidden_states.shape[1]
            enc, hidden_states = hidden_states.split_with_sizes([n_txt, hidden_states.shape[1] - n_txt], dim=1)
            hidden_states = attn.to_out[0](hidden_states.contiguous()); hidden_states = attn.to_out[1](hidden_states)
            enc = attn.to_add_out(enc.contiguous())
            return hidden_states, enc
        return hidden_states


def _install(transformer, proc_cls):
    orig = {}
    for name, mod in transformer.named_modules():
        if name.endswith(".attn") and hasattr(mod, "processor"):
            orig[name] = mod.processor
            mod.processor = proc_cls()
    return orig


def _restore(transformer, orig):
    for name, mod in transformer.named_modules():
        if name in orig:
            mod.processor = orig[name]


def _comic(panels, captions, cols=2):
    from PIL import Image, ImageDraw, ImageFont
    n = len(panels); cols = min(cols, n); rows = (n + cols - 1) // cols
    S, pad, cap = 512, 16, 46
    sheet = Image.new("RGB", (cols * S + (cols + 1) * pad, rows * (S + cap) + (rows + 1) * pad), "white")
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        f = ImageFont.load_default()
    for i, im in enumerate(panels):
        r, c = divmod(i, cols); x = pad + c * (S + pad); y = pad + r * (S + cap) + r * pad
        sheet.paste(im.resize((S, S)), (x, y))
        cap_txt = captions[i] if i < len(captions) and captions[i] else ""
        if cap_txt:
            d.text((x + 6, y + S + 8), cap_txt[:60], fill="black", font=f)
    return sheet


class StoryDiffusionFluxBlock(ModularPipelineBlocks):
    """Training-free consistent-character generation: a set of frames of the same character stay consistent
    via Consistent Self-Attention, optionally composed into a comic sheet."""

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
            InputParam("character_prompt", required=True),      # e.g. "a young woman with red hair, freckles"
            InputParam("scene_prompts", required=True),          # list; "#caption" suffix -> panel caption; "[NC]" prefix -> no character
            InputParam("prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=3.5),
            InputParam("story_share", default=True),             # False -> independent frames (no-op)
            InputParam("share_ratio", default=0.3),              # fraction of cross-frame tokens shared (keeps scenes distinct)
            InputParam("share_start_frac", default=0.35),        # share only after this fraction of steps (scene set first)
            InputParam("comic_layout", default=None),            # None or "grid"
            InputParam("comic_cols", default=2),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    def _encode_prompt(self, components, prompts, prompt_2, max_sequence_length, device, dtype):
        prompt_2 = prompts if prompt_2 is None else prompt_2
        tok, te = components.tokenizer, components.text_encoder
        clip_ids = tok(prompts, padding="max_length", max_length=tok.model_max_length, truncation=True,
                       return_tensors="pt").input_ids
        pooled = te(clip_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=te.dtype, device=device)
        tok2, te2 = components.tokenizer_2, components.text_encoder_2
        t5 = tok2(prompt_2, padding="max_length", max_length=max_sequence_length, truncation=True,
                  return_tensors="pt").input_ids
        emb = te2(t5.to(device), output_hidden_states=False)[0].to(dtype=te2.dtype, device=device)
        text_ids = torch.zeros(emb.shape[1], 3, device=device, dtype=dtype)
        return emb, pooled, text_ids

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1); quant = vsf * 2
        num_channels_latents = tr.config.in_channels // 4
        H, W = int(bs.height), int(bs.width)
        if H % quant or W % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        nsteps = int(bs.num_inference_steps)

        # parse scenes: "#caption" -> caption; "[NC]" -> scene without the character
        prompts, captions = [], []
        for sc in bs.scene_prompts:
            body, _, cap = sc.partition("#")
            captions.append(cap.strip())
            body = body.strip()
            if body.startswith("[NC]"):
                prompts.append(body[4:].strip())
            else:
                prompts.append(f"{bs.character_prompt}, {body}")
        batch = len(prompts)

        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, prompts, bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = (torch.full([1], bs.guidance_scale, device=device, dtype=torch.float32).expand(batch)
                    if tr.config.guidance_embeds else None)

        lh, lw = 2 * (H // quant), 2 * (W // quant)
        latents = randn_tensor((batch, num_channels_latents, lh, lw), generator=bs.generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, batch, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        cfg = scheduler.config
        mu = calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)

        start_step = int(float(bs.share_start_frac) * len(timesteps))
        orig = _install(tr, ConsistentSelfAttnProcessor)
        try:
            for i, t in enumerate(timesteps):
                jkw = {"story_share": bool(bs.story_share) and i >= start_step, "share_ratio": float(bs.share_ratio)}
                noise_pred = tr(hidden_states=latents, timestep=t.expand(batch).to(dtype) / 1000, guidance=guidance,
                                pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                                img_ids=img_ids, joint_attention_kwargs=jkw, return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            lat = FluxPipeline._unpack_latents(latents, H, W, vsf)
            lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
            decoded = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
            from diffusers.image_processor import VaeImageProcessor
            panels = VaeImageProcessor(vae_scale_factor=vsf).postprocess(decoded, output_type="pil")
        finally:
            _restore(tr, orig)

        if bs.comic_layout:
            sheet = _comic(panels, captions, cols=int(bs.comic_cols))
            bs.images = [sheet] + panels
        else:
            bs.images = panels
        self.set_block_state(state, bs)
        return components, state
