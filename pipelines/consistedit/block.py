# ConsistEdit for FLUX — highly consistent, precise training-free visual editing as a
# Modular Diffusers block.
#
# Method: ConsistEdit ("ConsistEdit: Highly Consistent and Precise Training-free Visual Editing",
# arXiv:2510.17803 — Yin, Chen, Ni, Dai; SIGGRAPH Asia 2025). Reference: zxYin/ConsistEdit_Code
# (Apache-2.0). A training-free attention-control method designed for MM-DiT: the source image is
# rectified-flow INVERTED under the source prompt while every SINGLE-stream block caches the
# projected vision tokens of the trajectory; the edit denoises from that latent under the target
# prompt, fusing the cached source tokens into the target ones BEFORE attention (mask-guided
# pre-attention fusion). Three design points from the paper:
#   (1) vision-only control  — only the image-token parts of Q/K/V are touched; the text parts
#       always come from the target prompt (interfering with text tokens destabilizes generation);
#   (2) differentiated Q/K/V — Q/K carry STRUCTURE, V carries CONTENT: outside the edit mask the
#       source Q/K/V all replace the target's (structure AND colour held); inside the mask the
#       source Q/K are enforced for the first alpha*T steps (structure) while V stays
#       mask-blended (content);
#   (3) homogeneous layers   — control is applied to every step of every edited layer (FLUX's
#       single blocks, which carry the general generation information), not just late layers.
# Unlike FlowEdit (structure-preserving but the background drifts) and KV-Edit (pixel-precise
# background via cached K/V but no structure control inside the edit region), ConsistEdit gives
# smoothly adjustable structural consistency *within* the edited region while keeping the rest
# intact.
#
# Modular-Diffusers adaptation: the paper's FLUX variant operates on the SINGLE blocks, so the
# injection is an attention-processor swap installed ONLY on `single_transformer_blocks` (a subset
# of the proven HRDiT/KV-Edit attention seam). The cache/fusion state is threaded through
# `joint_attention_kwargs['consistedit']` (named kwarg — unnamed kwargs are dropped by
# FluxAttention; no module globals, concurrency-safe). Processors are restored in `finally`.
# With `consistency_strength=0` (or `mask=None`) the seam is disarmed — the kwarg is None, the
# processor falls through to stock attention — and the run is the bit-exact stock FLUX denoise.
#
# Memory: caching every block's tokens for every step would cost 20-80 GB (38 single blocks x
# 28 steps x 3 tensors x n_img x 3072), so the inversion instead keeps only the LATENT trajectory
# (a few MB) and the edit denoise REPLAYS the source branch one step at a time — capture step i
# from the cached latent, then fuse it into the target branch of the same step. Only one step of
# tokens (~0.7 GB at 512^2, ~2.9 GB at 1024^2, on CPU) is ever live, matching the reference's
# per-step pairing. Cost: three transformer passes per step instead of two.
#
# STATUS: follows the PROVEN attention seam; the capture/fusion path is gated by the e2e.ipynb
# spike (all-keep mask -> reconstructs the source; consistency OFF == the stock denoise).
# Authored with AI assistance (Claude), validated by the Remyx AI team; method credit to the
# ConsistEdit authors. Uses FLUX.1-dev (non-commercial license).

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


class ConsistEditFluxAttnProcessor(FluxAttnProcessor):
    """Flux single-block attention with ConsistEdit token fusion, driven by
    `joint_attention_kwargs['consistedit']` (None -> stock Flux attention, bit-exact).

    consistedit payload (per transformer call):
      mode:  "capture" — inversion pass: record the vision tokens of the (text+image) sequence,
                        then run STOCK attention (capture is read-only; the inversion trajectory
                        itself is the plain FLUX one)
             "fuse"    — edit denoise: blend the cached source vision tokens into the target ones
                        BEFORE attention, per the paper's mask-guided pre-attention fusion
      store: dict keyed (id(attn), step) -> (q_s, k_s, v_s), CPU tensors [B, n_seq, heads, head_dim]
      step:  scheduler step index (aligns the inversion capture with the denoise step)
      m:     BoolTensor [n_img] — True = EDIT region (target prompt drives), False = keep (source
             tokens replace). One value per packed image token (a 2x2 latent patch).
      struct: bool — True while the consistency window is on (step < alpha*T): the source Q/K are
             enforced EVERYWHERE, i.e. full structural preservation even inside the edit region.
             False afterwards: only V stays source-blended, so the target prompt may change shape.
      n_txt: text tokens prepended inside single-stream blocks (encoder_hidden_states=None there,
             the concatenation happens in FluxSingleTransformerBlock.forward)

    Only the vision slice [n_txt:] of Q/K/V is ever written: the text parts ([:n_txt]) always come
    from the target prompt — ConsistEdit's "vision-only" insight. Fusion happens on the normalized
    pre-RoPE tokens; positions are identical between the two passes (same img_ids), so blending
    before or after RoPE is equivalent and pre-attention is what the paper specifies.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, consistedit=None):
        if consistedit is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)
        if encoder_hidden_states is not None:   # double blocks — never installed there, stay stock
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        q, k, v, _, _, _ = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
        v = v.unflatten(-1, (attn.heads, -1))

        step = consistedit["step"]
        key = (id(attn), step)
        if consistedit["mode"] == "capture":
            consistedit["store"][key] = (q.to("cpu"), k.to("cpu"), v.to("cpu"))
            # stock math for the inversion trajectory itself (bit-exact; capture is read-only)
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        q_s, k_s, v_s = consistedit["store"][key]
        q_s = q_s.to(device=q.device, dtype=q.dtype)
        k_s = k_s.to(device=k.device, dtype=k.dtype)
        v_s = v_s.to(device=v.device, dtype=v.dtype)
        n_txt = consistedit["n_txt"]
        # per-token blend weight over the VISION tokens: 1 = SOURCE token, 0 = TARGET token.
        # torch.lerp(target, source, w): w=0 -> target, w=1 -> source, so the weight is the KEEP
        # indicator (m is the EDIT mask -> invert it).
        w = (~consistedit["m"]).to(device=q.device, dtype=q.dtype).view(1, -1, 1, 1)   # [1, n_img, 1, 1]
        # structure fusion (Eq. 5): while the consistency window is on, the source Q/K are enforced
        # in the edit region too — full structural preservation, shape cannot change; once the
        # window closes the edit region takes the target Q/K and only V stays source-blended.
        w_qk = torch.ones_like(w) if consistedit["struct"] else w
        # Blend only the VISION slice in place; the text tokens [:n_txt] stay the target's, so Q/K/V
        # keep the full joint length (n_txt + n_img) that image_rotary_emb and the single-stream
        # block expect. (Collapsing to the vision slice dropped the text tokens and mismatched RoPE.)
        q[:, n_txt:] = torch.lerp(q[:, n_txt:], q_s[:, n_txt:], w_qk)
        k[:, n_txt:] = torch.lerp(k[:, n_txt:], k_s[:, n_txt:], w_qk)
        # content fusion: V is source only outside the mask (source V everywhere caused colour
        # shifts; target V everywhere broke the non-edited content — the paper blends by mask).
        v[:, n_txt:] = torch.lerp(v[:, n_txt:], v_s[:, n_txt:], w)

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        hidden_states = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(q.dtype)
        # single-stream blocks: out_dim=None -> pre_only, to_out is Identity; proj_out lives in the
        # block, so the processor returns the raw attended tokens.
        return hidden_states


class ConsistEditBlock(ModularPipelineBlocks):
    """Highly consistent, precise training-free editing: RF-invert the source under its prompt while
    the single blocks cache their vision tokens, then denoise under the target prompt with those
    tokens fused in pre-attention — adjustable structural consistency inside the edit region, the
    rest of the image held intact."""

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
            InputParam("image", required=True),             # source image: PIL / numpy RGB / path
            InputParam("prompt", required=True),            # TARGET prompt (the edit)
            InputParam("source_prompt", required=True),     # describes the source image
            InputParam("mask", default=None),               # edit region (white=edit, black=keep);
                                                            # None -> whole image editable
            InputParam("consistency_strength", default=0.3),# alpha = ratio of steps the source Q/K are
                                                            # enforced in the edit region. Validated on
                                                            # FLUX: ~0.3 lands a clean edit while holding
                                                            # structure + background; 0.6-1.0 preserve so
                                                            # hard the shape barely changes (≈no edit);
                                                            # 0 = no control (stock denoise, can
                                                            # over-edit / look cartoonish). Raise toward
                                                            # 1 for subtle pose-preserving tweaks.
            InputParam("prompt_2", default=None),
            InputParam("source_prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=None),             # default: source image size (snapped to /16)
            InputParam("width", default=None),
            InputParam("T_steps", default=28),
            InputParam("guidance_scale", default=3.5),      # edit (target) guidance
            InputParam("src_guidance_scale", default=1.0),  # inversion guidance (low = faithful)
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
        alpha = float(bs.consistency_strength)
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

        # --- mask -> edit token mask (True = edit). Each packed token = a 2x2 latent patch
        # = a 16x16 pixel cell, row-major over (lh//2, lw//2) (matches _pack_latents). ---
        n_img = x_src_packed.shape[1]
        m = None
        if bs.mask is not None:
            mk = self._to_mask(bs.mask).resize((lw // 2, lh // 2))
            m = torch.from_numpy(np.asarray(mk) >= 128).flatten().to(device)   # white=edit, black=keep
            if not bool(m.any()):
                m = None   # nothing to edit -> whole image editable

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

        # alpha is "a ratio of steps for applying attention control": the source Q/K are enforced
        # in the edit region over the FIRST alpha*T steps of the denoise (the high-noise window,
        # where structure is decided). After that only V stays source-blended, letting the target
        # prompt change shape. alpha=1 -> structural preservation for the whole trajectory.
        n_struct = int(round(alpha * T_steps))
        arm = alpha > 0.0
        # no mask -> nothing to keep, the source token is the target token outside the edit region
        # only when the mask says so; with no mask every vision token is editable and the control
        # degenerates to the stock target denoise, so disarm the seam entirely (bit-exact).
        arm = arm and (m is not None)
        # vision tokens must line up 1:1 between the capture and the fuse pass
        arm = arm and (m.numel() == n_img)

        def _jak(mode, store, step, n_txt, struct=True):
            if not arm:
                return None   # disarmed -> stock attention, joint_attention_kwargs=None
            return {"consistedit": {"mode": mode, "store": store, "step": step, "m": m,
                                    "struct": bool(struct), "n_txt": n_txt}}

        # --- install the processor ONLY on the single blocks (the paper's FLUX target layer) ---
        # set_attn_processor takes a FULL dict, so: single blocks get the ConsistEdit processor,
        # double blocks keep exactly what they had (restored verbatim in `finally`).
        originals = dict(tr.attn_processors)
        single = getattr(tr, "single_transformer_blocks", None)
        single_names = set()
        if single is not None:
            for i in range(len(single)):
                single_names |= {n for n in originals if n.startswith(f"single_transformer_blocks.{i}.")}
        procs = dict(originals)
        for n in single_names:
            procs[n] = ConsistEditFluxAttnProcessor()
        store = {}

        def _step_i(i):
            scheduler._init_step_index(timesteps[i])
            return scheduler.sigmas[scheduler.step_index], scheduler.sigmas[scheduler.step_index + 1]

        try:
            tr.set_attn_processor(procs)

            # --- rectified-flow inversion (reference pass) under the source prompt, keeping the
            # latent trajectory so the fused pass can replay the same steps ---
            lat_src = [None] * (T_steps + 1)
            lat_src[T_steps] = x_src_packed.clone()
            for i in range(T_steps - 1, -1, -1):
                s_i, s_ip1 = _step_i(i)
                v = self._calc_v(tr, lat_src[i + 1], src_embeds, src_pooled, src_text_ids, img_ids, src_g,
                                 timesteps[i], dtype, _jak("capture", store, i, src_embeds.shape[1]))
                lat_src[i] = (lat_src[i + 1].to(torch.float32) + (s_i - s_ip1) * v.to(torch.float32)).to(dtype)

            # --- edit denoise under the target prompt: at each step re-run the SOURCE branch from
            # the cached trajectory to capture that step's vision tokens, then fuse them into the
            # target branch of the same step. Only ONE step of tokens is live at a time (~0.7-2.9 GB
            # for 512^2-1024^2 on CPU) instead of the whole trajectory's (20-80 GB). ---
            latents = lat_src[0].clone()
            for i, t in enumerate(timesteps):
                _step_i(i)          # re-point the scheduler at step i (step() advances it)
                _ = self._calc_v(tr, lat_src[i + 1], src_embeds, src_pooled, src_text_ids, img_ids, src_g, t, dtype,
                                 _jak("capture", store, i, src_embeds.shape[1]))
                v = self._calc_v(tr, latents, tar_embeds, tar_pooled, tar_text_ids, img_ids, tar_g, t, dtype,
                                 _jak("fuse", store, i, tar_embeds.shape[1], struct=i < n_struct))
                latents = scheduler.step(v, t, latents, return_dict=False)[0]
                store.clear()       # only ever one step of tokens is live
        finally:
            tr.set_attn_processor(originals)
            store.clear()
            lat_src = None

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
