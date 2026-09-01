# Stitch for FLUX — training-free bounding-box position control as a Modular Diffusers block.
#
# Method: Stitch ("Stitch: Training-Free Position Control in Multimodal Diffusion Transformers",
# arXiv:2509.26644 — Bader, Pach, Bravo, Belongie, Akata). Given a global prompt plus a few
# (bounding-box, sub-prompt) regions, put THIS object at THIS location — no training, no detector,
# no LLM: the caller supplies `regions` like a layout-to-image API. Paper: FLUX GenEval-Position
# ~0.22 -> ~0.70. Three phases:
#
#   A  Region Binding (tau in [0,S)) — the transformer runs once per region, plus once for the
#      background prompt p0. Each object pass adds an additive joint-attention mask so the box
#      confines generation:
#         M(in  -> out) = -inf    inside-box image tokens don't attend outside the box
#         M(out -> txt) = -inf    outside-box image tokens don't attend to the sub-prompt text
#         M(txt -> out) = -inf    the sub-prompt text doesn't attend to outside-box image tokens
#      (in->in, in<->txt, txt<->txt stay open; the background pass gets NO mask.) Q/K/V projections
#      are untouched — only the additive mask M changes.
#   B  Cutout + composite (at tau=S) — read the text->image attention of one fixed head
#      (block 14, head 20 on FLUX.1-dev, paper Appendix F), average it over the non-pad text tokens,
#      sort descending and keep tokens until the cumulative mass reaches eta=0.95, then 2-D max-pool
#      the selected token mask with kernel kappa=5 into a binary foreground. Each object's foreground
#      latent tokens are written into the background latent at the box -> composite C.
#   C  Refine (tau in [S,T)) — denoise C with ONE pass on the full global prompt, no masks, to the
#      end. Decode.
#
# Modular-Diffusers adaptation — two proven seams, both restored in `finally`:
#   * Region Binding rides the HRDiT/KV-Edit attention seam: a FluxAttnProcessor subclass whose only
#     deviation from stock is one additive bias added to the attention logits, threaded through
#     `joint_attention_kwargs['stitch']` (named kwarg, no module globals, concurrency-safe). The
#     (joint x joint) bias is built ONCE per region per call, not per block per step.
#   * Cutout capture rides the PuLID block-index seam: the same processor records the raw
#     text->image attention when it runs inside `single_transformer_blocks[cutout_head[0]].attn`, at
#     the last Region-Binding step only. (A literal forward hook on the block cannot see the weights —
#     they exist only inside the attention call — so the capture is keyed to that block's own attn
#     module instead; same block, same step, one seam fewer.)
#
# With `regions` empty the joint_attention_kwargs stay None and no processor is installed, so the
# block is a bit-exact no-op against stock FluxPipeline (same seed/steps/prompt).
#
# STATUS: both seams follow the house patterns proven by the shipped blocks. The cutout head index is
# paper-reported for FLUX.1-dev and may differ in another build — it is exposed as `cutout_head`, and
# smoke.ipynb milestone C dumps the neighbouring heads so it can be re-picked without touching code.
# Authored with AI assistance (Claude), validated by the Remyx AI team; method credit to the Stitch
# authors. Uses FLUX.1-dev (non-commercial license).

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

_NEG = -1e4                  # additive -inf stand-in that survives bf16 (a true -inf turns every
                             # fully-masked row into NaN after softmax; text keys keep rows non-empty)
_DEFAULT_CUTOUT_HEAD = (14, 20)   # paper Appendix F, FLUX.1-dev


class StitchProcessor(FluxAttnProcessor):
    """FLUX joint attention with Stitch Region Binding, driven by
    `joint_attention_kwargs['stitch']` (None -> stock Flux attention, bit-exact no-op).

    payload (one dict per transformer call, shared by every attention module in it):
      bias        Tensor [1,1,n,n] — additive box bias for THIS pass (None -> stock attention)
      capture_id  id() of the attn module to record from, or None
      head        head index to record when this module is the capture target
      store       dict; on capture, store["a_txt_img"] = the raw text->image attention of `head`

    The joint sequence is [text (n_txt), image (n_img)] in BOTH stream layouts: the double blocks get
    two streams and concatenate them in that order, the single blocks receive them pre-concatenated
    with encoder_hidden_states=None. The image block therefore always starts at n_txt on the joint
    axis, and the box bias is built on that axis by the caller.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, stitch=None):
        if stitch is None or stitch["bias"] is None:
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask,
                                    image_rotary_emb)
        query, key, value, e_q, e_k, e_v = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
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

        if stitch["capture_id"] is not None and id(attn) == stitch["capture_id"]:
            # Raw text->image attention of one head, BEFORE the box bias is applied — the weights the
            # cutout threshold runs on, exactly as this block computed them. Rows = text queries,
            # columns = image keys, both on the joint [text, image] axis.
            qh = query[:, :, stitch["head"], :].float()
            kh = key[:, :, stitch["head"], :].float()
            a = torch.softmax(qh @ kh.transpose(-1, -2) * (qh.shape[-1] ** -0.5), dim=-1)
            stitch["store"]["a_txt_img"] = a[:, :stitch["n_txt"], stitch["n_txt"]:].detach()

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=stitch["bias"].to(device=query.device, dtype=query.dtype))
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            n_txt = encoder_hidden_states.shape[1]
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [n_txt, hidden_states.shape[1] - n_txt], dim=1)
            hidden_states = attn.to_out[0](hidden_states.contiguous())
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states.contiguous())
            return hidden_states, encoder_hidden_states
        return hidden_states       # single blocks: out_dim=None -> to_out is Identity


def _install_attn(transformer, proc):
    orig = {}
    for name, mod in transformer.named_modules():
        if name.endswith(".attn") and hasattr(mod, "processor"):
            orig[name] = mod.processor
            mod.processor = proc
    return orig


def _restore_attn(transformer, orig):
    for name, mod in transformer.named_modules():
        if name in orig:
            mod.processor = orig[name]


def _region_bias(inside, n_txt, n_img, dtype, device):
    """The additive joint-attention bias M for one Region-Binding pass.

    `inside` is a BoolTensor [n_img] marking the pass's box. Joint axis is [text, image], so the
    image block sits at [n_txt:]. Three rules (see the header):
      - every outside-box IMAGE key is blocked for every query   -> M(in->out) AND M(txt->out)
      - text keys are blocked for outside-box IMAGE queries      -> M(out->txt)
    Image->image inside, image<->text inside and text<->text stay open (bias 0). The first rule also
    blocks an outside-box token's own column (its self-attention), which is harmless: an outside-box
    query is a background token in an OBJECT pass, and it keeps the inside-box image keys open — so no
    row is ever fully blocked and softmax stays defined.
    """
    n = n_txt + n_img
    img_out = ~inside.to(device=device)
    m = torch.zeros(n, n, device=device, dtype=dtype)
    m[:, n_txt:] = torch.where(img_out, _NEG, torch.zeros_like(m[:, n_txt:]))
    rows = (img_out.nonzero(as_tuple=False).squeeze(-1) + n_txt).to(device=device)
    m[rows, :n_txt] = _NEG
    return m[None, None, :, :]


class StitchBlock(ModularPipelineBlocks):
    """Training-free position control on off-the-shelf FLUX: given a global prompt and a few
    (bounding-box, sub-prompt) regions, generate each object confined to its box for the first S
    steps (Region Binding), cut each object's foreground out with one attention head (Cutout),
    composite the foregrounds onto the background latent, then denoise unconstrained (Refine)."""

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
            InputParam("prompt", required=True),            # global prompt P (the whole scene)
            InputParam("regions", default=None),            # [{"box": [x0,y0,x1,y1], "prompt": str}]
                                                            # None/[] -> stock FLUX (bit-exact no-op)
            InputParam("background_prompt", default=""),    # "" -> derived neutral background
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=50),  # T (paper Table 1)
            InputParam("region_bind_steps", default=10),    # S: Region-Binding steps tau in [0,S)
            InputParam("guidance_scale", default=3.5),
            InputParam("cutout_eta", default=0.95),         # cumulative attention mass kept
            InputParam("cutout_kernel", default=5),         # kappa: 2-D max-pool kernel
            InputParam("cutout_head", default=_DEFAULT_CUTOUT_HEAD),   # (block, head) — tunable
            InputParam("max_sequence_length", default=512),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    # ------------------------------------------------------------------ text encoding
    def _encode_prompt(self, components, prompt, max_sequence_length, device, dtype):
        """CLIP pooled + T5 embeds for one prompt (the house _encode_prompt, single-prompt form)."""
        tok, te = components.tokenizer, components.text_encoder
        clip_ids = tok([prompt], padding="max_length", max_length=tok.model_max_length, truncation=True,
                       return_overflowing_tokens=False, return_length=False, return_tensors="pt").input_ids
        pooled = te(clip_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=te.dtype, device=device)
        tok2, te2 = components.tokenizer_2, components.text_encoder_2
        t5_ids = tok2([prompt], padding="max_length", max_length=max_sequence_length, truncation=True,
                      return_length=False, return_overflowing_tokens=False, return_tensors="pt").input_ids
        emb = te2(t5_ids.to(device), output_hidden_states=False)[0].to(dtype=te2.dtype, device=device)
        return emb, pooled

    def _encode_pass(self, components, prompt, max_sequence_length, device, dtype):
        """A pass's text conditioning plus its NON-PAD token count.

        The count matters only for the cutout: a sub-prompt padded to 512 contributes a handful of
        informative tokens, and averaging the cutout head's text->image attention over ~505 pad rows
        would swamp the signal the paper averages over.
        """
        emb, pooled = self._encode_prompt(components, prompt, max_sequence_length, device, dtype)
        ids = components.tokenizer_2([prompt], padding="max_length", max_length=max_sequence_length,
                                     truncation=True, return_length=False, return_overflowing_tokens=False,
                                     return_tensors="pt").input_ids[0]
        pad = components.tokenizer_2.pad_token_id
        n_real = int((ids != pad).sum().item()) if pad is not None else int(ids.shape[0])
        return emb, pooled, max(1, n_real)

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def _box_tokens(box, gh, gw):
        """BoolTensor [gh*gw]: True where the packed token's grid cell falls inside the normalized
        box. Row-major (r, c) over the packed grid, matching FluxPipeline._pack_latents. Half-open
        intervals, so abutting boxes tile the canvas with no token claimed twice."""
        x0, y0, x1, y1 = (float(v) for v in box)
        r = torch.arange(gh, dtype=torch.float32).unsqueeze(1)      # (gh,1)
        c = torch.arange(gw, dtype=torch.float32).unsqueeze(0)      # (1,gw)
        cy, cx = (r + 0.5) / gh, (c + 0.5) / gw
        return ((cx > x0) & (cx <= x1) & (cy > y0) & (cy <= y1)).reshape(-1)

    @staticmethod
    def _cumulative_foreground(w, eta):
        """The Cutout threshold: sort the per-token weights descending and keep the smallest prefix
        whose cumulative mass reaches `eta` of the total. Monotone in `eta`; `eta >= 1` saturates at
        the full-mass prefix (== every token), so it is clamped to just under 1."""
        w = w.detach().float().cpu()
        total = float(w.sum())
        if not total > 0:
            return torch.ones_like(w, dtype=torch.bool)
        eta = min(float(eta), 1.0 - 1e-6)
        order = torch.argsort(w, descending=True)
        csum = torch.cumsum(w[order], dim=0) / total
        n = int((csum < eta).sum().item()) + 1                 # first index that reaches eta
        keep = torch.zeros_like(w, dtype=torch.bool)
        keep[order[: max(1, min(n, w.numel()))]] = True
        return keep

    def _cutout(self, a_txt_img, n_real, inside, gh, gw, eta, kappa):
        """One region's binary foreground token mask from the captured text->image attention."""
        w = a_txt_img[0, :n_real, :].float().mean(dim=0)        # average over non-pad text tokens
        keep = self._cumulative_foreground(w, eta).to(a_txt_img.device)
        keep = keep & inside                                    # the pass confined it to the box
        if not bool(keep.any()):
            return inside
        grid = keep.reshape(1, 1, gh, gw).float()               # close holes / solidify the object
        grid = F.max_pool2d(grid, kernel_size=kappa, stride=1, padding=kappa // 2)
        fg = (grid.reshape(-1) > 0) & inside
        return fg if bool(fg.any()) else inside

    # ------------------------------------------------------------------ transformer call
    def _calc(self, tr, latents, t, guidance, pooled, emb, text_ids, img_ids, jak, dtype):
        return tr(hidden_states=latents, timestep=(t.expand(latents.shape[0]) / 1000).to(dtype),
                  guidance=guidance, pooled_projections=pooled, encoder_hidden_states=emb,
                  txt_ids=text_ids, img_ids=img_ids, joint_attention_kwargs=jak, return_dict=False)[0]

    def _denoise(self, tr, scheduler, latents, timesteps, start, guidance, pooled, emb, text_ids,
                 img_ids, jak_fn, dtype):
        """Plain FLUX denoise from timesteps[start:] (jak_fn(i) -> joint_attention_kwargs or None).

        Re-inits the scheduler's step index before each step: the Region-Binding steps interleave
        K+1 independent latent trajectories through ONE scheduler, and `step` only self-inits when
        step_index is None — without the reset, pass 2 would advance with pass 1's sigma.
        """
        for i in range(start, len(timesteps)):
            t = timesteps[i]
            v = self._calc(tr, latents, t, guidance, pooled, emb, text_ids, img_ids, jak_fn(i), dtype)
            scheduler._init_step_index(t)
            latents = scheduler.step(v, t, latents, return_dict=False)[0]
        return latents

    # ------------------------------------------------------------------ the block
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
        regions = list(bs.regions) if bs.regions else []
        T_steps = int(bs.num_inference_steps)
        S = max(0, min(int(bs.region_bind_steps), T_steps))      # tau in [0, S)
        eta, kappa = float(bs.cutout_eta), int(bs.cutout_kernel)
        cut_block, cut_head = (int(v) for v in bs.cutout_head)
        L = int(bs.max_sequence_length)
        guidance = (torch.full([1], float(bs.guidance_scale), device=device, dtype=torch.float32)
                    if tr.config.guidance_embeds else None)

        # --- prompts: the global prompt conditions pooled everywhere and Phases B/C's text; the
        # background pass gets p0; each region's pass gets its own sub-prompt. Separate passes, so
        # there are no text spans to route — each pass encodes exactly one prompt. ---
        global_emb, global_pooled = self._encode_prompt(components, bs.prompt, L, device, dtype)
        bg_prompt = str(bs.background_prompt) or f"{bs.prompt}, empty background"
        bg_emb, _, _ = self._encode_pass(components, bg_prompt, L, device, dtype)
        reg_emb, reg_pooled, reg_len = [], [], []
        for rg in regions:
            e, p, n = self._encode_pass(components, rg["prompt"], L, device, dtype)
            reg_emb.append(e); reg_pooled.append(p); reg_len.append(n)

        # --- packed latent + ids. Packed latents are (B, seq, 64): tokens on dim 1, so box mapping,
        # foreground selection and compositing all index dim 1. ---
        lh, lw = 2 * (H // quant), 2 * (W // quant)
        gh, gw = lh // 2, lw // 2                                 # packed image-token grid
        n_img = gh * gw
        base = randn_tensor((1, num_channels_latents, lh, lw), generator=bs.generator,
                            device=device, dtype=dtype)
        base = FluxPipeline._pack_latents(base, 1, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, gh, gw, device, dtype)

        # --- mu at the packed image-token count (gh*gw), not the pixel count ---
        sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
        cfg = scheduler.config
        mu = calculate_shift(n_img, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, sigmas=sigmas, mu=mu)

        def _txt_ids(n):
            return torch.zeros(n, 3, device=device, dtype=dtype)

        # ---------------------------------------------------------------- no-op path
        # regions empty: stock FLUX on the global prompt. No processor is installed and the joint
        # attention kwargs stay None, so this is bit-exact against FluxPipeline.
        if not regions:
            latents = self._denoise(tr, scheduler, base, timesteps, 0, guidance, global_pooled,
                                    global_emb, _txt_ids(global_emb.shape[1]), img_ids,
                                    lambda i: None, dtype)
            return self._decode(components, state, bs, latents, H, W, vsf, vae)

        # ---------------------------------------------------------------- validate boxes
        box_masks = []
        for k, rg in enumerate(regions):
            bm = self._box_tokens(rg["box"], gh, gw)
            if not bool(bm.any()):
                raise ValueError(f"regions[{k}] box {rg['box']} selects no image token at {W}x{H}px.")
            box_masks.append(bm)

        # One additive bias per region, built once (not per block per step): S steps x (19 double +
        # 38 single) attn modules would otherwise rebuild the same (n_txt+n_img)^2 tensor ~500x.
        biases = [_region_bias(bm, L, n_img, torch.float32, device) for bm in box_masks]

        # The attn module the cutout reads from: the single block the paper names.
        try:
            capture_attn = tr.single_transformer_blocks[cut_block].attn
            capture_id = id(capture_attn)
        except (AttributeError, IndexError):
            capture_id = None                                    # no single blocks -> full-box cutout

        # ---------------------------- Phase A: Region Binding, tau in [0,S)
        # K+1 passes per step from the SAME initial noise, so the object latents and the background
        # latent stay registered token-for-token for the composite. The passes are independent, so
        # they could equally run in one batch — kept sequential for VRAM (the brief's K<=3 note).
        # S == 0 (or 1) degenerates to compositing un-denoised noise, which is meaningless — clamp.
        S = max(1, S)
        traj = {"bg": base.clone()}
        fg_tokens = {}                       # region -> (foreground mask, kept latent tokens)
        orig = _install_attn(tr, StitchProcessor())
        try:
            store = {}

            def _jak(bias, capture=False):
                return {"stitch": {"bias": bias, "capture_id": capture_id if capture else None,
                                   "head": cut_head, "store": store, "n_txt": L}}

            last = S - 1                      # tau=S-1 is the last bound step: capture its attention
            for k in range(len(regions)):
                traj[k] = base.clone()
            for i in range(S):
                t = timesteps[i]
                capture = (i == last)
                v = self._calc(tr, traj["bg"], t, guidance, global_pooled, bg_emb,
                               _txt_ids(bg_emb.shape[1]), img_ids, None, dtype)
                scheduler._init_step_index(t)
                traj["bg"] = scheduler.step(v, t, traj["bg"], return_dict=False)[0]
                for k in range(len(regions)):
                    v = self._calc(tr, traj[k], t, guidance, reg_pooled[k], reg_emb[k],
                                   _txt_ids(reg_emb[k].shape[1]), img_ids,
                                   _jak(biases[k], capture=capture), dtype)
                    scheduler._init_step_index(t)
                    traj[k] = scheduler.step(v, t, traj[k], return_dict=False)[0]
                    if not capture:
                        continue
                    # ------------------ Phase B: cutout + composite (at tau=S)
                    a = store.pop("a_txt_img", None)
                    if a is None:
                        if capture_id is not None:
                            print(f"[stitch] cutout head ({cut_block},{cut_head}) produced no "
                                  f"attention for region {k}; using the whole box as foreground.")
                        fg_tokens[k] = (box_masks[k], traj[k])   # degenerate: box-wide foreground
                        continue
                    fg = self._cutout(a, reg_len[k], box_masks[k], gh, gw, eta, kappa)
                    fg_tokens[k] = (fg, traj[k][:, fg, :])       # the object's foreground tokens

            if len(fg_tokens) != len(regions):
                raise RuntimeError(f"cutout ran for {len(fg_tokens)}/{len(regions)} regions.")
            composite = traj["bg"].clone()
            for k in range(len(regions)):
                fg, toks = fg_tokens[k]
                composite[:, fg, :] = toks.to(composite.dtype)

            # ---------------------------- Phase C: refine on the FULL global prompt, no masks
            latents = self._denoise(tr, scheduler, composite, timesteps, S, guidance, global_pooled,
                                    global_emb, _txt_ids(global_emb.shape[1]), img_ids,
                                    lambda i: None, dtype)
        finally:
            _restore_attn(tr, orig)

        return self._decode(components, state, bs, latents, H, W, vsf, vae)

    # ------------------------------------------------------------------ decode
    def _decode(self, components, state, bs, latents, H, W, vsf, vae):
        from diffusers.image_processor import VaeImageProcessor
        if bs.output_type == "latent":
            image = latents
        else:
            lat = FluxPipeline._unpack_latents(latents, H, W, vsf)   # unpack with PIXEL dims
            lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]     # [B,C,H,W]
            image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)
        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
