"""FluxLens — run a training-free intervention *recipe* on FLUX, and sweep configurations.

The 14 shipped pipelines here are hand-authored ``block.py`` files. Most of them are the SAME move: install
:class:`FluxIntervention`, capture/replace/share/bias some ``q/k/v`` at some blocks for some steps. FluxLens makes
that move *data*: a **recipe** (a dict / YAML row in ``recipes/``) declares ``(site, schedule, op, params)``, and
``run(recipe, inputs)`` interprets it into ``flux_modular`` payloads. So each method becomes a config, and new
configurations (including compositions the source papers never tried) are just new rows — explored with ``sweep``.

Validated on an A100 (2026-09): ``freecontrol`` as a recipe reproduces the shipped block (depth-corr 0.883 ≈ 0.89);
``structure_appearance`` (freecontrol.capture+replace MERGED with appearance.redux) composes in one ``run`` path.

Scope note — this is the **FLUX adapter**. The op menu is model-agnostic but the adapter (token layout, RoPE,
plumbing, block enumeration) is per-checkpoint; an SD3.5 spike showed the adapter ports but Q-replace itself is
rope-specific, so method transfer is verified per-checkpoint, not assumed. See ``recipes/README.md``.
"""

import numpy as np
import torch

from .attention import (
    flux_intervention, last_single_attn_ids, edge_attn_ids, PAYLOAD_KEY,
)
from .plumbing import pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift

_FLUX = "black-forest-labs/FLUX.1-dev"
_REDUX = "black-forest-labs/FLUX.1-Redux-dev"


class FluxLens:
    """FLUX adapter + recipe interpreter. Construct once (loads FLUX.1-dev; Redux lazily), then ``run`` recipes."""

    def __init__(self, model=_FLUX, dtype=torch.bfloat16, device="cuda", steps=28, height=1024, width=1024):
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(model, torch_dtype=dtype).to(device)
        self.tr, self.vae = self.pipe.transformer, self.pipe.vae
        self.device, self.dtype = device, dtype
        self.steps, self.H, self.W = steps, height, width
        self.n_img = (height // 16) * (width // 16)
        self._redux = None
        torch.set_grad_enabled(False)   # inference only; raw tr() capture would else retain a grad graph -> OOM

    # ---- adapter: site selection (FLUX-specific) ----
    def _sites(self, spec):
        spec = spec or {}
        if spec.get("stream") == "single":
            return last_single_attn_ids(self.tr, int(spec.get("last_n", 25)))
        if spec.get("stream") == "both":
            return edge_attn_ids(self.tr, int(spec.get("edge", 2)))
        return set()

    # ---- adapter: conditioning ----
    def _prompt_embeds(self, text, L=512):
        pe, pooled, _ = self.pipe.encode_prompt(prompt=text, prompt_2=text, device=self.device,
                                                num_images_per_prompt=1, max_sequence_length=L)
        return pe, pooled

    def _load_redux(self):
        if self._redux is None:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
            self._redux = (
                SiglipVisionModel.from_pretrained(_REDUX, subfolder="image_encoder", torch_dtype=self.dtype).to(self.device),
                SiglipImageProcessor.from_pretrained(_REDUX, subfolder="feature_extractor"),
                ReduxImageEncoder.from_pretrained(_REDUX, subfolder="image_embedder", torch_dtype=self.dtype).to(self.device),
            )
        return self._redux

    def _redux_embed(self, img, scale=1.0):
        ienc, feat, emb = self._load_redux()
        fe = feat(images=img, return_tensors="pt").to(self.device)
        sig = ienc(pixel_values=fe.pixel_values.to(self.dtype)).last_hidden_state
        return emb(sig).image_embeds.to(self.device, self.dtype) * float(scale)

    def _encode_x0(self, img):
        x = self.pipe.image_processor.preprocess(img, height=self.H, width=self.W).to(self.device, self.vae.dtype)
        z = self.vae.encode(x).latent_dist.mode()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return pack_latents(z, 1, self.vae.config.latent_channels, 2 * (self.H // 16), 2 * (self.W // 16))

    # ---- op: LCD one-step image-Q capture (trailing-n_img, length-agnostic) ----
    def _capture_q(self, img, sigma, timestep, ids):
        bank = {}
        xt = ((1.0 - float(sigma)) * self._encode_x0(img)).to(self.dtype)
        pe, pooled = self._prompt_embeds("")
        tids = torch.zeros(pe.shape[1], 3, device=self.device, dtype=self.dtype)
        iid = prepare_latent_image_ids(self.H // 16, self.W // 16, self.device, self.dtype)
        g = torch.full((1,), 3.5, device=self.device, dtype=self.dtype)
        ts = torch.full((1,), float(timestep) / 1000, device=self.device, dtype=self.dtype)

        def cap(q, k, v, off, attn, pl):
            if id(attn) in ids:
                bank[id(attn)] = q[:, -self.n_img:].detach()
            return q, k, v

        with flux_intervention(self.tr):
            self.tr(hidden_states=xt, timestep=ts, guidance=g, pooled_projections=pooled, encoder_hidden_states=pe,
                    txt_ids=tids, img_ids=iid,
                    joint_attention_kwargs={PAYLOAD_KEY: {"pre_rope": cap, "n_txt": pe.shape[1]}}, return_dict=False)
        return bank

    # ---- the interpreter: dispatch on recipe kind, one entry point ----
    @torch.no_grad()
    def run(self, recipe, inputs, seed=0, **overrides):
        """Interpret a recipe dict against ``inputs`` -> a PIL image.

        Two conditioning kinds are wired: the default capture-Q / Redux / replace-Q path
        (freecontrol, appearance, structure_appearance) and ``regional`` (multi-prompt + bias mask).
        """
        P = {**recipe.get("params", {}), **overrides}
        mode = recipe.get("run", "default")     # default | regional | batch | edit
        if mode == "regional":
            return self._run_regional(recipe, inputs, P, seed)
        if mode == "batch":
            return self._run_batch(recipe, inputs, P, seed)
        if mode == "edit":
            return self._run_edit(recipe, inputs, P, seed)
        return self._run_default(recipe, inputs, P, seed)

    def _run_default(self, recipe, inputs, P, seed):
        ids = self._sites(recipe.get("site", {"stream": "single", "last_n": int(P.get("last_n", 25))}))

        bank = {}
        cap = recipe.get("capture")
        if cap:
            bank = self._capture_q(inputs[cap.get("source", "ref_structure")],
                                   P.get("sigma", cap.get("sigma", 0.35)),
                                   int(cap.get("timestep", 661)), ids)

        pe, pooled = self._prompt_embeds(inputs["prompt"])
        cond = recipe.get("condition")
        enc = pe
        if cond and cond.get("kind") == "redux":
            rx = self._redux_embed(inputs[cond.get("source", "ref_appearance")], scale=P.get("redux_scale", 1.0))
            enc = torch.cat([rx, pe], dim=1)

        replace = any(o.get("op") == "replace_q" for o in recipe.get("ops", []))
        S = float(P.get("S", 0.3))
        cut = int(S * self.steps)
        st = {"step": 0}

        def cb(pp, i, t, kw):
            st["step"] = i + 1
            return kw

        def pre_rope(q, k, v, off, attn, pl):
            if replace and st["step"] < cut and id(attn) in ids and id(attn) in bank:
                q = q.clone(); q[:, -self.n_img:] = bank[id(attn)].to(q)
            return q, k, v

        jkw = {PAYLOAD_KEY: {"pre_rope": pre_rope, "n_txt": int(enc.shape[1])}} if replace else None
        with flux_intervention(self.tr):
            return self.pipe(prompt_embeds=enc, pooled_prompt_embeds=pooled, height=self.H, width=self.W,
                             num_inference_steps=self.steps, guidance_scale=float(P.get("guidance", 6.5)),
                             generator=torch.Generator("cpu").manual_seed(int(seed)),
                             joint_attention_kwargs=jkw, callback_on_step_end=cb).images[0]

    # ---- regional: multi-prompt + a joint-attention bias mask (op: bias) ----
    @torch.no_grad()
    def _run_regional(self, recipe, inputs, P, seed):
        """inputs = {base_prompt, regions:[{prompt, bbox:[x0,y0,x1,y1] normalized}]}. Faithful to
        pipelines/regional-prompting: each image token attends to its region's prompt (+ base), routed by an
        additive mask. Position-agnostic (bias), so this is also the best cross-model-transfer candidate."""
        cond = recipe.get("condition") or {}
        L = int(P.get("region_seq_len", cond.get("region_seq_len", 128)))
        exclusive = bool(P.get("exclusive", cond.get("exclusive", True)))
        strength = float(P.get("isolate_strength", cond.get("isolate_strength", 0.0)))
        regions = list(inputs["regions"])

        base_pe, base_pooled = self._prompt_embeds(inputs["base_prompt"], L=L)
        embs, spans, off = [base_pe], [(0, base_pe.shape[1])], base_pe.shape[1]
        for rg in regions:
            e, _ = self._prompt_embeds(rg["prompt"], L=L)
            embs.append(e); spans.append((off, off + e.shape[1])); off += e.shape[1]
        prompt_embeds = torch.cat(embs, dim=1)
        n_txt = prompt_embeds.shape[1]
        gh = gw = self.H // 16
        n_img = gh * gw

        region_of = np.full(n_img, -1, dtype=np.int64)
        ys = (np.arange(n_img) // gw + 0.5) / gh
        xs = (np.arange(n_img) % gw + 0.5) / gw
        for r, rg in enumerate(regions):
            x0, y0, x1, y1 = rg["bbox"]
            region_of[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = r
        joint = n_txt + n_img
        NEG = -1e4
        bias = torch.zeros(joint, joint, dtype=torch.float32)
        text_allow = torch.zeros(n_img, n_txt, dtype=torch.bool)
        assigned = torch.from_numpy(region_of >= 0)
        bs0, bs1 = spans[0]
        if exclusive:
            text_allow[~assigned, bs0:bs1] = True
        else:
            text_allow[:, bs0:bs1] = True
        for r, (s0, s1) in enumerate(spans[1:]):
            text_allow[torch.from_numpy(region_of == r), s0:s1] = True
        bias[n_txt:, :n_txt] = torch.where(text_allow, 0.0, NEG)
        if strength > 0:
            ro = torch.from_numpy(region_of)
            img_allow = torch.eye(n_img, dtype=torch.bool)
            for r in range(len(regions)):
                sel = ro == r
                img_allow |= (sel.unsqueeze(1) & sel.unsqueeze(0))
            un = ro == -1
            img_allow |= un.unsqueeze(0) | un.unsqueeze(1)
            bias[n_txt:, n_txt:] = torch.where(img_allow, 0.0, -strength)
        mask = bias.unsqueeze(0).unsqueeze(0).to(device=self.device, dtype=self.dtype)
        region_bias = lambda q, k, ntxt, attn, pl: mask

        jkw = {PAYLOAD_KEY: {"bias": region_bias, "n_txt": n_txt}}
        with flux_intervention(self.tr):
            return self.pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=base_pooled,
                             height=self.H, width=self.W, num_inference_steps=self.steps,
                             guidance_scale=float(P.get("guidance", 3.5)),
                             generator=torch.Generator("cpu").manual_seed(int(seed)),
                             joint_attention_kwargs=jkw).images[0]

    # ---- consistency: batched frames sharing K/V across the batch (op: post_rope share) ----
    @torch.no_grad()
    def _run_batch(self, recipe, inputs, P, seed):
        """inputs = {character_prompt, scene_prompts:[str]} ("#cap" -> caption, "[NC]" prefix -> no character).
        Faithful to pipelines/story-diffusion: each frame also attends to a sampled pool of ALL frames' K/V
        (post-rope) after ``share_start_frac`` of steps -> a consistent character across scenes. Returns a list."""
        from diffusers.utils.torch_utils import randn_tensor
        cond = recipe.get("condition") or {}
        share_ratio = float(P.get("share_ratio", cond.get("share_ratio", 0.3)))
        start_frac = float(P.get("share_start_frac", cond.get("share_start_frac", 0.35)))
        share_on = bool(P.get("story_share", True))

        prompts = []
        for sc in inputs["scene_prompts"]:
            body = sc.partition("#")[0].strip()
            prompts.append(body[4:].strip() if body.startswith("[NC]") else f"{inputs['character_prompt']}, {body}")
        batch = len(prompts)
        pe, pooled, tids = self.pipe.encode_prompt(prompt=prompts, prompt_2=prompts, device=self.device,
                                                   num_images_per_prompt=1, max_sequence_length=512)
        gen = torch.Generator("cpu").manual_seed(int(seed))
        lh, lw = 2 * (self.H // 16), 2 * (self.W // 16)
        ch = self.tr.config.in_channels // 4
        latents = pack_latents(randn_tensor((batch, ch, lh, lw), generator=gen, device=self.device, dtype=self.dtype),
                               batch, ch, lh, lw)
        img_ids = prepare_latent_image_ids(lh // 2, lw // 2, self.device, self.dtype)
        guidance = (torch.full([1], float(P.get("guidance", 3.5)), device=self.device, dtype=self.dtype).expand(batch)
                    if self.tr.config.guidance_embeds else None)
        cfg = self.pipe.scheduler.config
        sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps)
        mu = calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        self.pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
        timesteps = self.pipe.scheduler.timesteps
        start_step = int(start_frac * len(timesteps))

        def _share(q, k, v, n_txt, attn, pl):
            B, S, h, D = k.shape
            if B <= 1:
                return q, k, v
            pk, pv = k.reshape(B * S, h, D), v.reshape(B * S, h, D)
            if share_ratio < 1.0:
                n = max(1, int(share_ratio * B * S))
                idx = torch.randperm(B * S, device=k.device)[:n]
                pk, pv = pk[idx], pv[idx]
            pk = pk.unsqueeze(0).expand(B, pk.shape[0], h, D)
            pv = pv.unsqueeze(0).expand(B, pv.shape[0], h, D)
            return q, torch.cat([k, pk], dim=1), torch.cat([v, pv], dim=1)

        with flux_intervention(self.tr):
            for i, t in enumerate(timesteps):
                active = share_on and i >= start_step
                jkw = {PAYLOAD_KEY: {"post_rope": _share}} if active else None
                noise = self.tr(hidden_states=latents, timestep=t.expand(batch).to(self.dtype) / 1000,
                                guidance=guidance, pooled_projections=pooled, encoder_hidden_states=pe,
                                txt_ids=tids, img_ids=img_ids, joint_attention_kwargs=jkw, return_dict=False)[0]
                latents = self.pipe.scheduler.step(noise, t, latents, return_dict=False)[0]
        z = unpack_latents(latents, self.H, self.W, 8).to(self.vae.device)
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        img = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img, output_type="pil")

    # ---- editing: RF-inversion caching background K/V, then edit-denoise substituting it (op: substitute) ----
    @torch.no_grad()
    def _run_edit(self, recipe, inputs, P, seed):
        """inputs = {image, prompt (target), source_prompt, mask (white=edit / black=keep, or None)}.
        Faithful to pipelines/kv-edit, but driven through the shared FluxIntervention pre-rope hook instead of a
        bespoke processor: invert the source under its prompt caching background-token K/V per (block, step),
        then denoise under the target prompt substituting that K/V so only the masked region changes."""
        from diffusers.utils.torch_utils import randn_tensor  # noqa: F401  (kept for parity with block imports)
        sched = self.pipe.scheduler
        L = int(P.get("max_sequence_length", 512))
        gh = gw = self.H // 16
        n_img = gh * gw

        x = self.pipe.image_processor.preprocess(inputs["image"], height=self.H, width=self.W).to(self.device, self.vae.dtype)
        x0 = self.vae.encode(x).latent_dist.mode()
        x0 = (x0 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        lh, lw = x0.shape[2], x0.shape[3]
        latents = pack_latents(x0.to(self.dtype), 1, self.vae.config.latent_channels, lh, lw)
        img_ids = prepare_latent_image_ids(lh // 2, lw // 2, self.device, self.dtype)

        bg = None
        mask = inputs.get("mask")
        if mask is not None:
            from PIL import Image
            m = mask.convert("L").resize((lw // 2, lh // 2))
            bg = torch.from_numpy(np.asarray(m) < 128).flatten().to(self.device)   # True = keep (background)
            if not bool(bg.any()):
                bg = None
        if bg is None:                                   # whole-image edit -> nothing to preserve -> plain regen
            bg = torch.zeros(n_img, dtype=torch.bool, device=self.device)
        bg_idx = bg.nonzero(as_tuple=False).squeeze(-1)

        src_pe, src_pooled = self._prompt_embeds(inputs["source_prompt"], L=L)
        tar_pe, tar_pooled = self._prompt_embeds(inputs["prompt"], L=L)
        src_tids = torch.zeros(src_pe.shape[1], 3, device=self.device, dtype=self.dtype)
        tar_tids = torch.zeros(tar_pe.shape[1], 3, device=self.device, dtype=self.dtype)
        g_embeds = self.tr.config.guidance_embeds
        src_g = torch.full((1,), float(P.get("src_guidance", 1.0)), device=self.device, dtype=self.dtype) if g_embeds else None
        tar_g = torch.full((1,), float(P.get("guidance", 3.5)), device=self.device, dtype=self.dtype) if g_embeds else None

        cfg = sched.config
        sigmas = np.linspace(1.0, 1.0 / self.steps, self.steps)
        mu = calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        sched.set_timesteps(sigmas=sigmas, mu=mu, device=self.device)
        timesteps = sched.timesteps

        store = {}

        def _op(mode):
            def _fn(q, k, v, off, attn, pl):
                idx = bg_idx + off
                key = (id(attn), pl["step"])
                if idx.numel() == 0:
                    return q, k, v
                if mode == "capture":
                    store[key] = (k[:, idx].detach().to("cpu"), v[:, idx].detach().to("cpu"))
                elif key in store:
                    sk, sv = store[key]
                    k = k.clone(); v = v.clone()
                    k[:, idx] = sk.to(k); v[:, idx] = sv.to(v)
                return q, k, v
            return _fn

        def _vel(lat, pe, pooled, tids, gd, t, op, n_txt, step):
            jkw = None if op is None else {PAYLOAD_KEY: {"pre_rope": op, "n_txt": n_txt, "step": step}}
            return self.tr(hidden_states=lat, timestep=(t.expand(1) / 1000).to(self.dtype), guidance=gd,
                           pooled_projections=pooled, encoder_hidden_states=pe, txt_ids=tids, img_ids=img_ids,
                           joint_attention_kwargs=jkw, return_dict=False)[0]

        active = bg_idx.numel() > 0
        cap = _op("capture") if active else None
        sub = _op("substitute") if active else None
        with flux_intervention(self.tr):
            # RF inversion (reverse) under the source prompt, caching bg K/V per step
            for i in range(len(timesteps) - 1, -1, -1):
                t = timesteps[i]
                sched._init_step_index(t)
                s_i = sched.sigmas[sched.step_index]; s_ip1 = sched.sigmas[sched.step_index + 1]
                v = _vel(latents, src_pe, src_pooled, src_tids, src_g, t, cap, src_pe.shape[1], i)
                latents = (latents.to(torch.float32) + (s_i - s_ip1) * v.to(torch.float32)).to(self.dtype)
            # edit denoise (forward) under the target prompt, substituting bg K/V per step
            for i, t in enumerate(timesteps):
                v = _vel(latents, tar_pe, tar_pooled, tar_tids, tar_g, t, sub, tar_pe.shape[1], i)
                latents = sched.step(v, t, latents, return_dict=False)[0]

        z = unpack_latents(latents, self.H, self.W, 8).to(self.vae.device)
        z = z / self.vae.config.scaling_factor + self.vae.config.shift_factor
        img = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
        return self.pipe.image_processor.postprocess(img, output_type="pil")[0]

    # ---- sweep: run a recipe over a grid of overrides, return (images, records) ----
    def sweep(self, recipe, axes, inputs, seed=0):
        """``axes`` = {param: [values...]}. Returns (list[(label, image, overrides)], list[dict]).

        Metrics live in the notebook (a panel, not one scalar — depth-corr was gameable twice). This just
        produces the images + the override record for each cell; the caller scores + eyeballs the grid.
        """
        import itertools
        keys = list(axes.keys())
        cells, records = [], []
        for combo in itertools.product(*[axes[k] for k in keys]):
            ov = dict(zip(keys, combo))
            img = self.run(recipe, inputs, seed=seed, **ov)
            label = " ".join(f"{k}={v}" for k, v in ov.items())
            cells.append((label, img, ov))
            records.append(dict(ov))
        return cells, records
