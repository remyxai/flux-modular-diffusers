"""Recipe interpreter — ONE implementation of the run-loops, driven by an *adapter*.

A recipe (see ``recipes/``) is data; ``run_recipe(a, recipe, inputs)`` turns it into ``flux_modular`` payloads and
runs it. ``a`` is an adapter exposing the model pieces, so the SAME interpreter serves both:
  * :class:`flux_modular.runner.RecipeRunner` (wraps a ``FluxPipeline`` — for exploration/sweeps), and
  * a generated ``block.py`` via :class:`ComponentsAdapter` (wraps Modular-Diffusers ``components`` — the shipped
    turnkey pipeline). So a validated recipe becomes a ~30-line block instead of a re-implemented denoise loop.

Adapter contract (duck-typed): attributes ``tr, vae, scheduler, image_processor, device, dtype, steps, H, W,
n_img`` and methods ``encode_prompt(prompt, L=512) -> (embeds, pooled)`` and ``redux_embed(img, scale=1.0)``.
"""

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from .attention import flux_intervention, last_single_attn_ids, edge_attn_ids, PAYLOAD_KEY
from .plumbing import pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift


# ------------------------------------------------------------------ shared helpers (adapter-driven)
def _sites(a, spec):
    spec = spec or {}
    if spec.get("stream") == "single":
        return last_single_attn_ids(a.tr, int(spec.get("last_n", 25)))
    if spec.get("stream") == "both":
        return edge_attn_ids(a.tr, int(spec.get("edge", 2)))
    return set()


def _text_ids(a, n):
    return torch.zeros(n, 3, device=a.device, dtype=a.dtype)


def _timesteps(a):
    cfg = a.scheduler.config
    sigmas = np.linspace(1.0, 1.0 / a.steps, a.steps)
    mu = calculate_shift(a.n_img, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    a.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=a.device)
    return a.scheduler.timesteps


def _encode_x0(a, img):
    x = a.image_processor.preprocess(img, height=a.H, width=a.W).to(a.device, a.vae.dtype)
    z = a.vae.encode(x).latent_dist.mode()
    z = (z - a.vae.config.shift_factor) * a.vae.config.scaling_factor
    return pack_latents(z, 1, a.vae.config.latent_channels, 2 * (a.H // 16), 2 * (a.W // 16))


def _decode(a, latents):
    z = unpack_latents(latents, a.H, a.W, 8).to(a.vae.device)
    z = z / a.vae.config.scaling_factor + a.vae.config.shift_factor
    img = a.vae.decode(z.to(a.vae.dtype), return_dict=False)[0]
    return a.image_processor.postprocess(img, output_type="pil")


def _capture_q(a, img, sigma, timestep, ids):
    bank = {}
    xt = ((1.0 - float(sigma)) * _encode_x0(a, img)).to(a.dtype)
    pe, pooled = a.encode_prompt("")
    tids = _text_ids(a, pe.shape[1])
    iid = prepare_latent_image_ids(a.H // 16, a.W // 16, a.device, a.dtype)
    g = torch.full((1,), 3.5, device=a.device, dtype=a.dtype)
    ts = torch.full((1,), float(timestep) / 1000, device=a.device, dtype=a.dtype)

    def cap(q, k, v, off, attn, pl):
        if id(attn) in ids:
            bank[id(attn)] = q[:, -a.n_img:].detach()
        return q, k, v

    with flux_intervention(a.tr):
        a.tr(hidden_states=xt, timestep=ts, guidance=g, pooled_projections=pooled, encoder_hidden_states=pe,
             txt_ids=tids, img_ids=iid,
             joint_attention_kwargs={PAYLOAD_KEY: {"pre_rope": cap, "n_txt": pe.shape[1]}}, return_dict=False)
    return bank


def _denoise(a, latents, enc, pooled, img_ids, guidance_scale, payload=None):
    """From-noise denoise loop (mirrors FluxPipeline's dev path). ``payload(step)`` -> joint_attention_kwargs|None."""
    ts = _timesteps(a)
    tids = _text_ids(a, enc.shape[1])
    batch = latents.shape[0]
    guidance = (torch.full([1], float(guidance_scale), device=a.device, dtype=a.dtype).expand(batch)
                if a.tr.config.guidance_embeds else None)
    with flux_intervention(a.tr):
        for i, t in enumerate(ts):
            jkw = payload(i) if payload is not None else None
            noise = a.tr(hidden_states=latents, timestep=t.expand(batch).to(a.dtype) / 1000, guidance=guidance,
                         pooled_projections=pooled, encoder_hidden_states=enc, txt_ids=tids, img_ids=img_ids,
                         joint_attention_kwargs=jkw, return_dict=False)[0]
            latents = a.scheduler.step(noise, t, latents, return_dict=False)[0]
    return latents


def _noise_latents(a, batch, seed):
    lh, lw = 2 * (a.H // 16), 2 * (a.W // 16)
    ch = a.tr.config.in_channels // 4
    g = torch.Generator("cpu").manual_seed(int(seed))
    lat = pack_latents(randn_tensor((batch, ch, lh, lw), generator=g, device=a.device, dtype=a.dtype), batch, ch, lh, lw)
    return lat, prepare_latent_image_ids(lh // 2, lw // 2, a.device, a.dtype)


# ------------------------------------------------------------------ structure schedule (soft replace)
def _replace_schedule(P, steps):
    """Return step -> weight in [0,1] for structure Q-replace. ``schedule``: 'cutoff' (default, hard on/off —
    back-compatible), 'linear', or 'cosine' (smooth 1->0 decay over the first ``S`` fraction of steps). A smooth
    decay releases structure gradually instead of a hard step, which mitigates the over-lock hard cutoffs cause."""
    kind = P.get("schedule", "cutoff")
    cut = max(1, int(float(P.get("S", 0.3)) * steps))
    if kind == "linear":
        return lambda i: float(max(0.0, 1.0 - i / cut)) if i < cut else 0.0
    if kind == "cosine":
        return lambda i: float(0.5 * (1.0 + np.cos(np.pi * i / cut))) if i < cut else 0.0
    return lambda i: 1.0 if i < cut else 0.0


def _replace_q_op(a, bank, ids, st, sched):
    """pre_rope: mix the banked reference image-Q into the current image-Q by the schedule weight w(step).
    w=1 -> full replace (== the old hard cutoff), 0<w<1 -> lerp (soft), w=0 -> untouched."""
    def pre_rope(q, k, v, off, attn, pl):
        w = sched(st["step"])
        if w > 0.0 and id(attn) in ids and id(attn) in bank:
            ref = bank[id(attn)].to(q)
            q = q.clone()
            q[:, -a.n_img:] = ref if w >= 1.0 else torch.lerp(q[:, -a.n_img:], ref, float(w))
        return q, k, v
    return pre_rope


# ------------------------------------------------------------------ dispatch
def run_recipe(a, recipe, inputs, seed=0, **overrides):
    P = {**recipe.get("params", {}), **overrides}
    mode = recipe.get("run", "default")
    if mode == "regional":
        return _run_regional(a, recipe, inputs, P, seed)
    if mode == "composed":
        return _run_composed(a, recipe, inputs, P, seed)
    if mode == "batch":
        return _run_batch(a, recipe, inputs, P, seed)
    if mode == "edit":
        return _run_edit(a, recipe, inputs, P, seed)
    return _run_default(a, recipe, inputs, P, seed)


def _run_default(a, recipe, inputs, P, seed):
    ids = _sites(a, recipe.get("site", {"stream": "single", "last_n": int(P.get("last_n", 25))}))
    bank = {}
    cap = recipe.get("capture")
    if cap:
        bank = _capture_q(a, inputs[cap.get("source", "ref_structure")], P.get("sigma", cap.get("sigma", 0.35)),
                          int(cap.get("timestep", 661)), ids)
    pe, pooled = a.encode_prompt(inputs["prompt"])
    cond = recipe.get("condition")
    enc = pe
    if cond and cond.get("kind") == "redux":
        rx = a.redux_embed(inputs[cond.get("source", "ref_appearance")], scale=P.get("redux_scale", 1.0))
        enc = torch.cat([rx, pe], dim=1)
    replace = any(o.get("op") == "replace_q" for o in recipe.get("ops", []))
    st = {"step": 0}
    pre_rope = _replace_q_op(a, bank, ids, st, _replace_schedule(P, a.steps))

    def payload(i):
        st["step"] = i
        return {PAYLOAD_KEY: {"pre_rope": pre_rope, "n_txt": int(enc.shape[1])}} if replace else None

    latents, img_ids = _noise_latents(a, 1, seed)
    out = _denoise(a, latents, enc, pooled, img_ids, float(P.get("guidance", 6.5)), payload)
    return _decode(a, out)[0]


def _regional_encoding(a, inputs, P):
    """Build the multi-prompt encoder embeds + the joint-attention region bias mask. Shared by
    _run_regional and _run_composed. Returns (enc, base_pooled, mask, n_txt)."""
    cond = P.get("_cond") or {}
    L = int(P.get("region_seq_len", cond.get("region_seq_len", 128)))
    exclusive = bool(P.get("exclusive", cond.get("exclusive", True)))
    strength = float(P.get("isolate_strength", cond.get("isolate_strength", 0.0)))
    regions = list(inputs["regions"])

    base_pe, base_pooled = a.encode_prompt(inputs["base_prompt"], L=L)
    embs, spans, off = [base_pe], [(0, base_pe.shape[1])], base_pe.shape[1]
    for rg in regions:
        e, _ = a.encode_prompt(rg["prompt"], L=L)
        embs.append(e); spans.append((off, off + e.shape[1])); off += e.shape[1]
    enc = torch.cat(embs, dim=1)
    n_txt, n_img = enc.shape[1], a.n_img
    gh = gw = a.H // 16

    region_of = np.full(n_img, -1, dtype=np.int64)
    ys = (np.arange(n_img) // gw + 0.5) / gh
    xs = (np.arange(n_img) % gw + 0.5) / gw
    for r, rg in enumerate(regions):
        x0, y0, x1, y1 = rg["bbox"]
        region_of[(xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)] = r
    joint = n_txt + n_img
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
    bias[n_txt:, :n_txt] = torch.where(text_allow, 0.0, -1e4)
    if strength > 0:
        ro = torch.from_numpy(region_of)
        img_allow = torch.eye(n_img, dtype=torch.bool)
        for r in range(len(regions)):
            sel = ro == r
            img_allow |= (sel.unsqueeze(1) & sel.unsqueeze(0))
        un = ro == -1
        img_allow |= un.unsqueeze(0) | un.unsqueeze(1)
        bias[n_txt:, n_txt:] = torch.where(img_allow, 0.0, -strength)
    mask = bias.unsqueeze(0).unsqueeze(0).to(device=a.device, dtype=a.dtype)
    return enc, base_pooled, mask, n_txt


def _run_regional(a, recipe, inputs, P, seed):
    enc, base_pooled, mask, n_txt = _regional_encoding(a, inputs, {**P, "_cond": recipe.get("condition")})
    payload = lambda i: {PAYLOAD_KEY: {"bias": (lambda q, k, ntxt, attn, pl: mask), "n_txt": n_txt}}
    latents, img_ids = _noise_latents(a, 1, seed)
    out = _denoise(a, latents, enc, base_pooled, img_ids, float(P.get("guidance", 3.5)), payload)
    return _decode(a, out)[0]


def _run_composed(a, recipe, inputs, P, seed):
    """Single-pass COMPOSITION: reference-structure lock (freecontrol Q-replace) + per-region prompts
    (regional bias) in one denoise — both hooks in one payload. inputs = {ref_structure, base_prompt, regions}."""
    enc, base_pooled, mask, n_txt = _regional_encoding(a, inputs, {**P, "_cond": recipe.get("condition")})
    ids = _sites(a, recipe.get("site", {"stream": "single", "last_n": int(P.get("last_n", 25))}))
    bank = {}
    cap = recipe.get("capture")
    if cap:
        bank = _capture_q(a, inputs[cap.get("source", "ref_structure")], P.get("sigma", cap.get("sigma", 0.35)),
                          int(cap.get("timestep", 661)), ids)
    replace = any(o.get("op") == "replace_q" for o in recipe.get("ops", []))
    st = {"step": 0}
    region_bias = lambda q, k, ntxt, attn, pl: mask
    pre_rope = _replace_q_op(a, bank, ids, st, _replace_schedule(P, a.steps))

    def payload(i):
        st["step"] = i
        p = {"bias": region_bias, "n_txt": n_txt}
        if replace:
            p["pre_rope"] = pre_rope
        return {PAYLOAD_KEY: p}

    latents, img_ids = _noise_latents(a, 1, seed)
    out = _denoise(a, latents, enc, base_pooled, img_ids, float(P.get("guidance", 3.5)), payload)
    return _decode(a, out)[0]


def _run_batch(a, recipe, inputs, P, seed):
    cond = recipe.get("condition") or {}
    share_ratio = float(P.get("share_ratio", cond.get("share_ratio", 0.3)))
    start_frac = float(P.get("share_start_frac", cond.get("share_start_frac", 0.35)))
    share_on = bool(P.get("story_share", True))

    char = inputs.get("character_prompt", "")
    prompts = []
    for sc in inputs["scene_prompts"]:
        body = sc.partition("#")[0].strip()
        if body.startswith("[NC]"):
            prompts.append(body[4:].strip())
        else:
            prompts.append(f"{char}, {body}" if char else body)
    pe, pooled = a.encode_prompt(prompts)
    # optional: condition every frame on a reference image (Redux appearance) — a character-from-a-photo across
    # scenes. NOTE: Redux carries appearance/look, NOT tight facial identity (that needs the identity/residual path).
    ref = inputs.get("reference_image")
    if ref is not None:
        rx = a.redux_embed(ref, scale=float(P.get("redux_scale", 1.0)))
        pe = torch.cat([rx.expand(pe.shape[0], -1, -1), pe], dim=1)
    latents, img_ids = _noise_latents(a, len(prompts), seed)
    start_step = int(start_frac * a.steps)

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

    def payload(i):
        return {PAYLOAD_KEY: {"post_rope": _share}} if (share_on and i >= start_step) else None

    out = _denoise(a, latents, pe, pooled, img_ids, float(P.get("guidance", 3.5)), payload)
    return _decode(a, out)


def _run_edit(a, recipe, inputs, P, seed):
    sched = a.scheduler
    L = int(P.get("max_sequence_length", 512))
    n_img = a.n_img

    x = a.image_processor.preprocess(inputs["image"], height=a.H, width=a.W).to(a.device, a.vae.dtype)
    x0 = a.vae.encode(x).latent_dist.mode()
    x0 = (x0 - a.vae.config.shift_factor) * a.vae.config.scaling_factor
    lh, lw = x0.shape[2], x0.shape[3]
    latents = pack_latents(x0.to(a.dtype), 1, a.vae.config.latent_channels, lh, lw)
    img_ids = prepare_latent_image_ids(lh // 2, lw // 2, a.device, a.dtype)

    bg = None
    mask = inputs.get("mask")
    if mask is not None:
        m = mask.convert("L").resize((lw // 2, lh // 2))
        bg = torch.from_numpy(np.asarray(m) < 128).flatten().to(a.device)
        if not bool(bg.any()):
            bg = None
    if bg is None:
        bg = torch.zeros(n_img, dtype=torch.bool, device=a.device)
    bg_idx = bg.nonzero(as_tuple=False).squeeze(-1)

    src_pe, src_pooled = a.encode_prompt(inputs["source_prompt"], L=L)
    tar_pe, tar_pooled = a.encode_prompt(inputs["prompt"], L=L)
    src_tids, tar_tids = _text_ids(a, src_pe.shape[1]), _text_ids(a, tar_pe.shape[1])
    ge = a.tr.config.guidance_embeds
    src_g = torch.full((1,), float(P.get("src_guidance", 1.0)), device=a.device, dtype=a.dtype) if ge else None
    tar_g = torch.full((1,), float(P.get("guidance", 3.5)), device=a.device, dtype=a.dtype) if ge else None

    cfg = sched.config
    sigmas = np.linspace(1.0, 1.0 / a.steps, a.steps)
    mu = calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                         cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
    sched.set_timesteps(sigmas=sigmas, mu=mu, device=a.device)
    timesteps = sched.timesteps

    op_kind = (recipe.get("ops") or [{}])[0].get("op", "substitute")
    store = {}

    def _vel(lat, pe, pooled, tids, gd, t, op, n_txt, step, struct=True):
        jkw = None if op is None else {PAYLOAD_KEY: {"pre_rope": op, "n_txt": n_txt, "step": step, "struct": struct}}
        return a.tr(hidden_states=lat, timestep=(t.expand(1) / 1000).to(a.dtype), guidance=gd,
                    pooled_projections=pooled, encoder_hidden_states=pe, txt_ids=tids, img_ids=img_ids,
                    joint_attention_kwargs=jkw, return_dict=False)[0]

    active = bg_idx.numel() > 0
    n_struct = int(float(P.get("consistency_strength", 0.0)) * a.steps)

    if op_kind == "blend":
        single_ids = {id(b.attn) for b in a.tr.single_transformer_blocks}
        keep_w = bg.to(a.dtype).view(1, -1, 1, 1)

        def _cap(q, k, v, off, attn, pl):
            if id(attn) in single_ids:
                store[(id(attn), pl["step"])] = (q.detach().to("cpu"), k.detach().to("cpu"), v.detach().to("cpu"))
            return q, k, v

        def _fuse(q, k, v, off, attn, pl):
            key = (id(attn), pl["step"])
            if id(attn) in single_ids and key in store:
                qs, ks, vs = (t.to(q) for t in store[key])
                wqk = torch.ones_like(keep_w) if pl["struct"] else keep_w
                q = q.clone(); k = k.clone(); v = v.clone()
                q[:, off:] = torch.lerp(q[:, off:], qs[:, off:], wqk)
                k[:, off:] = torch.lerp(k[:, off:], ks[:, off:], wqk)
                v[:, off:] = torch.lerp(v[:, off:], vs[:, off:], keep_w)
            return q, k, v

        cap = _cap if active else None
        with flux_intervention(a.tr):
            lat = [None] * (a.steps + 1); lat[a.steps] = latents.clone()
            for i in range(len(timesteps) - 1, -1, -1):
                t = timesteps[i]; sched._init_step_index(t)
                s_i, s_ip1 = sched.sigmas[sched.step_index], sched.sigmas[sched.step_index + 1]
                v = _vel(lat[i + 1], src_pe, src_pooled, src_tids, src_g, t, None, src_pe.shape[1], i)
                lat[i] = (lat[i + 1].to(torch.float32) + (s_i - s_ip1) * v.to(torch.float32)).to(a.dtype)
            latents = lat[0].clone()
            for i, t in enumerate(timesteps):
                sched._init_step_index(t)
                _vel(lat[i + 1], src_pe, src_pooled, src_tids, src_g, t, cap, src_pe.shape[1], i)
                fuse = (lambda *ar: _fuse(*ar)) if active else None
                v = _vel(latents, tar_pe, tar_pooled, tar_tids, tar_g, t, fuse, tar_pe.shape[1], i, struct=i < n_struct)
                latents = sched.step(v, t, latents, return_dict=False)[0]
                store.clear()
    else:
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
        cap = _op("capture") if active else None
        sub = _op("substitute") if active else None
        with flux_intervention(a.tr):
            for i in range(len(timesteps) - 1, -1, -1):
                t = timesteps[i]; sched._init_step_index(t)
                s_i, s_ip1 = sched.sigmas[sched.step_index], sched.sigmas[sched.step_index + 1]
                v = _vel(latents, src_pe, src_pooled, src_tids, src_g, t, cap, src_pe.shape[1], i)
                latents = (latents.to(torch.float32) + (s_i - s_ip1) * v.to(torch.float32)).to(a.dtype)
            for i, t in enumerate(timesteps):
                v = _vel(latents, tar_pe, tar_pooled, tar_tids, tar_g, t, sub, tar_pe.shape[1], i)
                latents = sched.step(v, t, latents, return_dict=False)[0]

    return _decode(a, latents)[0]


# ------------------------------------------------------------------ block-side adapter (Modular components)
class ComponentsAdapter:
    """Adapter over Modular-Diffusers ``components`` so a generated ``block.py`` can call :func:`run_recipe`."""

    def __init__(self, components, steps=28, height=1024, width=1024):
        from diffusers.image_processor import VaeImageProcessor
        c = components
        self._c = c
        self.tr, self.vae, self.scheduler = c.transformer, c.vae, c.scheduler
        self.device, self.dtype = self.tr.device, self.tr.dtype
        self.steps, self.H, self.W = int(steps), int(height), int(width)
        self.n_img = (self.H // 16) * (self.W // 16)
        self.image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(self.vae.config.block_out_channels) - 1))

    def encode_prompt(self, prompt, L=512):
        c = self._c
        if isinstance(prompt, str):
            prompt = [prompt]
        ci = c.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        pooled = c.text_encoder(ci.to(c.text_encoder.device), output_hidden_states=False).pooler_output.to(self.device, self.dtype)
        t5 = c.tokenizer_2(prompt, padding="max_length", max_length=L, truncation=True, return_tensors="pt").input_ids
        emb = c.text_encoder_2(t5.to(c.text_encoder_2.device))[0].to(self.device, self.dtype)
        return emb, pooled

    def redux_embed(self, img, scale=1.0):
        c = self._c
        fe = c.feature_extractor(images=img, return_tensors="pt").to(c.image_encoder.device)
        sig = c.image_encoder(pixel_values=fe.pixel_values.to(c.image_encoder.dtype)).last_hidden_state
        return c.image_embedder(sig).image_embeds.to(self.device, self.dtype) * float(scale)
