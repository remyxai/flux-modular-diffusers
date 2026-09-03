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

    # ---- the interpreter: one code path for every recipe ----
    @torch.no_grad()
    def run(self, recipe, inputs, seed=0, **overrides):
        """Interpret a recipe dict against ``inputs`` (``prompt`` + any ``ref_*`` images) -> a PIL image."""
        P = {**recipe.get("params", {}), **overrides}
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
