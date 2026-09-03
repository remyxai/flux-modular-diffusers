"""FluxLens — run a training-free intervention *recipe* on FLUX, and sweep configurations.

Most of the shipped ``pipelines/`` are the SAME move: install :class:`FluxIntervention`, capture/replace/share/bias
some ``q/k/v`` at some blocks for some steps. A **recipe** (a YAML row in ``recipes/``) makes that move *data*;
``run(recipe, inputs)`` interprets it, and ``sweep`` explores new configurations (incl. compositions the source
papers never tried). FluxLens is the exploration adapter (wraps a ``FluxPipeline``); the SAME interpreter
(:mod:`flux_modular.interpret`) drives generated ``block.py`` files via :class:`ComponentsAdapter`.

Validated on an A100 (2026-09): all 7 recipes; ``freecontrol``-as-config depth-corr 0.883 ≈ shipped block 0.89.
Scope: the op menu is model-agnostic, the adapter is per-checkpoint (an SD3.5 spike showed the adapter ports but
Q-replace is rope-specific) — method transfer is verified per checkpoint. See ``recipes/README.md``.
"""

import torch

from .interpret import run_recipe

_FLUX = "black-forest-labs/FLUX.1-dev"
_REDUX = "black-forest-labs/FLUX.1-Redux-dev"


class FluxLens:
    """FLUX adapter + recipe runner. Construct once (loads FLUX.1-dev; Redux lazily), then ``run``/``sweep`` recipes."""

    def __init__(self, model=_FLUX, dtype=torch.bfloat16, device="cuda", steps=28, height=1024, width=1024):
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(model, torch_dtype=dtype).to(device)
        self.tr, self.vae = self.pipe.transformer, self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.image_processor = self.pipe.image_processor
        self.device, self.dtype = device, dtype
        self.steps, self.H, self.W = steps, height, width
        self.n_img = (height // 16) * (width // 16)
        self._redux = None
        torch.set_grad_enabled(False)   # inference only; a raw tr() capture would else retain a grad graph -> OOM

    # ---- adapter interface (consumed by flux_modular.interpret) ----
    def encode_prompt(self, prompt, L=512):
        pe, pooled, _ = self.pipe.encode_prompt(prompt=prompt, prompt_2=prompt, device=self.device,
                                                num_images_per_prompt=1, max_sequence_length=L)
        return pe, pooled

    def redux_embed(self, img, scale=1.0):
        if self._redux is None:
            from transformers import SiglipVisionModel, SiglipImageProcessor
            from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
            self._redux = (
                SiglipVisionModel.from_pretrained(_REDUX, subfolder="image_encoder", torch_dtype=self.dtype).to(self.device),
                SiglipImageProcessor.from_pretrained(_REDUX, subfolder="feature_extractor"),
                ReduxImageEncoder.from_pretrained(_REDUX, subfolder="image_embedder", torch_dtype=self.dtype).to(self.device),
            )
        ienc, feat, emb = self._redux
        fe = feat(images=img, return_tensors="pt").to(self.device)
        sig = ienc(pixel_values=fe.pixel_values.to(self.dtype)).last_hidden_state
        return emb(sig).image_embeds.to(self.device, self.dtype) * float(scale)

    # ---- run one recipe / sweep a grid ----
    @torch.no_grad()
    def run(self, recipe, inputs, seed=0, **overrides):
        """Interpret a recipe dict against ``inputs`` -> a PIL image (or a list of frames for ``run: batch``)."""
        return run_recipe(self, recipe, inputs, seed=seed, **overrides)

    def sweep(self, recipe, axes, inputs, seed=0):
        """``axes`` = {param: [values...]}. Returns (list[(label, image, overrides)], list[dict]). Metrics live in
        the notebook (a panel, not one scalar — depth-corr was gameable twice); the caller scores + eyeballs."""
        import itertools
        keys = list(axes.keys())
        cells, records = [], []
        for combo in itertools.product(*[axes[k] for k in keys]):
            ov = dict(zip(keys, combo))
            img = self.run(recipe, inputs, seed=seed, **ov)
            cells.append((" ".join(f"{k}={v}" for k, v in ov.items()), img, ov))
            records.append(dict(ov))
        return cells, records
