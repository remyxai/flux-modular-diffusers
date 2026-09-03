"""Identity bridge — PuLID face-lock via the :func:`flux_residual` seam.

Phase-0 validated (A100): routing PuLID's injection through ``flux_residual`` with an id-keyed ``camap`` is
BIT-IDENTICAL to PuLID's own per-forward counter (pixel MAD 0.00) and preserves identity (ArcFace 0.76 vs 0.07
no-identity baseline). So instead of re-implementing the ID stack, this bridges the SHIPPED PuLID pipeline
(``remyxai/pulid-flux-modular`` — ``_PuLIDEncoder`` + trained ``pulid_ca`` + ``guozinan/PuLID`` weights) to the seam.

**OPEN-WEIGHT, not training-free**: uses PuLID's trained ID adapter + InsightFace/facexlib/EVA-CLIP. Deps are heavy
and fetched on first use. The PuLID injection is: ``out[1] += id_weight * pulid_ca[k](id_embedding, out[1])`` on the
double (``i%2``) and single (``i%4``) blocks — exactly a ``flux_residual`` fn.
"""

import numpy as np

_PULID_REPO = "remyxai/pulid-flux-modular"
_PULID = {}   # process-wide cache: the ID encoder + its module (heavy to build)


def load_pulid_encoder(device, dtype):
    """Load (once) the shipped PuLID ID encoder + its module. Returns ``(encoder, module)``."""
    if "enc" not in _PULID:
        import sys
        from diffusers import ModularPipeline
        pp = ModularPipeline.from_pretrained(_PULID_REPO, trust_remote_code=True)
        mod = sys.modules[type(pp.blocks).__module__]
        _PULID["mod"] = mod
        _PULID["enc"] = mod._PuLIDEncoder(device, dtype)   # InsightFace + facexlib + EVA-CLIP + IDFormer + weights
    return _PULID["enc"], _PULID["mod"]


def id_embedding_from(encoder, module, image):
    """Face image (PIL) -> PuLID id_embedding (the ID tokens injected at each block)."""
    id_np = module.resize_numpy_image_long(np.asarray(image.convert("RGB")), 1024)
    return encoder.get_id_embedding(id_np)


def pulid_camap(transformer, double_interval=2, single_interval=4):
    """``{id(block): k}`` mapping each injection block to its ``pulid_ca`` index — double (``i%2``) then single
    (``i%4``) in forward order. This reproduces PuLID's per-forward counter (Phase-0: bit-identical)."""
    camap, k = {}, 0
    for i, b in enumerate(transformer.transformer_blocks):
        if i % double_interval == 0:
            camap[id(b)] = k; k += 1
    for i, b in enumerate(transformer.single_transformer_blocks):
        if i % single_interval == 0:
            camap[id(b)] = k; k += 1
    return camap


def op_identity(encoder, id_embedding, id_weight, camap):
    """flux_residual fn: ``h += id_weight * pulid_ca[camap[id(block)]](id_embedding, h)`` — the PuLID residual."""
    def fn(h, bid):
        add = id_weight * encoder.pulid_ca[camap[bid]](id_embedding, h.to(id_embedding.dtype))
        return h + add.to(h.dtype)
    return fn
