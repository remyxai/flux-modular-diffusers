"""flux_modular — shared abstractions for FLUX Modular Diffusers community blocks.

Canonical source of truth. Because ``trust_remote_code`` loads only FLAT sibling ``.py`` files, each HF
pipeline repo ships a bundled flat ``flux_modular.py`` (see ``bundle.py``) vendored beside its ``block.py``.

The attention primitive subclasses diffusers' own ``FluxAttnProcessor`` and is driven through the existing
``joint_attention_kwargs`` seam — see ``attention.py``.
"""

from .plumbing import pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift
from .attention import (
    FluxIntervention,
    flux_intervention,
    edge_attn_ids,
    PAYLOAD_KEY,
    op_append,
    op_capture_image_kv,
    op_capture_q,
    op_replace_q,
    last_single_attn_ids,
)

# Recipe layer (methods-as-configs) + the FLUX runner/sweeper. Imported lazily-friendly: the heavy
# diffusers/transformers loads happen only when FluxLens is constructed, not at ``import flux_modular``.
from .recipes import load_recipe, load_recipes

__all__ = [
    "pack_latents", "unpack_latents", "prepare_latent_image_ids", "calculate_shift",
    "FluxIntervention", "flux_intervention", "edge_attn_ids", "PAYLOAD_KEY",
    "op_append", "op_capture_image_kv",
    "op_capture_q", "op_replace_q", "last_single_attn_ids",
    "load_recipe", "load_recipes",
]


def __getattr__(name):
    # lazy: import the heavy (diffusers-dependent) symbols only on first use, not at package load
    if name == "FluxLens":
        from .lens import FluxLens
        return FluxLens
    if name in ("run_recipe", "ComponentsAdapter"):
        from . import interpret
        return getattr(interpret, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
