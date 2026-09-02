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
    edge_blocks,
    PAYLOAD_KEY,
    op_append,
    op_capture_image_kv,
    op_substitute,
    op_blend,
)

__all__ = [
    "pack_latents", "unpack_latents", "prepare_latent_image_ids", "calculate_shift",
    "FluxIntervention", "flux_intervention", "edge_blocks", "PAYLOAD_KEY",
    "op_append", "op_capture_image_kv", "op_substitute", "op_blend",
]
