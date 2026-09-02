"""flux_modular — shared abstractions for FLUX Modular Diffusers community blocks.

Canonical source of truth. Because ``trust_remote_code`` loads only FLAT sibling ``.py`` files, each HF
pipeline repo ships a bundled flat ``flux_modular.py`` (see ``bundle.py``) vendored beside its ``block.py``;
this package is what that bundle is generated from and what the monorepo tests import.
"""

from .plumbing import pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift
from .attention import (
    flux_qkv,
    flux_out,
    InterventionAttnProcessor,
    edge_blocks,
    install_processors,
    attention_share,
)

__all__ = [
    "pack_latents", "unpack_latents", "prepare_latent_image_ids", "calculate_shift",
    "flux_qkv", "flux_out", "InterventionAttnProcessor", "edge_blocks",
    "install_processors", "attention_share",
]
