"""Lightweight tests for the shared abstractions (run in CI / Colab; needs torch, not a GPU).

The heavy attention path is validated per-pipeline in the smoke/e2e notebooks (it needs a real FLUX
transformer). Here we cover the pure-tensor plumbing + the intervention wiring that CAN run on CPU.
"""

import torch

from flux_modular import (
    pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift,
    InterventionAttnProcessor, edge_blocks,
)


def test_pack_unpack_roundtrip():
    b, c, h, w = 1, 16, 128, 128            # 1024px latent grid
    z = torch.randn(b, c, h, w)
    packed = pack_latents(z, b, c, h, w)
    assert packed.shape == (b, (h // 2) * (w // 2), c * 4)
    back = unpack_latents(packed, 1024, 1024, 8)
    assert back.shape == z.shape
    assert torch.allclose(back, z, atol=1e-6)


def test_image_ids_shape():
    ids = prepare_latent_image_ids(64, 64, "cpu", torch.float32)
    assert ids.shape == (64 * 64, 3)


def test_calculate_shift_monotonic():
    assert calculate_shift(256) < calculate_shift(4096)


def test_edge_blocks():
    class _T:
        transformer_blocks = list(range(19))
        single_transformer_blocks = list(range(38))
    keys = edge_blocks(_T(), n=2)
    assert "transformer_blocks.0.attn.processor" in keys
    assert "transformer_blocks.18.attn.processor" in keys
    assert "single_transformer_blocks.37.attn.processor" in keys
    assert len(keys) == 8


def test_processor_hooks_optional():
    p = InterventionAttnProcessor()          # no hooks -> pure pass-through wiring
    assert p.pre_rope is None and p.post_rope is None and p.attn_bias is None and p.tap is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok:", name)
    print("all tests passed")
