"""Lightweight tests for the shared abstractions (run in CI / Colab; needs torch, not a GPU).

The heavy attention path is validated per-pipeline in the smoke/e2e notebooks (it needs a real FLUX
transformer). Here we cover the pure-tensor plumbing + the intervention wiring that CAN run on CPU.
"""

import torch

from flux_modular import (
    pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift,
    FluxIntervention, flux_intervention, edge_blocks, PAYLOAD_KEY,
    op_append, op_capture_image_kv, op_substitute, op_blend,
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


def test_intervention_is_flux_processor_subclass():
    # subclasses diffusers' FluxAttnProcessor so no-payload calls delegate to the stock (bit-exact) path
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    assert issubclass(FluxIntervention, FluxAttnProcessor)
    assert PAYLOAD_KEY == "flux_mod"


def test_op_builders_return_callables():
    assert callable(op_append({}))
    assert callable(op_capture_image_kv({}, 4096))
    assert callable(op_substitute(None, None, None))
    assert callable(op_blend(None, None, None, 0.5))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok:", name)
    print("all tests passed")
