"""Lightweight tests for the shared abstractions (run in CI / Colab; needs torch, not a GPU).

The heavy attention path is validated per-pipeline in the smoke/e2e notebooks (it needs a real FLUX
transformer). Here we cover the pure-tensor plumbing + the intervention wiring that CAN run on CPU.
"""

import torch

from flux_modular import (
    pack_latents, unpack_latents, prepare_latent_image_ids, calculate_shift,
    FluxIntervention, edge_attn_ids, PAYLOAD_KEY,
    op_append, op_capture_image_kv,
    op_capture_q, op_replace_q, last_single_attn_ids,
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


def test_edge_attn_ids():
    class _A:
        pass
    class _Blk:
        def __init__(self): self.attn = _A()
    class _T:
        transformer_blocks = [_Blk() for _ in range(19)]
        single_transformer_blocks = [_Blk() for _ in range(38)]
    ids = edge_attn_ids(_T(), n=2)
    assert len(ids) == 8   # first-2 + last-2 of each of the two streams, dedup'd


def test_payload_key_is_named_param():
    # diffusers' FluxAttention.forward filters joint_attention_kwargs to keys that are EXPLICIT named
    # params of processor.__call__; a **kwargs payload key is silently dropped. So PAYLOAD_KEY MUST be
    # a named parameter of FluxIntervention.__call__ or every op becomes a no-op.
    import inspect
    params = inspect.signature(FluxIntervention.__call__).parameters
    assert PAYLOAD_KEY in params, f"'{PAYLOAD_KEY}' must be a named __call__ param (diffusers drops **kwargs keys)"


def test_intervention_is_flux_processor_subclass():
    # subclasses diffusers' FluxAttnProcessor so no-payload calls delegate to the stock (bit-exact) path
    from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
    assert issubclass(FluxIntervention, FluxAttnProcessor)
    assert PAYLOAD_KEY == "flux_mod"


def test_op_builders_return_callables():
    assert callable(op_append({}))
    assert callable(op_capture_image_kv({}, 4096))
    assert callable(op_capture_q({}))
    assert callable(op_replace_q({}))


def test_freecontrol_ops():
    class _Blk:
        def __init__(self): self.attn = object()
    class _T:
        transformer_blocks = [_Blk() for _ in range(19)]
        single_transformer_blocks = [_Blk() for _ in range(38)]
    ids = last_single_attn_ids(_T(), n=25)
    assert len(ids) == 25
    assert callable(op_capture_q({})) and callable(op_replace_q({}))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok:", name)
    print("all tests passed")
