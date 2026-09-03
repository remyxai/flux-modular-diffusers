"""Residual / feature hook — modulate a FLUX block's OUTPUT (image stream), not its attention q/k/v.

This is the seam identity methods (PuLID-class) inject through: a residual add / feature modulation on the block
output. FLUX blocks (BOTH ``transformer_blocks`` and ``single_transformer_blocks``) return
``(encoder_hidden_states, image_hidden_states)`` — the image stream is ``out[1]`` (n_img tokens) — confirmed by a
probe on an A100. ``flux_residual(transformer, ids, fn)`` installs forward-hooks on the blocks in ``ids`` and
replaces ``out[1]`` with ``fn(out[1], id(block))``; restore on exit.

The training-free demo op is a **feature echo** (capture a reference's block features, blend the generation's
features toward them on a schedule) — validated: fires 84-192 hooks, depth-corr 0.36->0.96 at cut 0.3 (a strong
reference reconstruction). A real face-LOCK plugs a trained ID adapter (eva_clip+arcface+projection, e.g. PuLID)
in as ``fn`` instead of this echo.
"""

from contextlib import contextmanager

import torch


@contextmanager
def flux_residual(transformer, ids, fn):
    """Install forward-hooks on blocks whose ``id()`` is in ``ids``; ``fn(image_feat, block_id) -> image_feat``
    modulates the block-output image stream (``out[1]``). No-op / restored on exit."""
    handles = []

    def _hook(module, inp, out):
        if isinstance(out, tuple):
            enc, h = out
            return (enc, fn(h, id(module)))
        return None   # unexpected (non-tuple) block output -> leave untouched

    for b in list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks):
        if id(b) in ids:
            handles.append(b.register_forward_hook(_hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def residual_block_ids(transformer, last_single=12):
    """``id()``s of the last-``last_single`` single-stream blocks — the residual injection sites."""
    nS = len(transformer.single_transformer_blocks)
    return {id(transformer.single_transformer_blocks[i]) for i in range(max(0, nS - last_single), nS)}


def op_capture_feat(bank):
    """residual fn: record each block's output image features into ``bank[id(block)]`` (reference pass)."""
    def _fn(h, bid):
        bank[bid] = h.detach()
        return h
    return _fn


def op_inject_feat(bank, w_fn, st):
    """residual fn: blend the generation's block features toward the banked reference features by ``w_fn(step)``
    (``st['step']`` is set per denoise step by the caller). w=1 -> full reference, 0 -> untouched."""
    def _fn(h, bid):
        w = w_fn(st["step"])
        return torch.lerp(h, bank[bid].to(h), float(w)) if (bid in bank and w > 0.0) else h
    return _fn
