"""Composable attention intervention for FLUX / MMDiT joint attention.

Across our shipped blocks, 9/13 reimplement the *entire* FLUX joint-attention forward
(``to_q/k/v`` -> norm -> head-split -> prepend text -> RoPE -> SDPA -> text/image split) just to insert
ONE operation: an additive mask (stitch, regional-prompting), a K/V capture+inject (appearance-transfer,
kv-edit, story-diffusion, style-aligned), a Q/K/V blend (consistedit), or an attention-weight tap (stitch
cutout). Each copy re-hits the same footguns (bf16 ``-inf`` -> NaN, the ``[text, image]`` split,
single-stream-still-prepends-text, RoPE on appended K/V, SDPA hiding weights).

This module computes the stock forward ONCE (validated bit-exact) and lets a block supply only its
intervention via typed hooks. Candidate for upstream (diffusers ``attention_processor`` / modular).
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb


def flux_qkv(attn, hidden_states, encoder_hidden_states):
    """Stock FLUX joint-attention front-half, computed once and correctly.

    Returns ``q, k, v`` as ``(B, heads, seq, head_dim)`` with text tokens prepended for the double stream,
    plus ``n_txt`` (# prepended text tokens; 0 for the single stream). The image tokens are the trailing
    ``seq - n_txt`` on dim 2 in both stream layouts.
    """
    b = hidden_states.shape[0]
    hd = attn.heads
    q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
    dh = q.shape[-1] // hd
    shp = lambda t: t.view(b, -1, hd, dh).transpose(1, 2)
    q, k, v = shp(q), shp(k), shp(v)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    n_txt = 0
    if encoder_hidden_states is not None:
        eq = shp(attn.add_q_proj(encoder_hidden_states))
        ek = shp(attn.add_k_proj(encoder_hidden_states))
        ev = shp(attn.add_v_proj(encoder_hidden_states))
        if attn.norm_added_q is not None:
            eq = attn.norm_added_q(eq)
        if attn.norm_added_k is not None:
            ek = attn.norm_added_k(ek)
        q, k, v = torch.cat([eq, q], 2), torch.cat([ek, k], 2), torch.cat([ev, v], 2)
        n_txt = eq.shape[2]
    return q, k, v, n_txt


def flux_out(attn, out, encoder_hidden_states, dtype):
    """Reassemble SDPA output ``(B, heads, seq, head_dim)`` and apply the FLUX output projections,
    splitting text/image for the double stream."""
    b = out.shape[0]
    out = out.transpose(1, 2).reshape(b, -1, out.shape[1] * out.shape[3]).to(dtype)
    if encoder_hidden_states is not None:
        enc, out = out[:, : encoder_hidden_states.shape[1]], out[:, encoder_hidden_states.shape[1]:]
        return attn.to_out[1](attn.to_out[0](out)), attn.to_add_out(enc)
    return attn.to_out[0](out) if hasattr(attn, "to_out") else out


class InterventionAttnProcessor:
    """Wraps stock FLUX joint attention and inserts one intervention. Bit-exact stock attention when no
    hook is set (validated). Hooks receive ``n_txt`` so a block never re-derives the text/image split:

      * ``pre_rope(q, k, v, n_txt)  -> (q, k, v)``   blend/replace on raw projections (e.g. ConsistEdit)
      * ``post_rope(q, k, v, n_txt) -> (q, k, v)``   capture/inject position-baked K/V (appearance, KV-Edit)
      * ``attn_bias(q, k, n_txt)    -> Tensor|None``  additive attention bias; use ``-1e4`` (bf16-safe), NOT -inf
      * ``tap(out, n_txt)``                           observe the attention output (e.g. Stitch cutout)

    Install via :func:`install_processors` / :func:`attention_share` so originals are restored on exit.
    """

    def __init__(self, key=None, pre_rope=None, post_rope=None, attn_bias=None, tap=None):
        self.key = key
        self.pre_rope = pre_rope
        self.post_rope = post_rope
        self.attn_bias = attn_bias
        self.tap = tap

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        q, k, v, n_txt = flux_qkv(attn, hidden_states, encoder_hidden_states)
        if self.pre_rope is not None:
            q, k, v = self.pre_rope(q, k, v, n_txt)
        if image_rotary_emb is not None:
            q, k = apply_rotary_emb(q, image_rotary_emb), apply_rotary_emb(k, image_rotary_emb)
        if self.post_rope is not None:
            q, k, v = self.post_rope(q, k, v, n_txt)
        bias = self.attn_bias(q, k, n_txt) if self.attn_bias is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
        if self.tap is not None:
            self.tap(out, n_txt)
        return flux_out(attn, out, encoder_hidden_states, q.dtype)


def edge_blocks(transformer, n=2):
    """Processor keys for the first-``n`` and last-``n`` blocks of BOTH FLUX streams — the empirically
    effective injection sites across our pipelines (dedup'd for small models)."""
    nD = len(transformer.transformer_blocks)
    nS = len(transformer.single_transformer_blocks)
    idxD = list(dict.fromkeys(list(range(n)) + list(range(nD - n, nD))))
    idxS = list(dict.fromkeys(list(range(n)) + list(range(nS - n, nS))))
    return ([f"transformer_blocks.{i}.attn.processor" for i in idxD] +
            [f"single_transformer_blocks.{i}.attn.processor" for i in idxS])


@contextmanager
def install_processors(transformer, make, block_keys):
    """Install ``make(key) -> processor`` on ``block_keys``; restore the originals on exit (the
    save/set/restore-in-``finally`` boilerplate currently in 10/13 blocks)."""
    orig = dict(transformer.attn_processors)
    procs = dict(orig)
    for key in block_keys:
        procs[key] = make(key)
    transformer.set_attn_processor(procs)
    try:
        yield
    finally:
        transformer.set_attn_processor(dict(orig))


@contextmanager
def attention_share(transformer, block_keys, mode, bank=None, n_img=None):
    """Cross-pass attention sharing for FLUX. A ``capture`` pass records each block's IMAGE-token K/V into
    ``bank``; an ``inject`` pass concatenates the banked K/V onto the current pass so queries attend to the
    donor as an appearance/consistency library (queries unchanged). Auto-restores on exit::

        with attention_share(tr, edge_blocks(tr), "capture", n_img=n) as bank:
            tr(donor)                      # records image-only K/V
        with attention_share(tr, edge_blocks(tr), "inject", bank=bank):
            tr(target)                     # injects

    K/V are captured POST-RoPE (position-baked) and appended as a position-agnostic library. Image-only
    slicing (``n_img`` required for capture) keeps it encoder-length agnostic, so it composes with
    prompt-/Redux-extended conditioning.
    """
    bank = {} if bank is None else bank

    def make(key):
        def post(q, k, v, n_txt, _k=key):
            if mode == "capture":
                bank[_k] = (k[:, :, -n_img:].detach(), v[:, :, -n_img:].detach())
            elif mode == "inject" and _k in bank:
                rk, rv = bank[_k]
                k = torch.cat([k, rk.to(k)], dim=2)
                v = torch.cat([v, rv.to(v)], dim=2)
            return q, k, v
        return InterventionAttnProcessor(key=key, post_rope=post)

    with install_processors(transformer, make, block_keys):
        yield bank
