"""Composable attention intervention for FLUX joint attention — built on diffusers' own APIs.

Design principle: reuse existing diffusers machinery, add only what's missing.
  * subclass ``FluxAttnProcessor`` and ``return super().__call__(...)`` when there is no payload
    -> the stock path is bit-exact AND keeps the attention-backend dispatch (flash/sage/sdpa);
  * for the intervention path, reuse ``_get_qkv_projections``, ``apply_rotary_emb(sequence_dim=1)``
    and ``dispatch_attention_fn`` exactly as the stock processor does (so we stay on diffusers'
    current convention rather than a hand-rolled forward);
  * drive it through the existing ``joint_attention_kwargs`` seam (the HRDiT/kv-edit named-kwarg
    pattern) — no new dispatch mechanism.

The only genuinely new code is (a) a small op menu and (b) an explicit-attention path for exposing
attention *weights* (``dispatch_attention_fn`` cannot, e.g. stitch's cutout). This generalizes the
pattern kv-edit already uses. Candidate for upstream (diffusers attention_processor / modular).
"""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections

try:  # backend-aware attention (flash/sage/sdpa) — same call the stock processor uses
    from diffusers.models.attention_dispatch import dispatch_attention_fn
except Exception:  # very old diffusers fallback
    dispatch_attention_fn = None

PAYLOAD_KEY = "flux_mod"   # joint_attention_kwargs[PAYLOAD_KEY] -> intervention payload (None -> stock)


class FluxIntervention(FluxAttnProcessor):
    """Reusable FLUX joint-attention intervention. Install once per model; drive per call via
    ``joint_attention_kwargs={"flux_mod": payload}``. No payload -> stock ``FluxAttnProcessor`` (bit-exact,
    backend-dispatched). The payload is a dict of optional callables applied on the canonical q/k/v:

      pre_rope(q, k, v, off, attn, pl) -> (q, k, v)   # normalized, pre-RoPE, image tokens at [.., off:]
                                                        #   (substitute/blend — kv-edit, consistedit)
      post_rope(q, k, v, ntxt, attn, pl) -> (q, k, v)  # position-baked (append/share — appearance, style-aligned)
      bias(q, k, ntxt, attn, pl) -> Tensor|None        # additive attn mask; use -1e4, NOT -inf (stitch, regional)
      tap(out, ntxt, attn, pl)                          # observe attention output
      weights(q, k, ntxt, attn, pl)                     # explicit softmax(QK^T) to expose weights (stitch cutout)

    ``off`` = image-token start (0 for double-stream where the image is projected alone; ``pl['n_txt']``
    for single-stream where the block receives cat([text, image])). ``ntxt`` = prepended text length.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        pl = kwargs.get(PAYLOAD_KEY)
        if pl is None:  # stock path — bit-exact + backend dispatch, zero reimplementation
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        # --- intervention path: mirrors FluxAttnProcessor.__call__, reusing the same helpers ---
        q, k, v, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
        v = v.unflatten(-1, (attn.heads, -1))

        # pre-RoPE op runs on the image stream (before text is prepended), matching kv-edit's offset rule
        off = 0 if encoder_hidden_states is not None else int(pl.get("n_txt", 0))
        if pl.get("pre_rope") is not None:
            q, k, v = pl["pre_rope"](q, k, v, off, attn, pl)

        if encoder_hidden_states is not None:  # double stream: prepend text q/k/v
            eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
            ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
            ev = ev.unflatten(-1, (attn.heads, -1))
            q, k, v = torch.cat([eq, q], 1), torch.cat([ek, k], 1), torch.cat([ev, v], 1)
        n_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else int(pl.get("n_txt", 0))

        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        if pl.get("post_rope") is not None:
            q, k, v = pl["post_rope"](q, k, v, n_txt, attn, pl)

        bias = pl["bias"](q, k, n_txt, attn, pl) if pl.get("bias") is not None else None

        if pl.get("weights") is not None:
            # explicit attention so weights are observable (dispatch_attention_fn cannot expose them)
            qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # (B, heads, seq, dh)
            scale = qh.shape[-1] ** -0.5
            logits = (qh @ kh.transpose(-1, -2)) * scale
            if bias is not None:
                logits = logits + bias
            probs = logits.softmax(dim=-1)
            pl["weights"](probs, n_txt, attn, pl)
            out = (probs @ vh).transpose(1, 2)                                     # back to (B, seq, heads, dh)
        else:
            out = dispatch_attention_fn(q, k, v, attn_mask=bias,
                                        backend=getattr(self, "_attention_backend", None),
                                        parallel_config=getattr(self, "_parallel_config", None))
        out = out.flatten(2, 3).to(q.dtype)
        if pl.get("tap") is not None:
            pl["tap"](out, n_txt, attn, pl)

        if encoder_hidden_states is not None:
            enc, out = out.split_with_sizes([n_txt, out.shape[1] - n_txt], dim=1)
            out = attn.to_out[1](attn.to_out[0](out.contiguous()))
            enc = attn.to_add_out(enc.contiguous())
            return out, enc
        return out


def edge_blocks(transformer, n=2):
    """Processor keys for the first-``n`` and last-``n`` blocks of BOTH FLUX streams (dedup'd)."""
    nD, nS = len(transformer.transformer_blocks), len(transformer.single_transformer_blocks)
    idxD = list(dict.fromkeys(list(range(n)) + list(range(nD - n, nD))))
    idxS = list(dict.fromkeys(list(range(n)) + list(range(nS - n, nS))))
    return ([f"transformer_blocks.{i}.attn.processor" for i in idxD] +
            [f"single_transformer_blocks.{i}.attn.processor" for i in idxS])


@contextmanager
def flux_intervention(transformer, block_keys=None):
    """Install :class:`FluxIntervention` on ``block_keys`` (default: all attn processors); restore the
    originals on exit. Drive per call via ``joint_attention_kwargs={PAYLOAD_KEY: payload}``; with no
    payload each processor is bit-exact stock, so this is safe to leave installed across a whole run."""
    orig = dict(transformer.attn_processors)
    keys = block_keys if block_keys is not None else list(orig.keys())
    procs = dict(orig)
    for key in keys:
        procs[key] = FluxIntervention()
    transformer.set_attn_processor(procs)
    try:
        yield
    finally:
        transformer.set_attn_processor(dict(orig))


# ---- ready-made ops (blocks may also pass their own callables) -----------------------------------
def op_append(bank):
    """post_rope: concat this block's banked (position-baked) donor image K/V — appearance/style share.
    Keyed on ``id(attn)`` so one shared payload serves every installed block (kv-edit's convention)."""
    def _fn(q, k, v, n_txt, attn, pl):
        rk, rv = bank[id(attn)]
        return q, torch.cat([k, rk.to(k)], 1), torch.cat([v, rv.to(v)], 1)
    return _fn


def op_capture_image_kv(bank, n_img):
    """post_rope: record this block's trailing ``n_img`` (image) K/V into ``bank[id(attn)]`` (donor pass)."""
    def _fn(q, k, v, n_txt, attn, pl):
        bank[id(attn)] = (k[:, -n_img:].detach(), v[:, -n_img:].detach())
        return q, k, v
    return _fn


def op_substitute(k_src, v_src, idx):
    """pre_rope: in-place swap of K/V at image-token ``idx`` (background preservation — kv-edit)."""
    def _fn(q, k, v, off, attn, pl):
        j = idx + off
        k[:, j] = k_src.to(device=k.device, dtype=k.dtype)
        v[:, j] = v_src.to(device=v.device, dtype=v.dtype)
        return q, k, v
    return _fn


def op_blend(q_s, k_s, v_s, w, off=0):
    """pre_rope: lerp toward a donor's image-slice q/k/v (structural consistency — consistedit)."""
    def _fn(q, k, v, o, attn, pl):
        s = (off or o)
        q[:, s:] = torch.lerp(q[:, s:], q_s.to(q), w)
        k[:, s:] = torch.lerp(k[:, s:], k_s.to(k), w)
        v[:, s:] = torch.lerp(v[:, s:], v_s.to(v), w)
        return q, k, v
    return _fn
