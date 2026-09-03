"""flux_modular (bundled flat build) — GENERATED from flux_modular/ by bundle.py; do not edit here.
Vendored beside block.py so trust_remote_code loads it as a flat sibling."""

# ============================== plumbing.py ==============================
"""FLUX latent-packing / scheduling helpers.

These mirror the private staticmethods duplicated across ``FluxPipeline`` / ``FluxControlPipeline`` /
``FluxFillPipeline`` / Kontext — and re-copied into ~12/13 of our custom blocks. Kept public here so a
block imports them instead of re-deriving (and re-mis-deriving) them. Candidate for upstream as public
``diffusers.pipelines.flux`` utilities.
"""

import torch


def pack_latents(latents, batch, channels, height, width):
    """(B, C, H, W) -> (B, (H/2)*(W/2), C*4). ``height``/``width`` are LATENT-grid dims (e.g. 128 at 1024px)."""
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch, (height // 2) * (width // 2), channels * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    """Inverse of :func:`pack_latents`. ``height``/``width`` are PIXEL dims."""
    b, _, ch = latents.shape
    h = 2 * (height // (vae_scale_factor * 2))
    w = 2 * (width // (vae_scale_factor * 2))
    latents = latents.view(b, h // 2, w // 2, ch // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, ch // 4, h, w)


def prepare_latent_image_ids(height, width, device, dtype):
    """FLUX positional ids for a ``height`` x ``width`` packed grid (row-major)."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    """FlowMatch ``mu`` shift for a packed image-token sequence length (defaults = FLUX scheduler config)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)

# ============================== attention.py ==============================
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
                 attention_mask=None, image_rotary_emb=None, flux_mod=None):
        # NOTE: `flux_mod` MUST be a NAMED parameter (not **kwargs): diffusers' FluxAttention.forward
        # filters joint_attention_kwargs to keys that are explicit processor params, so a **kwargs
        # payload key is silently dropped ("... not expected ... will be ignored"). Blocks pass
        # joint_attention_kwargs={PAYLOAD_KEY: payload} where PAYLOAD_KEY == "flux_mod".
        pl = flux_mod
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
        if bias is not None and bias.dtype != torch.bool:
            bias = bias.to(device=q.device, dtype=q.dtype)   # dispatch/SDPA require mask dtype == query dtype

        # side-channel: hand post-rope q/k to the callable so it can compute explicit attention weights
        # (dispatch_attention_fn cannot expose them) at ITS target block/head and store them. Does NOT
        # change the output (which stays on the dispatched path) — e.g. stitch reads the raw UNbiased head
        # attention here while the output stays on the box-biased dispatch.
        if pl.get("weights") is not None:
            pl["weights"](q, k, n_txt, attn, pl)

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
def edge_attn_ids(transformer, n=2):
    """``id()``s of the attn modules at the first-``n``/last-``n`` blocks of both streams — for op
    self-gating when FluxIntervention is installed on ALL blocks (install-all avoids the
    'not expected by FluxAttnProcessor' warning that a partial install triggers on stock blocks)."""
    nD, nS = len(transformer.transformer_blocks), len(transformer.single_transformer_blocks)
    idxD = list(dict.fromkeys(list(range(n)) + list(range(nD - n, nD))))
    idxS = list(dict.fromkeys(list(range(n)) + list(range(nS - n, nS))))
    return {id(transformer.transformer_blocks[i].attn) for i in idxD} | \
           {id(transformer.single_transformer_blocks[i].attn) for i in idxS}


def op_append(bank, ids=None):
    """post_rope: concat this block's banked (position-baked) donor image K/V — appearance/style share.
    Keyed on ``id(attn)``; if ``ids`` is given, only fires for those blocks (install-all + self-gate)."""
    def _fn(q, k, v, n_txt, attn, pl):
        if (ids is None or id(attn) in ids) and id(attn) in bank:
            rk, rv = bank[id(attn)]
            return q, torch.cat([k, rk.to(k)], 1), torch.cat([v, rv.to(v)], 1)
        return q, k, v
    return _fn


def op_capture_image_kv(bank, n_img, ids=None):
    """post_rope: record this block's trailing ``n_img`` (image) K/V into ``bank[id(attn)]`` (donor pass);
    only for blocks in ``ids`` if given."""
    def _fn(q, k, v, n_txt, attn, pl):
        if ids is None or id(attn) in ids:
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


def last_single_attn_ids(transformer, n=25):
    """``id()``s of the attn modules in the LAST ``n`` single-stream blocks (FreeControl's injection sites)."""
    nS = len(transformer.single_transformer_blocks)
    return {id(transformer.single_transformer_blocks[i].attn) for i in range(max(0, nS - n), nS)}


def op_capture_q(bank, ids=None):
    """pre_rope: store the projected Query at ``ids`` blocks into ``bank[id(attn)]`` (FreeControl capture pass)."""
    def _fn(q, k, v, off, attn, pl):
        if ids is None or id(attn) in ids:
            bank[id(attn)] = q.detach()
        return q, k, v
    return _fn


def op_replace_q(bank, ids=None):
    """pre_rope: replace the IMAGE-token Query with the banked reference Query (FreeControl structural control —
    K/V and text-Q stay dynamic so content follows the prompt). The caller gates WHICH steps this runs (step
    cutoff) by only including it in the payload for early steps."""
    def _fn(q, k, v, off, attn, pl):
        if (ids is None or id(attn) in ids) and id(attn) in bank and bank[id(attn)].shape == q.shape:
            q = q.clone(); q[:, off:] = bank[id(attn)][:, off:].to(q)
        return q, k, v
    return _fn

