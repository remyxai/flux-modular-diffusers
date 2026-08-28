# Brief — KV-Edit → Modular Diffusers (training-free editing, precise background preservation)

**Target repo (private until validated):** `remyxai/kv-edit-flux-modular`
**Block class:** `KVEditBlock` · **Axis:** editing · **Mode:** training-free (no extra weights)
**Prereq:** read `../CONVENTIONS.md`. Related to `flowedit.md` (both training-free FLUX editing) — KV-Edit's edge is **background preservation**.

## One-liner
Training-free text editing that keeps the **unedited background pixel-precise**: cache the source image's
attention **K/V** for background tokens and reuse them during the edit denoise, so only the masked/edited
region changes. Fixes FlowEdit's weaker background fidelity — a *stronger* editing pipeline, not a duplicate.

## Vetting (done)
- **Base:** FLUX (rectified-flow DiT). ✓ image, single-A100.
- **License/repo:** [`Xilluill/KV-Edit`](https://github.com/Xilluill/KV-Edit) — **Apache-2.0**, ⭐388. ✓ permissive.
- **Not in diffusers:** code hits = 0; only `rf_inversion` + our FlowEdit cover editing → **gap**. ✓
- **In-flight:** none. ✓ (also `gh issue view --comments` at scope time per the Detail-Daemon lesson)
- Paper: KV-Edit (arXiv 2502.17363 — VERIFIED authors: Zhu, Zhang, Shao, Tang).

## Mechanism (encode from the ref)
KV-Edit inverts/encodes the source, then during editing **preserves the K/V of background tokens** (region to
keep) while regenerating the foreground under the target prompt — so background content is retained exactly
rather than re-synthesized. Needs a mask (edit region) + source image + target prompt.

## Modular block design
- **Injection pattern:** *attention-processor swap threading cached background K/V via `joint_attention_kwargs`*
  (HRDiT/attn-spike seam) — the processor substitutes source K/V for background image tokens. Restore in `finally`.
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `image`(source), `prompt`(target), `source_prompt`, `mask`(edit region; optional → auto/whole-image),
  `T_steps`, `guidance_scale`, `generator`, `output_type`.
- `__call__`: VAE-encode source → capture per-block source K/V (a reference pass) → denoise under target prompt
  with background K/V substituted for kept tokens → decode.

## Tests
- **smoke.ipynb:** publish PRIVATE → load (assert `KVEditBlock`) → tiny edit → returns image.
- **e2e.ipynb:** an edit that changes the subject; **validate background is preserved** (SSIM/L1 of the unmasked
  region vs source ≈ high) while the target changes (CLIP edit-direction). Compare vs FlowEdit (KV-Edit should
  preserve background better). **Spike:** the K/V-substitution seam on FLUX joint attention (no-op when mask empty).

## Risks
1. Capturing + substituting per-block source K/V (memory + correct token alignment) — the spike gates it.
2. Mask → token-grid mapping (reuse the CatVTON/segformer masking if auto-mask wanted).

## Deliverables
`block.py` · configs · `smoke.ipynb` · `e2e.ipynb` · `README.md` (verified citation).
