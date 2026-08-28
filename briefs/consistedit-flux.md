# Brief — ConsistEdit → Modular Diffusers (highly-consistent training-free editing)

**Target repo (private until validated):** `remyxai/consistedit-flux-modular`
**Block class:** `ConsistEditBlock` · **Axis:** editing · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`. The **FLUX-native** editing option (vs FlowEdit's ODE, KV-Edit's KV-preserve).

## One-liner
Highly consistent, precise training-free visual editing **designed for FLUX** — the paper applies its edit
operations specifically to FLUX's **single blocks**, giving strong structure/consistency during edits.

## Vetting (done)
- **Base:** **FLUX** (paper explicitly targets FLUX double/single blocks). ✓
- **License/repo:** **[VERIFY]** locate the official repo (arXiv should link "Code") + confirm license before build.
- **Not in diffusers:** `consistedit` code hits = 0 → **gap**. ✓  (also `--comments` check at scope time)
- Paper: ConsistEdit (arXiv 2510.17803 — VERIFIED authors: Yin, Chen, Ni, Dai).

## Mechanism (encode from the ref)
Attention-level editing applied to the **single-stream blocks** of FLUX for consistency + precision (read the
paper for the exact per-block operation — likely masked/guided attention manipulation between source and target).
Training-free; no extra weights.

## Modular block design
- **Injection pattern:** *attention-processor swap on the single blocks* (subset of the attn-spike seam; the
  processor can be installed only on `single_transformer_blocks` per the paper). Restore in `finally`; no-op when off.
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `image`(source), `prompt`(target), `source_prompt`, plus the paper's edit-strength/consistency knobs
  (default to ref), `guidance_scale`, `generator`, `output_type`.
- `__call__`: encode source + prompts → denoise with the ConsistEdit single-block processor → decode.

## Tests
- **smoke.ipynb:** load (assert `ConsistEditBlock`) → tiny edit → image.
- **e2e.ipynb:** structure-preserving edit; validate consistency (background/structure ≈ preserved) + target change
  (CLIP). **Spike:** the single-block editing seam (no-op when off; consistency ON vs OFF).

## Risks
1. **[VERIFY] repo/license** first — brief-blocking until confirmed.
2. Exact single-block operation (read the ref) — the spike gates fidelity.

## Deliverables
`block.py` · configs · `smoke.ipynb` · `e2e.ipynb` · `README.md` (verified citation + repo/license).
