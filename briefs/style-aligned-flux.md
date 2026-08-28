# Brief — StyleAligned → FLUX (training-free style-consistent set generation)

**Target repo (private until validated):** `remyxai/style-aligned-flux-modular`
**Block class:** `StyleAlignedFluxBlock` · **Workflow axis:** style / consistency · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`.

## One-liner
Generate a **set** of images that share one coherent style, training-free, by sharing attention across the
batch (StyleAligned). The SDXL version already ships in diffusers
(`pipeline_sdxl_style_aligned.py`) — **FLUX is the gap**; port the shared-attention to FLUX's MMDiT.

## Vetting (done)
- **Base to port to:** FLUX (MMDiT). Reference is SDXL. ✓ image, single-A100.
- **License/repo:** `google/style-aligned` — **Apache-2.0**, ⭐1315. ✓
- **Not in diffusers for FLUX:** SDXL pipeline exists; no FLUX variant (code hits are the SDXL one) → **gap**. ✓
- **In-flight:** none for a FLUX StyleAligned. ✓
- Paper: "Style Aligned Image Generation via Shared Attention" (arXiv 2312.02133 — **[VERIFY] authors**).

## Mechanism (to port from SDXL → FLUX)
StyleAligned makes every image in a batch attend to a shared **reference** (batch item 0): it shares K/V from
the reference and applies **AdaIN** to align Q/K statistics, so the set is style-consistent without training.
The SDXL impl swaps the attention processor. Port = a **FLUX joint-attention processor** that shares K/V across
the batch (ref = index 0) with AdaIN, on the image-stream attention.

## Modular block design
- **Injection pattern:** *attention-processor swap* (HRDiT pattern) — install a shared-attention FluxAttnProcessor;
  restore in `finally`. Must be a **no-op when sharing is disabled** (== independent generations).
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `prompt`(required; str or list), `num_images`(default 3, if prompt is str), `reference_prompt`(optional
  = batch item 0), `share_group_norm`/`share_attention`(True), `guidance_scale`(3.5), `num_inference_steps`(28),
  `height`/`width`(1024), `generator`, `output_type`.
- `__call__`: build the batch of prompts → install shared-attn processor → standard FLUX denoise over the batch
  → decode → `bs.images=[...]` (the set).

## Tests
- **smoke.ipynb:** publish PRIVATE → load (assert `StyleAlignedFluxBlock`) → generate 2 images at low res/few steps
  → returns 2 images.
- **e2e.ipynb:** generate a 4-image set from varied prompts with sharing ON vs OFF; **validate style consistency**
  (ON = shared palette/texture across the set; OFF = independent) — a CLIP/Gram style-similarity delta is a good
  quantitative signal. **Spike first:** the FLUX joint-attention seam (text+image tokens in one attention) — does
  shared K/V + AdaIN across the batch produce consistency and stay a no-op when off?

## Risks / de-risk
1. **Main risk:** SDXL cross-attention ≠ FLUX joint-attention (MMDiT mixes text+image). The AdaIN + shared-K/V
   scheme must be re-derived for the joint stream — the spike gates it.
2. Which blocks to share on (all vs a subset) for quality.

## Deliverables
`block.py`, `modular_config.json`, `modular_model_index.json`, `smoke.ipynb`, `e2e.ipynb`, `README.md`.
