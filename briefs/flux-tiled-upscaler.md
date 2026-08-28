# Brief — FLUX Tiled Creative Upscaler → Modular Diffusers

**Target repo (private until validated):** `remyxai/tiled-upscaler-flux-modular`
**Block class:** `TiledUpscalerBlock` · **Axis:** processing (upscale/restore) · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`.

## One-liner
Creative upscaling: upscale an image, split into overlapping tiles, **refine each tile with FLUX img2img**, and
blend — adds detail well beyond a plain resize. Training-free; no extra weights.

## Vetting (done)
- **Base:** FLUX.1-dev (img2img). ✓ image, single-A100 (tile-by-tile, memory-friendly).
- **License/repo:** ref [`neuralwork/flux-tiled-upscaler`](https://github.com/neuralwork/flux-tiled-upscaler) — **MIT**. ✓
- **Not in diffusers:** SD `tiled_upscaling.py` exists; **no FLUX** tiled upscaler → **gap**. ✓
- No paper (a workflow) — attribute the neuralwork ref + the SD tiled-upscaling lineage. **[VERIFY]** any paper before citing.

## Mechanism
Upscale (e.g., Lanczos ×2) → overlapping tiles → per-tile FLUX img2img at low-moderate `strength` (denoise=0.3–0.5)
with a detail prompt → feather/blend overlaps → recompose. Optional tile-conditioning on the original (structure).

## Modular block design
- **Injection pattern:** *custom loop* (no attention/hooks) — tile → FLUX img2img denoise → blend. Reuse the FLUX
  latent-prep/denoise plumbing (DyPE/FlowEdit templates) per tile.
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `image`, `scale`(2), `tile_size`(1024), `tile_overlap`(128), `denoise_strength`(0.4), `prompt`(detail,
  optional), `num_inference_steps`, `guidance_scale`, `generator`, `output_type`.
- `__call__`: upscale → for each tile: encode → partial-noise (strength) → denoise → decode → feather-blend into
  the canvas → return the upscaled image.

## Tests
- **smoke.ipynb:** load (assert `TiledUpscalerBlock`) → ×2 on a small image (few tiles/steps) → returns larger image.
- **e2e.ipynb:** ×2/×4 on a photo; **validate** output resolution + added detail (sharpness/Laplacian-var up) with no
  visible tile seams (check overlap blending). No spike needed (no attention seam) — but verify seam-free blending.

## Risks
1. Tile-seam artifacts → feathered/weighted blending in overlaps (the main quality lever).
2. Global coherence at high strength (keep denoise moderate).

## Deliverables
`block.py` · configs · `smoke.ipynb` · `e2e.ipynb` · `README.md`.
