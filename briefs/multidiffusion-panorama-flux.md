# Brief — MultiDiffusion Panorama → Modular Diffusers (training-free ultra-wide)

**Target repo (private until validated):** `remyxai/panorama-flux-modular`
**Block class:** `PanoramaBlock` · **Axis:** processing (extend / wide generation) · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`.

## One-liner
Generate ultra-wide / panoramic images beyond FLUX's native aspect — training-free — by fusing many overlapping
denoise windows (MultiDiffusion). No extra weights.

## Vetting (done)
- **Base:** FLUX.1-dev. ✓ image; memory-friendly (windowed).
- **License/repo:** ref [`omerbt/MultiDiffusion`](https://github.com/omerbt/MultiDiffusion) is **unlicensed** → **clean-room from the paper** (we already cite it in the regional card). ✓ method not copyrightable.
- **Not in diffusers:** no FLUX panorama/MultiDiffusion pipeline → **gap**. ✓
- Paper: MultiDiffusion (arXiv 2302.08113 — VERIFIED authors: Bar-Tal, Yariv, Lipman, Dekel).

## Mechanism (clean-room from paper)
Denoise a wide latent by, at each step, running the transformer on overlapping crops (windows) of the latent and
**averaging** each window's velocity/prediction back into the shared canvas (per-pixel averaging over windows) —
so the whole wide image is globally coherent. FLUX flow-match: average the per-window model outputs each step.

## Modular block design
- **Injection pattern:** *custom denoise loop* (no attention/hooks) — per step, for each window: slice packed
  latents + the window's img_ids → transformer → scatter-average predictions into the full canvas → scheduler.step.
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `prompt`, `height`(1024), `width`(e.g. 2048–4096, wide), `window` (native tile, 1024), `stride`(512),
  `num_inference_steps`, `guidance_scale`, `generator`, `output_type`.
- `__call__`: prepare a wide packed latent → per step, tile into overlapping windows (with per-window img_ids),
  run FLUX per window, average overlaps → step → decode. (Reuse FLUX pack/unpack + `_prepare_latent_image_ids` per window.)

## Tests
- **smoke.ipynb:** load (assert `PanoramaBlock`) → 1536×768 few steps → returns a wide image, no error.
- **e2e.ipynb:** a 3072×1024 panorama; **validate** coherent wide scene with no repetition/seams at window
  boundaries (the averaging is the quality lever). Note vs the "woven blob" failure — MultiDiffusion keeps each
  window at native res so the sigma schedule stays sane (unlike a naive wide gen).

## Risks
1. Per-window `img_ids` / positional handling in packed FLUX latents (windows must have correct relative positions).
2. Boundary repetition if stride/overlap is too small.

## Deliverables
`block.py` · configs · `smoke.ipynb` · `e2e.ipynb` · `README.md` (clean-room note + MultiDiffusion citation).
