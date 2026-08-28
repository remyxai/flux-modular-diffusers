# Brief — FLUX Outpainting → Modular Diffusers

**Target repo (private until validated):** `remyxai/outpaint-flux-modular`
**Block class:** `OutpaintBlock` · **Axis:** processing (extend) · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`. Reuses the CatVTON FLUX-Fill plumbing.

## One-liner
Extend an image beyond its borders (outpaint) — training-free — by placing it on a larger canvas and inpainting
the new margins with **FLUX.1-Fill**. Same family as an object-removal companion (Fill + mask).

## Vetting (done)
- **Base:** **FLUX.1-Fill-dev** (transformer) + FLUX.1-dev (rest) — the CatVTON component setup. ✓
- **Mode/license:** training-free (no weights beyond FLUX Fill); non-commercial (FLUX-dev/Fill-dev). ✓
- **Not in diffusers:** Fill *inpaint* exists; **no FLUX outpaint pipeline** → **gap**. ✓ (ref `alexgenovese/flux-outpainting` low-star → build clean, not port)

## Mechanism
Paste the source onto a larger canvas at a chosen position (or symmetric margins), build a mask that is 0 over the
original and 1 over the new margins, and run FLUX Fill conditioned on a prompt to synthesize coherent extensions.

## Modular block design
- **Injection pattern:** *compose a `FluxFillPipeline` from components* (the CatVTON pattern — no hooks). No LoRA.
- `expected_components`: text_encoder(s)/tokenizer(s)/vae/scheduler from FLUX.1-dev + **transformer from FLUX.1-Fill-dev**.
- `inputs`: `image`, `prompt`(scene for the new area), `left`/`right`/`top`/`bottom` (px or fraction to extend) OR
  `target_height`/`target_width`, `num_inference_steps`, `guidance_scale`(~30 Fill), `generator`, `output_type`.
- `__call__`: build the enlarged canvas + margin mask → `FluxFillPipeline(image=canvas, mask_image=mask, ...)` → return.

## Tests
- **smoke.ipynb:** load (assert `OutpaintBlock`) → extend a small image by 256px on one side → returns larger image.
- **e2e.ipynb:** extend a photo on all sides; **validate** the original region is untouched (pixel-equal in the
  unmasked area) and the new margins are coherent/seamless. No attention spike needed (Fill compose).

## Risks
1. Seam at the original/new boundary → slight mask feather / overlap.
2. Non-commercial license (state it).

## Deliverables
`block.py` · configs · `smoke.ipynb` · `e2e.ipynb` · `README.md`. (Companion: an `object-removal` variant = Fill + a supplied/auto object mask — trivial extension of this block.)
