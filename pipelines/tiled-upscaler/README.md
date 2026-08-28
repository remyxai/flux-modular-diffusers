---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - upscaling
  - super-resolution
  - image-to-image
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# Tiled Creative Upscaler for FLUX — training-free detail-adding upscaling (Modular Diffusers custom block)

Upscale an image **and** add detail — no fine-tuning, no extra weights — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks) custom block.
Off-the-shelf **FLUX.1-dev** img2img is run tile-by-tile over the upscaled image at low-moderate denoise, so the
model re-renderings each region *as if it had been generated at the target resolution*: edges resharpen, textures
reappear, and JPEG/resize softness goes away — detail far beyond a plain Lanczos resize. And because the canvas is
processed one tile at a time, peak activation memory stays flat no matter the output size.

![low-res → upscaled](assets/tiled_upscaler_demo.png)

<sub>Left: the low-res source. Right: ×2 tiled upscale (`denoise_strength=0.4`) — texture and edge detail restored,
no visible tile seams. Training-free, no extra weights.</sub>

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/tiled-upscaler-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

big = pipe(
    image="photo.jpg",            # the low-res source
    scale=2,                      # 2x per side (4x works too — see the e2e notebook)
    denoise_strength=0.4,         # 0.3–0.5: enough to add detail, not enough to redraw
    num_inference_steps=28, guidance_scale=3.5,
).images[0]
big.save("upscaled.png")
```

**Dependencies:** `pip install transformers accelerate sentencepiece protobuf`. Nothing beyond the 7 FLUX.1-dev
components — the block is fully training-free and downloads no extra weights.

## How it works

1. **Upscale** the source ×`scale` with Lanczos (a *smooth* base: we want structure, not jaggies).
2. **Tile** the upscaled canvas into `tile_size` squares overlapping by `tile_overlap`.
3. **Refine each tile with FLUX img2img**: encode the tile, add noise at `denoise_strength`, then denoise the
   remaining steps toward the detail prompt. Partial noising is what makes this *creative* upscaling rather than
   a copy — it regenerates plausible high-frequency detail consistent with the prompt while the surviving
   low-frequency signal pins the content.
4. **Feather-blend** the refined tile back into the canvas with a 2-D ramp weight (1 in the interior, ramping to
   0 at the borders). Tiles accumulate into a numerator and a per-pixel weight-sum, then divide — so an overlap
   region is the *weight-normalized average* of its tiles. This is the main quality lever: it is what makes the
   seams invisible.

Nothing is mutated on the transformer (no hooks, no processor swap, no `pos_embed` patch) — it is a plain custom
denoise loop, so there is no seam to restore. Prompt embeds are computed once and shared by every tile.

**Limitations:** global coherence weakens as `denoise_strength` rises past ~0.5 (each tile starts to invent its own
content) — keep it moderate, and prefer a detail prompt that describes *texture*, not new subjects. A modest
`tile_size` (≤1024) is also the FLUX sweet spot, since the model was trained at that scale.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `image` | — | the low-res source image (path / PIL / numpy RGB) |
| `scale` | 2 | upscale factor per side |
| `denoise_strength` | 0.4 | per-tile img2img strength — **the** creativity knob (0.3–0.5 recommended) |
| `tile_size` | 1024 | per-tile side in px (snapped to a /16 multiple) |
| `tile_overlap` | 128 | overlap between neighbouring tiles, px — ↑ = smoother seams, ↑ cost |
| `prompt` | `None` | detail prompt; `None` uses a built-in sharp-detail prompt |
| `num_inference_steps` | 28 | steps per tile (effective steps ≈ `strength × steps`) |
| `guidance_scale` | 3.5 | FLUX guidance |
| `resample_filter` | `lanczos` | the base upscale resampler (`lanczos` / `bicubic` / `bilinear`) |

## Attribution & AI assistance

Training-free reimplementation for Modular Diffusers of the tiled img2img upscaling workflow, after
**[neuralwork/flux-tiled-upscaler](https://github.com/neuralwork/flux-tiled-upscaler)** (MIT) on FLUX.1-dev.
The workflow has **no single paper**; its lineage is the Stable Diffusion tiled-upscaling community pipeline
([diffusers `tiled_upscaling.py`](https://github.com/huggingface/diffusers/blob/main/examples/community/tiled_upscaling.py))
plus SDEdit-style partial noising for the per-tile img2img step. The Modular-Diffusers adaptation was authored
with AI assistance (Claude) and validated by the Remyx AI team. Uses FLUX.1-dev under its **non-commercial**
license.

## References

No paper anchors this method, so there is no verified citation — the accurate lineage is:

- `neuralwork/flux-tiled-upscaler` (MIT) — the direct FLUX reference implementation.
- [Meng et al., *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations*,
  arXiv:2108.01073](https://arxiv.org/abs/2108.01073) — the img2img partial-noising formulation each tile uses.

```bibtex
@misc{neuralwork2025fluxtiledupscaler,
  author = {{neuralwork}},
  title = {flux-tiled-upscaler},
  year = {2025},
  howpublished = {\url{https://github.com/neuralwork/flux-tiled-upscaler}},
  note = {MIT license}
}
@article{meng2022sdedit,
  title={SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations},
  author={Meng, Chenlin and Song, Yang and Song, Jiaming and Wu, Jiajun and Zhu, Jun-Yan and Ermon, Stefano},
  journal={arXiv preprint arXiv:2108.01073},
  year={2022}
}
```
