---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - subject-driven-generation
  - personalization
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# "Flux Already Knows" for FLUX — training-free subject-driven generation (Modular Diffusers custom block)

**🤗 Hub:** [remyxai/flux-subject-flux-modular](https://huggingface.co/remyxai/flux-subject-flux-modular) · **📄 Paper:** [arXiv:2504.11478](https://arxiv.org/abs/2504.11478) · **💻 Reference:** [bytedance/LatentUnfold](https://github.com/bytedance/LatentUnfold) · **📦 Monorepo:** [flux-modular-diffusers](https://github.com/remyxai/flux-modular-diffusers)

Put a **reference subject** — an object, a product, a character — into a brand-new scene with
**no training, no LoRA and no extra weights**, as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks) custom
block. Implements **LatentUnfold** ("Flux Already Knows", [arXiv:2504.11478](https://arxiv.org/abs/2504.11478))
on FLUX: the subject image is replicated as the tiles of a mosaic latent and the scene is generated into the
remaining tile, so FLUX sees the subject as plain image tokens and reproduces it. Cascade attention then
up-weights attention from the generated region onto the reference tiles.

This is the third leg of the personalization set: **PuLID** needs trained *face* weights, **CatVTON** covers
*garments* — this one is **arbitrary subjects, weight-free**.

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/flux-subject-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

image = pipe(
    subject_image="clock.png",                      # the reference subject (PIL / path / list)
    prompt="On a wooden desk in a cozy study, next to a stack of books and a cup of coffee.",
    subject_prompt="bright yellow retro alarm clock",   # one short description, for the meta prompt
    height=512, width=512,
).images[0]
image.save("scene.png")
```

`prompt` is the **new scene** (it lands in the meta prompt's `[IMAGE1]` slot). A `prompt` that already
contains `[IMAGE` is used verbatim, so the reference's hand-written prompts reproduce exactly.
`grid_shape=(1, 1)` drops the reference tiles entirely and runs stock FLUX text-to-image.

## How it works

1. **Mosaic layout** — the subject is centre-cropped square, resized to the tile size and VAE-encoded;
   the latents are tiled into every cell of a `grid_shape` mosaic except `(0, 0)`, which is left white.
   The scene is denoised *into* `(0, 0)` only, with the reference tiles re-noised to each step's level and
   held fixed. The subject therefore conditions generation as ordinary image tokens — no injection module.
2. **Cascade attention** — a swapped joint-attention processor average-pools the image-stream K/Q by each
   factor in `cascade`, re-rotates them with a RoPE built on the downsampled position grid, and adds the
   pooled attention map (scaled by `subject_strength / factor / len(cascade)`) to the generated tile's
   attention over the reference tiles. This raises subject fidelity without touching any weights.
3. **Meta prompting** — one caption per tile (`[IMAGE1]` = the user's scene, `[IMAGEk]` = a view of the
   subject), so T5 describes the whole mosaic. Captions come from a fixed rotation; pass a hand-written
   `[IMAGE…]` prompt to override.
4. **Decode tile `(0, 0)`** — only the generated tile is decoded and returned.

The processor is threaded through `joint_attention_kwargs['subject_cascade']` (named kwarg, no module
globals — concurrency-safe) and restored in a `finally`. `subject_strength=0.0`, an inactive layer, or
`cascade=()` falls through to stock FLUX attention — a bit-exact no-op.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `subject_image` | — | the reference subject: PIL / path / list (multi-view) |
| `prompt` | — | the **new scene** |
| `subject_prompt` | None | one short subject description, for the meta prompt |
| `grid_shape` | (3, 3) | mosaic rows/cols; `(1,1)` = stock FLUX text-to-image |
| `subject_strength` | 0.05 | cascade weight (the reference's `aug_att`); `0.0` = pure mosaic |
| `cascade` | (2, 3) | pooling factors; `()` disables cascade attention |
| `injection_steps` | 14 | apply cascade attention over the first N steps only |
| `cascade_start_frac` | 0.0 | skip the first frac of dual-stream layers (fidelity/prompt dial) |
| `remove_background` | False | opt-in RMBG-2.0 subject cutout (adds weights) |
| `guidance_scale` | 7.0 | the reference's value (not FLUX's usual 3.5) |
| `height` / `width` | 512 | size of the **output tile** |

**Fidelity vs. prompt adherence** is the same trade-off StyleAligned and StoryDiffusion hit: raise
`subject_strength` and the subject locks in while the scene drifts. `0.02–0.05` is the plateau; `0.2`+ is
where the prompt starts being ignored. `cascade_start_frac≈0.3` biases toward the prompt if the subject
over-dominates.

**VRAM.** Unlike stock FLUX (which uses fused SDPA), the cascade path materializes the full attention matrix
in order to add to it — ~4.2 GB of activation per cascade layer at the default `grid_shape=(3,3)` with 512px
tiles, on top of the ~24 GB FLUX.1-dev weights. On a 40 GB A100 use `height=width=384` (~1.5 GB) or
`grid_shape=(2,2)` with 512px tiles (~1 GB). `subject_strength=0.0` and `cascade=()` avoid the cost entirely.

## Dependencies

`diffusers>=0.40.0` (main, for `ModularPipeline`), `torch>=2.4.0`, `transformers`, `accelerate`. Optional:
`briaai/RMBG-2.0` when `remove_background=True` (that checkpoint carries its own non-commercial license —
off by default, so the default path adds **no** weights).

## Attribution & AI assistance

FLUX port of **LatentUnfold** ([bytedance/LatentUnfold](https://github.com/bytedance/LatentUnfold),
Apache-2.0) by Kang et al. Authored with AI assistance (Claude) and validated by the Remyx AI team.
Uses FLUX.1-dev under its **non-commercial** license, which this derivative inherits.

## Citation

```bibtex
@article{kang2025latentunfold,
  title={Flux Already Knows - Activating Subject-Driven Image Generation without Training},
  author={Kang, Hao and Fotiadis, Stathi and Jiang, Liming and Yan, Qing and Jia, Yumin and Liu, Zichuan and Chong, Min Jin and Lu, Xin},
  journal={arXiv preprint arXiv:2504.11478},
  year={2025}
}
```
