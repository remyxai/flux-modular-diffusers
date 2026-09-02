---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - appearance-transfer
  - style-transfer
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# Appearance Transfer for FLUX — training-free reference appearance/texture transfer (Modular Diffusers custom block)

**🤗 Hub:** [remyxai/appearance-transfer-flux-modular](https://huggingface.co/remyxai/appearance-transfer-flux-modular) · **📄 Paper:** [arXiv:2603.26767](https://arxiv.org/abs/2603.26767) · **📦 Monorepo:** [flux-modular-diffusers](https://github.com/remyxai/flux-modular-diffusers)

![Appearance transfer — the astronaut's structure preserved, re-rendered in the reference cup's ceramic/espresso material](assets/appearance_transfer_hero.png)

Transfer a **reference image's appearance** (color, texture, material) onto a **source image** while
**preserving the source's geometry** — no fine-tuning, no new weights — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks)
custom block. Clean-room implementation of **"A Training-Free Framework for High-Fidelity Appearance
Transfer via Diffusion Transformers"** ([arXiv:2603.26767](https://arxiv.org/abs/2603.26767)), built on
**FLUX.1-dev-Depth + a mask-weighted FLUX.1-Redux**. Unlike our
[regional-prompting](https://huggingface.co/remyxai/regional-prompting-flux-modular) (soft prompt routing)
or [stitch](https://huggingface.co/remyxai/stitch-flux-modular) (bounding-box placement) blocks, this is
**reference-driven material/texture transfer**.

## Usage

```python
import torch
from diffusers import ModularPipeline
from PIL import Image

pipe = ModularPipeline.from_pretrained("remyxai/appearance-transfer-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")   # ~36GB resident; prefer an 80GB A100 (this block runs ~170 transformer passes)

img = pipe(
    source_image=Image.open("source.png"),        # geometry to keep
    reference_image=Image.open("reference.png"),   # appearance to transfer
    reference_mask=None,                            # optional HxW [0,1] fg mask -> texture, not shape
    blend_k=0.25,                                   # lower = more appearance, higher = more structure
    height=1024, width=1024,
    output="images",
).images[0]
img.save("appearance_transfer.png")
```

Pass `reference_image=None` and the block is a **bit-exact identity** (returns the source unchanged).
No text prompt is required — appearance comes from the reference.

## How it works

Three training-free ingredients on frozen FLUX.1-Depth (all validated on an A100):

1. **Source inversion (structure prior).** The source is inverted along the FLUX.1-Depth flow with a
   **matched 2nd-order (RK2/midpoint) solver at guidance 1.0**, giving a content-rich trajectory. A
   **blended initialization** replays this trajectory for the first `blend_k` fraction of steps, locking
   the source geometry; depth control anchors it every step.
2. **Mask-weighted Redux (appearance).** The reference is encoded through the FLUX.1 Redux image encoder;
   its SigLIP patch tokens are **down-weighted by the reference foreground mask** to suppress the
   reference's *shape* and keep its *texture*. This global embedding conditions generation throughout.
3. **Attention context expansion (fine detail).** The reference's image-token Keys/Values are captured and
   **concatenated onto the source's** at the first-2 / last-2 blocks of both FLUX streams, so each source
   patch attends cross-image to the reference's appearance library. Query tokens are unchanged; capture is
   image-token-only so it composes with the Redux-extended conditioning.

The attention processors are swapped for the call and **restored in `finally`**; with `reference_image`
empty none are installed.

**Validated (e2e, 50 steps):** structure preserved (source↔result depth-map correlation **0.82–0.98**) with
appearance moved toward the reference (**+0.12 to +0.35** CLIP-to-reference gain). `blend_k=0.25` balances
the two; strong-structure subjects tolerate `0.2` (more transfer), delicate close-ups prefer `0.3–0.4`.

**Compute:** two RK2 inversions + generation ≈ ~170 transformer passes (~2–3× a stock FLUX.1-Depth run).
Comfortable on an 80GB A100 with all components resident; **CPU offload thrashes** at this call count.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `source_image` | — | image whose geometry is preserved (required) |
| `reference_image` | None | image whose appearance is transferred; `None` → identity (returns source) |
| `reference_mask` | None | HxW mask in `[0,1]`; down-weights background reference patches (texture, not shape) |
| `blend_k` | 0.25 | fraction of steps replaying the source (structure lock); **lower = more appearance, higher = more structure** |
| `num_inference_steps` | 50 | total steps (matched RK2 both directions) |
| `guidance_scale` | 10.0 | FLUX.1-Depth generation guidance |
| `invert_guidance` | 1.0 | RF-inversion guidance (best reconstruction) |
| `use_kv_injection` | True | attention appearance channel (secondary; fine detail) |
| `redux_mask_floor` | 0.1 | background patch weight floor for mask-weighted Redux |
| `height` / `width` | 1024 | canvas size |

## Dependencies

`diffusers` (main / ≥ 0.41), `transformers`, `accelerate`, `sentencepiece`, `protobuf`. A depth estimator
(`depth-anything/Depth-Anything-V2-Small-hf`) is lazily loaded for the source depth. Components:
**FLUX.1-Depth-dev** (transformer/vae/text-encoders/scheduler) + **FLUX.1-Redux-dev** (image encoder +
embedder). No new weights are trained. **Prompt → source/reference is up to the caller** (like a
reference-image API).

## Attribution & AI assistance

**Method** by Shengrong Gu, Ye Wang, Song Wu, Rui Ma, Qian Wang, Lanjun Wang, and Zili Yi
([arXiv:2603.26767](https://arxiv.org/abs/2603.26767)). **Clean-room implementation:** no reference code was
released, so this block was written from the paper's description alone (same discipline as our other FLUX
ports; the method itself is not copyrightable). Authored with AI assistance (Claude) and validated by the
Remyx AI team; method credit to the authors above. Uses **FLUX.1-dev-Depth** and **FLUX.1-Redux-dev** under
their **non-commercial** license — this derivative inherits that restriction.

## Citation

```bibtex
@misc{gu2026appearancetransfer,
  title={A Training-Free Framework for High-Fidelity Appearance Transfer via Diffusion Transformers},
  author={Gu, Shengrong and Wang, Ye and Wu, Song and Ma, Rui and Wang, Qian and Wang, Lanjun and Yi, Zili},
  year={2026}, eprint={2603.26767}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.26767}
}
```
