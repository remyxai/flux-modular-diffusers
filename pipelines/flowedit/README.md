---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - image-editing
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# FlowEdit for FLUX — training-free, inversion-free image editing (Modular Diffusers custom block)

Text-based image editing on off-the-shelf **FLUX.1-dev** — **no inversion, no fine-tuning** — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks)
custom block. Implements **FlowEdit** ([arXiv:2412.08629](https://arxiv.org/abs/2412.08629)): instead of
inverting the source image, it builds an ODE that transports it *directly* from the source prompt to the
target prompt — more faithful to the original (structure preserved) and faster than inversion-based editing.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10u1UtrxG6gV9ubmfnp9cBoodVHpQEug0?usp=sharing) — edit your own photo by changing the prompt.

![source → edits, structure preserved](assets/flowedit_demo.png)

<sub>One source image, two edits (matched source prompt → target prompt) — the subject changes, the pose /
composition / background are preserved. Training-free, no inversion.</sub>

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/flowedit-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

edited = pipe(
    image="cat.png",
    source_prompt="a cat",       # describes the source image
    prompt="a dog",              # the target edit
    T_steps=28, src_guidance_scale=1.5, tar_guidance_scale=5.5, n_max=24, n_min=0,
).images[0]
edited.save("edited.png")
```

**Tip:** the `source_prompt` should describe what's *actually in the image*; the `prompt` is the edit. A good
`source_prompt` is what makes the edit faithful. No extra weights or models — it's fully training-free.

## How it works

FlowEdit is **inversion-free**. At each step it draws noise to form a source point `zt_src` and a coupled
target point `zt_tar`, computes the model's velocity for each under the source and target prompts, and
integrates the **guided velocity difference** `Vt_tar − Vt_src` into a running edit latent — over a step window
(`n_max … n_min`), averaged over `n_avg` noise draws. Because it never inverts the image, the edit stays
anchored to the original structure. A self-contained custom denoise loop; the base weights are untouched.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `image` | — | source image (path / PIL / numpy RGB) |
| `source_prompt` | — | description of the source image |
| `prompt` | — | the target edit |
| `tar_guidance_scale` | 5.5 | edit strength (↑ = stronger edit) |
| `src_guidance_scale` | 1.5 | source guidance |
| `T_steps` | 28 | total steps |
| `n_max` / `n_min` | 24 / 0 | step window where the edit ODE is applied (raise `n_min` for an SDEdit-style tail) |
| `n_avg` | 1 | velocity-difference averaging (↑ = smoother, slower) |

## Attribution & AI assistance

Training-free reimplementation for Modular Diffusers of **FlowEdit** (MIT,
[fallenshock/FlowEdit](https://github.com/fallenshock/FlowEdit)). The Modular-Diffusers adaptation was
authored with AI assistance (Claude) and validated by the Remyx AI team; method credit to the FlowEdit
authors. Uses FLUX.1-dev under its **non-commercial** license.

## Citation

```bibtex
@misc{kulikov2024flowedit,
  title={FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models},
  author={Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
  year={2024},
  eprint={2412.08629}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.08629}
}
```
