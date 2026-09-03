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

# KV-Edit for FLUX — training-free editing with pixel-precise background preservation (Modular Diffusers custom block)

**🤗 Hub:** [remyxai/kv-edit-flux-modular](https://huggingface.co/remyxai/kv-edit-flux-modular) · **📄 Paper:** [arXiv:2502.17363](https://arxiv.org/abs/2502.17363) · **💻 Reference:** [Xilluill/KV-Edit](https://github.com/Xilluill/KV-Edit) · **📦 Monorepo:** [flux-recipes](https://github.com/remyxai/flux-recipes)

Text-based image editing on off-the-shelf **FLUX.1-dev** — **no training, no extra weights** — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks)
custom block. Implements **KV-Edit** ([arXiv:2502.17363](https://arxiv.org/abs/2502.17363)): the source
image is rectified-flow inverted once under its source prompt while every attention block caches the
K/V of the **background** tokens; the edit then denoises under the target prompt with those cached
background K/V substituted in place. The unedited background is **preserved, not re-synthesized** —
the failure mode this fixes vs. [FlowEdit](https://huggingface.co/remyxai/flowedit-flux-modular)
(structure-preserving, but the background still drifts).

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/kv-edit-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

edited = pipe(
    image="photo.png",
    mask="mask.png",             # white = region to edit, black = keep pixel-precise
    source_prompt="a cat sitting on a sofa",   # describes the source image
    prompt="a dog sitting on a sofa",          # the target edit
    T_steps=28, guidance_scale=3.5,
).images[0]
edited.save("edited.png")
```

**Tip:** `mask` is optional — with `mask=None` the whole image is editable and the K/V seam is
disarmed (bit-exact stock FLUX attention, i.e. plain RF-inversion editing). A good `source_prompt`
(describing what is *actually* in the image) is what makes the inversion — and therefore the
background fidelity — faithful. Invert once, edit many times: the cached K/V are per image+mask.

## How it works

1. **Invert** the source image with rectified-flow inversion under the `source_prompt`, running
   stock FLUX attention while a custom attention processor *captures* the normalized K/V of the
   background image tokens (the `mask == keep` region) at every block and step.
2. **Denoise** from the inverted latent under the target `prompt`: the processor *substitutes* the
   cached background K/V in place of the current ones — equivalent to the reference's "concat the
   background K/V memory with the foreground content" (softmax over keys is permutation-invariant).
3. **Decode.** Only the masked region is regenerated; background tokens attend to their own cached
   state, so the background is retained exactly rather than re-synthesized.

The seam is an attention-processor swap threaded through `joint_attention_kwargs['kv_edit']`
(concurrency-safe, no globals); the original processors are restored in a `finally`. Captured K/V
are kept on CPU (O(#background tokens) per block per step) and sliced back per denoise step.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `image` | — | source image (path / PIL / numpy RGB) |
| `mask` | `None` | edit region: white (255) = edit, black = keep; `None` = whole-image edit |
| `source_prompt` | — | description of the source image |
| `prompt` | — | the target edit |
| `guidance_scale` | 3.5 | edit (target) guidance (↑ = stronger edit) |
| `src_guidance_scale` | 1.0 | inversion guidance (low = more faithful inversion/background) |
| `T_steps` | 28 | inversion + denoise steps |
| `height` / `width` | source size | snapped to multiples of 16 |

## Attribution & AI assistance

Training-free reimplementation for Modular Diffusers of **KV-Edit** (Apache-2.0,
[Xilluill/KV-Edit](https://github.com/Xilluill/KV-Edit)). The Modular-Diffusers adaptation was
authored with AI assistance (Claude) and validated by the Remyx AI team; method credit to the
KV-Edit authors. Uses FLUX.1-dev under its **non-commercial** license.

## Citation

```bibtex
@misc{zhu2025kvedit,
  title={KV-Edit: Training-Free Image Editing for Precise Background Preservation},
  author={Zhu, Tianrui and Zhang, Shiyi and Shao, Jiawei and Tang, Yansong},
  year={2025},
  eprint={2502.17363}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.17363}
}
```
