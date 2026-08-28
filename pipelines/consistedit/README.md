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

# ConsistEdit for FLUX — highly consistent, precise training-free editing (Modular Diffusers custom block)

**🤗 Hub:** [remyxai/consistedit-flux-modular](https://huggingface.co/remyxai/consistedit-flux-modular) · **📄 Paper:** [arXiv:2510.17803](https://arxiv.org/abs/2510.17803) · **💻 Reference:** [zxYin/ConsistEdit_Code](https://github.com/zxYin/ConsistEdit_Code) (Apache-2.0) · **📦 Monorepo:** [flux-modular-diffusers](https://github.com/remyxai/flux-modular-diffusers)

Text-based image editing on off-the-shelf **FLUX.1-dev** — **no training, no extra weights** — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks)
custom block. Implements **ConsistEdit** ([arXiv:2510.17803](https://arxiv.org/abs/2510.17803),
SIGGRAPH Asia 2025): an attention-control method **designed for MM-DiT**, the architecture FLUX
uses. The source image is rectified-flow inverted once under its source prompt while FLUX's
**single blocks** cache their vision tokens; the edit denoises from that latent under the target
prompt with the cached source tokens fused in *before* attention.

This is the **FLUX-native** editing option in the catalog, next to
[FlowEdit](https://huggingface.co/remyxai/flowedit-flux-modular) (inversion-free, structure-preserving,
but the background drifts) and [KV-Edit](https://huggingface.co/remyxai/kv-edit-flux-modular)
(pixel-precise background via cached K/V, but no structure control *inside* the edit region).
ConsistEdit's distinction is **smoothly adjustable structural consistency within the edited region**:
at high `consistency_strength` the edit keeps the source's structure even against shape-changing
prompts (texture/colour edits still land); at low strength the target prompt is free to change shape —
and the non-edited region stays intact either way.

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/consistedit-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

edited = pipe(
    image="photo.png",
    mask="mask.png",             # white = region to edit, black = keep intact
    source_prompt="a cat sitting on a sofa",   # describes the source image
    prompt="a dog sitting on a sofa",          # the target edit
    consistency_strength=0.3,   # 0.3 (default) = clean edit + kept structure; 0.6-1.0 barely edits
    T_steps=28, guidance_scale=3.5,
).images[0]
edited.save("edited.png")
```

**Tip:** `mask` is optional — with `mask=None` the whole image is editable and the attention seam is
disarmed (bit-exact stock FLUX attention, i.e. plain RF-inversion editing). Use a mask whenever you
want the non-edited region held, and pick `consistency_strength` by task: **~0.3** (default) lands a
clean edit while holding structure + background; **0.6–1.0** preserve so strongly the shape barely
changes — use them for recolor / texture tweaks that must keep the exact silhouette; **0** removes the
control entirely (can over-edit / look cartoonish).

## How it works

1. **Invert** the source image with rectified-flow inversion under the `source_prompt`, keeping the
   **latent trajectory** (a few MB) — the reference pass that defines the source's token states.
2. **Denoise** from the inverted latent under the target `prompt`. At each step the block first
   replays the *source* branch at that step to *capture* its vision tokens, then *fuses* them into the
   target branch's tokens **before** attention — the paper's mask-guided pre-attention fusion (Eq. 5):
   - **Q, K carry structure.** Outside the edit mask the source Q/K replace the target's; inside the
     mask the source Q/K are enforced too, for the first `consistency_strength`·T steps. That step
     ratio *is* the consistency strength — α=1 enforces the source structure for the whole trajectory,
     α→0 hands shape back to the prompt.
   - **V carries content.** The source V replaces the target's outside the mask only (using the source
     V everywhere caused colour shifts; the target V everywhere broke the non-edited content).
3. **Decode.** The text parts of Q/K/V are never touched — ConsistEdit's "vision-only" finding is that
   interfering with text tokens destabilizes generation — and control is applied to every step of every
   edited layer, because each MM-DiT layer retains rich semantics (unlike U-Net's stage separation).

The paper targets the **single blocks** for FLUX specifically (they carry the general generation
information), so the processor is installed only on `single_transformer_blocks` — the double blocks
keep exactly their stock processors. The state is threaded through
`joint_attention_kwargs['consistedit']` (named kwarg; no module globals, concurrency-safe), and all
processors are restored in a `finally`.

Caching every block's tokens for every step would need 20–80 GB, so only the latent trajectory is
kept and the source branch is replayed one step at a time: **one step** of tokens (~0.7 GB at 512²,
~2.9 GB at 1024², held on CPU) is live at any moment. The price is three transformer passes per step
instead of two.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `image` | — | source image (path / PIL / numpy RGB) |
| `source_prompt` | — | description of the source image |
| `prompt` | — | the target edit |
| `mask` | `None` | edit region: white (255) = edit, black = keep; `None` = whole image editable |
| `consistency_strength` | 0.3 | α — ratio of denoise steps that enforce the source Q/K in the edit region. **0.3 (default)** = clean edit while structure/background hold; 0.6–1.0 = strong preservation (edit barely changes shape); 0.0 = no control (stock denoise, can over-edit) |
| `guidance_scale` | 3.5 | edit (target) guidance (↑ = stronger edit) |
| `src_guidance_scale` | 1.0 | inversion guidance (low = more faithful inversion) |
| `T_steps` | 28 | inversion + denoise steps |
| `height` / `width` | source size | snapped to multiples of 16 |

## Dependencies

`diffusers>=0.40.0`, `torch>=2.4.0`, `transformers`, `accelerate` — nothing beyond the FLUX.1-dev
components the block already declares. No extra weights are downloaded; the method is training-free.

## Attribution & AI assistance

Training-free reimplementation for Modular Diffusers of **ConsistEdit** (Apache-2.0,
[zxYin/ConsistEdit_Code](https://github.com/zxYin/ConsistEdit_Code)) by Zixin Yin, Ling-Hao Chen,
Lionel Ni and Xili Dai (SIGGRAPH Asia 2025). The Modular-Diffusers adaptation was authored with AI
assistance (Claude) and validated by the Remyx AI team; method credit to the ConsistEdit authors.
Uses FLUX.1-dev under its **non-commercial** license.

## Citation

```bibtex
@inproceedings{yin2025consistedit,
  title={ConsistEdit: Highly Consistent and Precise Training-free Visual Editing},
  author={Yin, Zixin and Chen, Ling-Hao and Ni, Lionel and Dai, Xili},
  booktitle={SIGGRAPH Asia 2025 Conference Papers},
  year={2025},
  eprint={2510.17803},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2510.17803}
}
```

This repository is a training-free reimplementation for Modular Diffusers on off-the-shelf FLUX.1;
all credit for the method goes to the authors above.
