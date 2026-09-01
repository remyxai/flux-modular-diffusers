---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - controllable-generation
  - layout-to-image
  - position-control
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# Stitch for FLUX — training-free bounding-box position control (Modular Diffusers custom block)

**🤗 Hub:** [remyxai/stitch-flux-modular](https://huggingface.co/remyxai/stitch-flux-modular) · **📄 Paper:** [arXiv:2509.26644](https://arxiv.org/abs/2509.26644) · **💻 Reference:** [ExplainableML/Stitch](https://github.com/ExplainableML/Stitch) *(unlicensed — see the clean-room note)* · **📦 Monorepo:** [flux-modular-diffusers](https://github.com/remyxai/flux-modular-diffusers)

Put **this object at this location** — no fine-tuning, no detector, no extra weights — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks)
custom block. Implements **Stitch** ([arXiv:2509.26644](https://arxiv.org/abs/2509.26644)): you give
a global prompt plus a few `(bounding box, sub-prompt)` regions, and each object is generated
**confined to its box**, cut out, composited onto the background, and blended. In the paper this
lifts FLUX's GenEval-Position score from **~0.22 to ~0.70** with zero training.

Off-the-shelf FLUX routinely fails simple spatial requests — ask for "a red cube to the left of a
blue sphere" and the objects swap, merge, or drift. Stitch fixes exactly that, and it is
**hard placement**, unlike our
[regional-prompting](https://huggingface.co/remyxai/regional-prompting-flux-modular) block, which
does *soft* per-region prompt routing (a base prompt plus per-region prompts in one pass, no boxes
and no compositing). Use Stitch when you know where things should go.

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/stitch-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

img = pipe(
    prompt="a red cube to the left of a blue sphere, on a plain studio backdrop",
    regions=[
        {"box": [0.05, 0.3, 0.45, 0.75], "prompt": "a red cube"},      # left
        {"box": [0.55, 0.3, 0.95, 0.75], "prompt": "a blue sphere"},   # right
    ],
    height=1024, width=1024,
).images[0]
img.save("stitch.png")
```

`box` is normalized `[x0, y0, x1, y1]` in `[0, 1]`, origin top-left, `y` down. Pass
`regions=None` (or `[]`) and the block is a **bit-exact no-op** against stock
`FluxPipeline` — no processor is installed and nothing is mutated.

## How it works

Three phases, all training-free on the frozen FLUX weights:

1. **Region Binding** (`region_bind_steps` = S = 10). For the first S steps the transformer runs
   **once per region, plus once for the background prompt** — K+1 passes per step, all started from
   the same noise so they stay registered token-for-token. In each object pass an **additive
   joint-attention mask** confines generation to the box: inside-box image tokens can't attend
   outside the box, outside-box tokens can't attend to the sub-prompt, and the sub-prompt can't
   attend outside the box. The mask is added to the attention logits before softmax; **Q/K/V
   projections are untouched**. The background pass runs unmasked.
2. **Cutout + composite** (at τ = S). For each object pass the block reads the **text→image
   attention of one fixed head** (`cutout_head`, default block 14 / head 20 for FLUX.1-dev), averages
   it over the non-pad text tokens, sorts descending, and keeps tokens until their cumulative mass
   reaches `cutout_eta` = 0.95 — then 2-D max-pools that token mask with `cutout_kernel` = 5 to close
   holes. Each object's foreground latent tokens are written into the background latent at the box.
3. **Refine** (steps S…T). The composite is denoised by a **single unmasked pass** on the full
   global prompt to the end, which blends the objects into one coherent scene. Decode.

Two implementation notes that matter on FLUX specifically: the joint attention matrix is
`(n_txt + n_img)`-square with text first, so the box mask is built on the **image block only** and
every box→token index is offset by `n_txt`; and packed FLUX latents are `(B, seq, 64)`, so
box mapping, foreground selection and compositing all index **tokens on dim 1**.

The attention processors are swapped for the duration of the call and **restored in `finally`**;
with `regions` empty none are installed at all.

**Compute:** Phase A costs **K+1 transformer passes per step for S steps** (K = number of regions).
At K ≤ 3 this is comfortable on a single A100; the passes are independent, so a VRAM-tight setup can
batch them instead of running them sequentially. A 2-region 50-step run is ≈ 10 steps × 3 passes +
40 steps × 1 pass ≈ 1.4× a stock 50-step generation.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `prompt` | — | global prompt for the whole scene (required) |
| `regions` | None | list of `{"box": [x0,y0,x1,y1], "prompt": str}` (normalized); `None`/`[]` → stock FLUX |
| `background_prompt` | `""` | background pass's prompt; `""` → derived from `prompt` |
| `region_bind_steps` | 10 | S: Region-Binding steps (paper Table 1) |
| `num_inference_steps` | 50 | T: total steps |
| `cutout_eta` | 0.95 | cumulative attention mass kept by the Cutout threshold |
| `cutout_kernel` | 5 | κ: 2-D max-pool kernel that solidifies the foreground mask |
| `cutout_head` | (14, 20) | `(block, head)` the Cutout reads — paper-reported for FLUX.1-dev; re-pick via the smoke notebook's head dump if your build's indexing differs |
| `guidance_scale` | 3.5 | FLUX guidance |
| `height` / `width` | 1024 | canvas size (multiple of 16) |

**Tuning:** if objects fuse with the background or each other, raise `region_bind_steps` (more
binding, sharper separation) or lower `cutout_eta` (a tighter foreground). If the composite looks
pasted-on, that is the blend — raise `num_inference_steps` so Phase C has more steps to reconcile
it. `cutout_head` is the one knob that is genuinely per-build: the paper reports block 14 / head 20
on FLUX.1-dev, and the smoke notebook dumps the text→image attention of the neighbouring heads at an
early step so you can pick the head whose high-attention tokens best isolate a single object.

**Fallback:** if the Cutout is unstable on your build, note that Region Binding alone already gives
strong position accuracy (the paper's ablation reports ~81% of the full method's spatial benefit) at
the cost of blend quality. Set `cutout_eta=1.0` to keep the full-mass prefix and you get
approximately that behaviour with the composite still in place.

## Dependencies

`diffusers` (main / ≥ 0.40), `transformers`, `accelerate`, `sentencepiece`, `protobuf`. No extra
weights, no detector, no LLM — **prompt → box decomposition is out of scope by design**: the caller
supplies `regions`, like a layout-to-image API.

## Attribution & AI assistance

**Stitch** by Jessica Bader, Mateusz Pach, Maria A. Bravo, Serge Belongie and Zeynep Akata.
**Clean-room implementation:** the reference repo
[ExplainableML/Stitch](https://github.com/ExplainableML/Stitch) carries **no license**, so none of
its code was read or copied — this block was written from the paper's description alone (the same
discipline as our [panorama](https://huggingface.co/remyxai/panorama-flux-modular) and
[regional-prompting](https://huggingface.co/remyxai/regional-prompting-flux-modular) ports; the
method itself is not copyrightable).

Authored with AI assistance (Claude) and validated by the Remyx AI team; method credit to the Stitch
authors. Uses FLUX.1-dev under its **non-commercial** license — this derivative inherits that
restriction.

## Citation

```bibtex
@misc{bader2025stitch,
  title={Stitch: Training-Free Position Control in Multimodal Diffusion Transformers},
  author={Bader, Jessica and Pach, Mateusz and Bravo, Maria A. and Belongie, Serge and Akata, Zeynep},
  year={2025}, eprint={2509.26644}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.26644}
}
```
