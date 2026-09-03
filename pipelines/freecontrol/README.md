---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - structural-control
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# FreeControl for FLUX — training-free structural control from a reference image (Modular Diffusers custom block)

**🤗 Hub:** [remyxai/freecontrol-flux-modular](https://huggingface.co/remyxai/freecontrol-flux-modular) · **📄 Paper:** [arXiv:2511.05219](https://arxiv.org/abs/2511.05219) · **📦 Monorepo:** [flux-recipes](https://github.com/remyxai/flux-recipes)

Give a **reference image** (structure) + a **target prompt** (content) and generate an image that follows the
prompt while keeping the reference's spatial layout — no fine-tuning, no new weights, **no inversion and no
gradient loop** — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks) custom block.
Clean-room implementation of **"FreeControl: Efficient, Training-Free Structural Control via One-Step Attention
Extraction"** ([arXiv:2511.05219](https://arxiv.org/abs/2511.05219)) on stock **FLUX.1-dev**. This is a new axis
for the portfolio: our [regional-prompting](https://huggingface.co/remyxai/regional-prompting-flux-modular) block
routes a prompt per region and [stitch](https://huggingface.co/remyxai/stitch-flux-modular) places a box —
FreeControl matches **an existing image's structure** with no depth/edge map and no ControlNet weights.

## Usage

```python
import torch
from diffusers import ModularPipeline
from PIL import Image

pipe = ModularPipeline.from_pretrained("remyxai/freecontrol-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")   # base FLUX.1-dev, ~36GB resident; prefer an 80GB A100

img = pipe(
    reference_image=Image.open("reference.png"),   # structure/layout to keep
    prompt="a bronze statue bust, museum, dramatic lighting",   # content
    structure_strength=0.3,   # step-cutoff dial: higher = tighter structure lock, lower = more prompt freedom
    height=1024, width=1024,
    output="images",
).images[0]
img.save("freecontrol.png")
```

Pass `reference_image=None` and the block is **bit-exact stock FLUX** (plain text-to-image, no capture, no
replacement).

## How it works

Two training-free phases on frozen FLUX.1-dev — no inversion, no optimization (~5% overhead over a stock run):

1. **One-step reference Query capture (LCD).** The reference is VAE-encoded to `x0`, then a *noise-free*
   latent is built by **Latent-Condition Decoupling**: `x̃ = (1 − σ)·x0` (default `σ=0.35`) — replacing the
   usual `x_{t*} = (1−σ)x0 + σ·ε`, which drops the stochastic-noise artifacts. A **single** transformer call at
   the key timestep **t\*=661** (empty prompt) captures the self-attention **Query** at the **last 25
   single-stream blocks**. Keys/Values are not captured — only Q.
2. **Target generation with Query replacement.** Standard FLUX denoise of the target prompt; in those same
   last-25 single blocks the **image-token Query** is replaced with the captured reference Query. Keys/Values
   and the **text** Query stay dynamic, so geometry follows the reference while content follows the prompt.

**The step cutoff is the dial the paper omits.** Injecting Q at *every* step over-locks FLUX (the prompt is
ignored — the reference subject survives unchanged). We inject only for the first `structure_strength` fraction
of steps (early steps set geometry; late steps are freed for the prompt). Exposed as `structure_strength`
(default **0.3**): higher → tighter structure lock, lower → more prompt freedom. The attention processors are
swapped for the call and restored in `finally`; with `reference_image=None` none fire.

**Validated (e2e, 28 steps):** round-trip capture→replace on the same prompt reproduces structure at
depth-corr **0.995**; on a cross-prompt transfer (astronaut → "bronze statue bust") `structure_strength=0.3`
keeps the reference layout (depth-corr ≈ 0.9) while the content becomes bronze; 0.5–0.7 revert to the
reference content, 1.0 over-locks. `σ` and the injection depth are secondary tunables.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `reference_image` | None | structure source; `None` → stock FLUX text-to-image |
| `prompt` | — | target content prompt (required) |
| `structure_strength` | 0.3 | fraction of steps that inject the reference Query; **higher = tighter structure, lower = more prompt freedom** |
| `sigma` | 0.35 | LCD scale in `x̃=(1−σ)x0` (≈0.25–0.5) |
| `key_timestep` | 661 | timestep `t*` (of 1000) for the one-step Query capture |
| `inject_last_n` | 25 | number of trailing single-stream blocks to inject |
| `num_inference_steps` | 28 | denoise steps |
| `guidance_scale` | 6.5 | generation guidance |
| `height` / `width` | 1024 | canvas size |

## Dependencies

`diffusers` (main / ≥ 0.41), `transformers`, `accelerate`, `sentencepiece`, `protobuf`. Components: base
**FLUX.1-dev** (transformer/vae/text-encoders/scheduler) — no depth model, no ControlNet, no new weights.
Built on the shared [`flux_modular`](../../flux_modular) attention primitive (vendored flat beside `block.py`
as `flux_modular.py` for `trust_remote_code`); the structural control is two small ops (`op_capture_q` /
`op_replace_q`) gated to `last_single_attn_ids`.

## Attribution & AI assistance

**Method** by Jiang Lin, Xinyu Chen, Song Wu, Zhiqiu Zhang, Jizhi Zhang, Ye Wang, Qiang Tang, Qian Wang, Jian
Yang, and Zili Yi ([arXiv:2511.05219](https://arxiv.org/abs/2511.05219)). **Clean-room implementation:** no
reference code was released, so this block was written from the paper's description alone (the method itself is
not copyrightable). Authored with AI assistance (Claude) and validated by the Remyx AI team; method credit to
the authors above. Uses **FLUX.1-dev** under its **non-commercial** license — this derivative inherits that
restriction.

## Citation

```bibtex
@misc{lin2025freecontrol,
  title={FreeControl: Efficient, Training-Free Structural Control via One-Step Attention Extraction},
  author={Lin, Jiang and Chen, Xinyu and Wu, Song and Zhang, Zhiqiu and Zhang, Jizhi and Wang, Ye and Tang, Qiang and Wang, Qian and Yang, Jian and Yi, Zili},
  year={2025}, eprint={2511.05219}, archivePrefix={arXiv}, primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.05219}
}
```
