# FLUX Modular Diffusers

Training-free and open-weight capabilities for off-the-shelf **FLUX**, each packaged as a one-line
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks) community
pipeline. Every pipeline loads the same way — swap the repo id, change the capability:

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/<pipeline>-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16); pipe.to("cuda")
```

Each `pipelines/<name>/` here is the **source** for a HF Hub repo (`remyxai/<name>-flux-modular`): the
`block.py`, its configs, the model card, and an end-to-end notebook a reviewer runs to verify before merge.

## Catalog

| pipeline | axis | what it does | Hub repo |
|---|---|---|---|
| [hrdit](pipelines/hrdit) | high-res | training-free 4K (resolution ladder + NTK RoPE + SPA + structure) | [🤗](https://huggingface.co/remyxai/hrdit-flux-modular) |
| [dype](pipelines/dype) | high-res | single-pass ultra-high-res (dynamic RoPE) + SEGA speckle fix | [🤗](https://huggingface.co/remyxai/dype-flux-modular) |
| [pulid](pipelines/pulid) | identity | face personalization from one photo | [🤗](https://huggingface.co/remyxai/pulid-flux-modular) |
| [catvton](pipelines/catvton) | try-on | a garment onto a person | [🤗](https://huggingface.co/remyxai/catvton-flux-modular) |
| [flowedit](pipelines/flowedit) | editing | inversion-free, structure-preserving text edit | [🤗](https://huggingface.co/remyxai/flowedit-flux-modular) |
| [regional-prompting](pipelines/regional-prompting) | control | a different prompt per region | [🤗](https://huggingface.co/remyxai/regional-prompting-flux-modular) |
| [style-aligned](pipelines/style-aligned) | style | one style across a set | [🤗](https://huggingface.co/remyxai/style-aligned-flux-modular) |
| [story-diffusion](pipelines/story-diffusion) | consistency | one character across a comic | [🤗](https://huggingface.co/remyxai/story-diffusion-flux-modular) |
| [tiled-upscaler](pipelines/tiled-upscaler) | upscale | creative ×2/×4 upscaling — tiled FLUX img2img adds detail beyond a resize | [🤗](https://huggingface.co/remyxai/tiled-upscaler-flux-modular) |

## How this repo works

- **`pipelines/<name>/`** — one self-contained pipeline: `block.py` + `modular_config.json` +
  `modular_model_index.json` + `README.md` (the model card) + `e2e.ipynb` (verify-before-merge).
- **`briefs/`** — specs for pipelines not yet built. Each brief is the input to draft a new `pipelines/<name>/`.
- **`template/`** — a skeleton pipeline dir to copy for new contributions.
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — how to implement, test, and document a pipeline, and the PR → review →
  publish flow.
- **[`CONVENTIONS.md`](CONVENTIONS.md)** — the technical house style (block layout, injection patterns, gotchas).

## Contribution flow (brief → PR → verify → publish)

1. A **brief** in `briefs/` specifies a candidate (mechanism, injection pattern, tests, license).
2. **Outrider** (or a contributor) drafts a PR adding `pipelines/<name>/` per `CONTRIBUTING.md`.
3. A **maintainer runs `pipelines/<name>/e2e.ipynb`** (Colab/A100) to verify it loads + produces the claimed result.
4. On pass: **merge**, publish the pipeline to `remyxai/<name>-flux-modular` (private → public), and link the
   public Colab in the card.

All pipelines use **FLUX.1-dev / FLUX.1-Fill-dev** under their **non-commercial** license; contributions credit
the original method authors.
