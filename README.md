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

## Two layers: a primitive + recipes, and the pipelines they generate

Under the pipelines is one shared primitive, [`flux_modular`](flux_modular/) — a FLUX joint-attention
intervention (`FluxIntervention`, subclassing diffusers' own `FluxAttnProcessor`) plus a small op menu
(capture/replace Q, share K/V, bias, substitute, blend, read weights) and the FLUX plumbing. Most pipelines here
are the *same move* expressed differently, so we also expose them as **[recipes](recipes/)** — declarative
`(site, schedule, op, params)` configs that [`FluxLens`](flux_modular/lens.py) runs:

```python
from flux_modular import FluxLens, load_recipes
lens, recipes = FluxLens(), load_recipes()
img = lens.run(recipes["freecontrol"], {"prompt": "a bronze statue bust", "ref_structure": ref}, S=0.3)
```

- **[`recipes/`](recipes/)** — one config per method (e.g. `freecontrol` reproduces its shipped block at
  depth-corr 0.883 ≈ 0.89). New capabilities — including **compositions the source papers never tried**
  (`structure_appearance` = `freecontrol` ⊕ `appearance`) — are new rows, not new code.
- **[`notebooks/explore.ipynb`](notebooks/explore.ipynb)** — run any recipe and **sweep new configurations** with
  a side-by-side comparison grid + a metric panel (rank only; the GO call is by eye).
- **[`pipelines/`](pipelines/)** — the turnkey HF outputs. A validated recipe can be realised as one.

The op menu is model-agnostic; the adapter (token layout, RoPE, plumbing) is per-checkpoint. A recipe declares
`requires:` so incompatible models reject cleanly. Cross-model transfer is **verified per method**, not assumed
(an SD3.5 probe found the adapter ports but Q-replace is rope-specific) — so recipes are FLUX-family for now. See
[`recipes/README.md`](recipes/README.md).

## Catalog

| pipeline | axis | what it does | 🤗 Hub | 📄 paper | source |
|---|---|---|---|---|---|
| [hrdit](pipelines/hrdit) | high-res | training-free 4K (resolution ladder + NTK RoPE + SPA + structure) | [hrdit-flux-modular](https://huggingface.co/remyxai/hrdit-flux-modular) | [arXiv:2608.07003](https://arxiv.org/abs/2608.07003) | [zylwithxy/HRDiT](https://github.com/zylwithxy/HRDiT) |
| [dype](pipelines/dype) | high-res | single-pass ultra-high-res (dynamic RoPE) + SEGA speckle fix | [dype-flux-modular](https://huggingface.co/remyxai/dype-flux-modular) | [arXiv:2510.20766](https://arxiv.org/abs/2510.20766) | [guyyariv/DyPE](https://github.com/guyyariv/DyPE) |
| [pulid](pipelines/pulid) | identity | face personalization from one photo | [pulid-flux-modular](https://huggingface.co/remyxai/pulid-flux-modular) | [arXiv:2404.16022](https://arxiv.org/abs/2404.16022) | [ToTheBeginning/PuLID](https://github.com/ToTheBeginning/PuLID) |
| [catvton](pipelines/catvton) | try-on | a garment onto a person | [catvton-flux-modular](https://huggingface.co/remyxai/catvton-flux-modular) | [arXiv:2407.15886](https://arxiv.org/abs/2407.15886) | [nftblackmagic/catvton-flux](https://github.com/nftblackmagic/catvton-flux) |
| [flowedit](pipelines/flowedit) | editing | inversion-free, structure-preserving text edit | [flowedit-flux-modular](https://huggingface.co/remyxai/flowedit-flux-modular) | [arXiv:2412.08629](https://arxiv.org/abs/2412.08629) | [fallenshock/FlowEdit](https://github.com/fallenshock/FlowEdit) |
| [regional-prompting](pipelines/regional-prompting) | control | a different prompt per region | [regional-prompting-flux-modular](https://huggingface.co/remyxai/regional-prompting-flux-modular) | [arXiv:2302.08113](https://arxiv.org/abs/2302.08113) | [hako-mikan/…regional-prompter](https://github.com/hako-mikan/sd-webui-regional-prompter) |
| [style-aligned](pipelines/style-aligned) | style | one style across a set | [style-aligned-flux-modular](https://huggingface.co/remyxai/style-aligned-flux-modular) | [arXiv:2312.02133](https://arxiv.org/abs/2312.02133) | [google/style-aligned](https://github.com/google/style-aligned) |
| [story-diffusion](pipelines/story-diffusion) | consistency | one character across a comic | [story-diffusion-flux-modular](https://huggingface.co/remyxai/story-diffusion-flux-modular) | [arXiv:2405.01434](https://arxiv.org/abs/2405.01434) | [HVision-NKU/StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) |
| [kv-edit](pipelines/kv-edit) | editing | masked text edit, background pixel-precise (cached K/V) | [kv-edit-flux-modular](https://huggingface.co/remyxai/kv-edit-flux-modular) | [arXiv:2502.17363](https://arxiv.org/abs/2502.17363) | [Xilluill/KV-Edit](https://github.com/Xilluill/KV-Edit) |
| [consistedit](pipelines/consistedit) | editing | FLUX-native text edit, adjustable structural consistency (vision-token fusion on the single blocks) | [consistedit-flux-modular](https://huggingface.co/remyxai/consistedit-flux-modular) | [arXiv:2510.17803](https://arxiv.org/abs/2510.17803) | [zxYin/ConsistEdit_Code](https://github.com/zxYin/ConsistEdit_Code) |
| [panorama](pipelines/panorama) | processing | ultra-wide / panoramic generation (fused windows, clean-room) | [panorama-flux-modular](https://huggingface.co/remyxai/panorama-flux-modular) | [arXiv:2302.08113](https://arxiv.org/abs/2302.08113) | [omerbt/MultiDiffusion](https://github.com/omerbt/MultiDiffusion) |
| [stitch](pipelines/stitch) | control | hard bounding-box object placement (Region Binding + cutout/composite, clean-room) | [stitch-flux-modular](https://huggingface.co/remyxai/stitch-flux-modular) | [arXiv:2509.26644](https://arxiv.org/abs/2509.26644) | [ExplainableML/Stitch](https://github.com/ExplainableML/Stitch) |
| [appearance-transfer](pipelines/appearance-transfer) | appearance | reference appearance/texture onto a source, structure preserved (Depth inversion + mask-weighted Redux + KV-share, clean-room) | [appearance-transfer-flux-modular](https://huggingface.co/remyxai/appearance-transfer-flux-modular) | [arXiv:2603.26767](https://arxiv.org/abs/2603.26767) | clean-room (no code released) |
| [freecontrol](pipelines/freecontrol) | control | reference-image structural/layout control, no depth map or ControlNet (one-step Query extraction, clean-room) | [freecontrol-flux-modular](https://huggingface.co/remyxai/freecontrol-flux-modular) | [arXiv:2511.05219](https://arxiv.org/abs/2511.05219) | clean-room (no code released) |

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
