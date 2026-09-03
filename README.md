# flux-recipes

A toolkit for **training-free FLUX interventions**: one attention primitive, a library of methods expressed as
**recipes**, a single interpreter that **runs / sweeps / generates** them, and the turnkey
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks) pipelines they
produce. Every published pipeline loads the same way — swap the repo id, change the capability:

```python
import torch
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("remyxai/<pipeline>-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16); pipe.to("cuda")
```

…and the *same* method is one config you can run and sweep — including compositions the source papers never tried:

```python
from flux_modular import RecipeRunner
runner = RecipeRunner()
img = runner.run("structure_appearance",                  # a method by name; freecontrol ⊕ appearance
                 {"prompt": "a portrait", "ref_structure": pose, "ref_appearance": material}, S=0.5)
```

## Architecture

Most of the shipped pipelines are the *same move* — install an attention intervention, capture/replace/share/bias
some `q/k/v` at some blocks for some steps. This repo factors that move into layers so a method is **data**, not a
hand-written denoise loop:

1. **[`flux_modular/`](flux_modular/) — the primitive.** A FLUX joint-attention intervention (`FluxIntervention`,
   subclassing diffusers' own `FluxAttnProcessor`) + an op menu (capture/replace Q, share K/V, bias, substitute,
   blend, read weights) + FLUX plumbing.
2. **[`recipes/`](recipes/) — methods as configs.** One YAML `(site, schedule, op, params)` row per method, each
   with an honest `validated:` flag. New capabilities and compositions are new rows.
3. **[`flux_modular/interpret.py`](flux_modular/interpret.py) — one interpreter.** `run_recipe(adapter, recipe,
   inputs)` drives every recipe over four run-loops (default / regional / batch / edit). It's adapter-driven, so
   the *same* code serves exploration and shipping.
4. **Run · sweep · generate.** [`RecipeRunner`](flux_modular/runner.py) wraps a `FluxPipeline` for exploration
   (**start here:** [`notebooks/papers_as_recipes.ipynb`](notebooks/papers_as_recipes.ipynb) runs published methods
   as configs + shows the codegen CLI; [`notebooks/explore.ipynb`](notebooks/explore.ipynb): sweep + compare);
   [`flux_modular/codegen.py`](flux_modular/codegen.py) (`scripts/gen_pipeline.py`) turns a validated recipe into
   a standalone `pipelines/<name>/` — `block.py` + configs + the vendored flat primitive — with **no
   re-implemented denoise loop**.
5. **[`pipelines/`](pipelines/) — the turnkey outputs.** Each is the source for a HF Hub repo
   (`remyxai/<name>-flux-modular`): `block.py`, configs, model card, and an e2e notebook a reviewer runs.

The op menu is model-agnostic; the **adapter** (token layout, RoPE, plumbing) is per-checkpoint, and a recipe
declares `requires:` so incompatible models reject cleanly. Cross-model transfer is **verified per method**, not
assumed (an SD3.5 probe found the adapter ports but Q-replace is rope-specific) — so recipes are FLUX-family for
now. See [`recipes/README.md`](recipes/README.md).

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
