# Recipes — training-free FLUX interventions as configs

Each `*.yaml` here is a **recipe**: a declarative `(site, schedule, op, params)` config that
[`RecipeRunner`](../flux_modular/runner.py) interprets into `flux_modular` payloads. The idea: most of our shipped
`pipelines/` are the *same move* — install `FluxIntervention`, capture/replace/share/bias some `q/k/v` at some
blocks for some steps — so we make that move **data**. A method becomes a row; a new configuration (including
compositions the source papers never tried) is a new row; exploring the neighbourhood is a `sweep`.

```python
from flux_modular.runner import RecipeRunner
from flux_modular.recipes import load_recipes

lens = RecipeRunner()                      # loads FLUX.1-dev (Redux lazily)
recipes = load_recipes()               # {name: recipe} from this directory
img = lens.run(recipes["freecontrol"], {"prompt": "a bronze statue bust",
                                         "ref_structure": Image.open("pose.png")}, S=0.3)
```

See [`notebooks/explore.ipynb`](../notebooks/explore.ipynb) to run any recipe and sweep new configurations with a
side-by-side comparison grid.

## Schema

| key | meaning |
|---|---|
| `name` / `description` | identity |
| `requires` | adapter capabilities the recipe needs (`rope`, `single_stream`, `redux`) — incompatible models reject, not run-wrong |
| `inputs` | required `run()` inputs (`prompt` + any `ref_*` image keys) |
| `site` | where: `{stream: single, last_n: N}` or `{stream: both, edge: n}` |
| `capture` | optional pre-pass banking reference Q: `{kind: lcd_q, sigma, timestep, source}` |
| `condition` | optional conditioning: `{kind: redux, source, mask?, mask_floor?}` (single, optional mask-weighting = material-not-shape) or `{kind: redux, sources: [{image, scale, mask?}, ...]}` (multi-donor, per-donor `redux_scale_<image>`) |
| `ops` | the intervention: `[{op: replace_q, tokens: image}, ...]` |
| `params` | default knobs (`S` = structure-inject fraction, `schedule` = `cutoff`/`linear`/`cosine` for hard vs smooth structure release, `redux_scale`, `guidance`, `last_n`, `sigma`) |
| `validated` | **honesty flag** — see below |
| `pipeline` / `paper` | links to the shipped turnkey block + source method |

### `validated` flag — don't overclaim

- **`block-parity`** — the recipe reproduces the hand-written `pipelines/<name>` block's metric on an A100
  (e.g. `freecontrol` config depth-corr **0.883** vs the shipped block's **0.89**).
- **`spike`** — validated by a GPU spike (fire-proofed vs a baseline + a metric + eyeball) but not cross-checked
  against a standalone block (`appearance`, `structure_appearance`, `regional`, `story`, `kv-edit`).
- **`expressible`** — the ops exist and the recipe *should* work, but it hasn't passed the lens on an A100 yet.
  **Validate via `validate_recipes.ipynb` before trusting it.** (All 7 shipped recipes are now A100-validated.)

## Scope — this is the FLUX adapter

The **op menu is model-agnostic**; the **adapter** (token layout, RoPE, plumbing, block enumeration) is
per-checkpoint. An SD3.5 spike showed the adapter *ports* (a ~40-line `SD3Intervention` captured/replaced Q on
all 19 target blocks) but **`replace_q` itself did not transfer** — it destabilises SD3.5 (Q-replace leans on
FLUX's rope-separable position). So: **method transfer is verified per `(method × model)`, never assumed.** The
position-agnostic ops (`bias`/mask, K/V-`share`) are the better cross-architecture bet and are the next thing to
test. Until then these recipes are **FLUX-family**.

## Method coverage

The interpreter has eight run paths — **default**, **regional**, **composed** (structure+regional), **residual** (block-output feature hook), **identity** (PuLID face-lock, open-weight), **identity_composed** (identity+structure+appearance three-way), **batch** (frame-shared K/V), and **edit** (RF-inversion → substitute). A method that needs a *fifth*
loop isn't fake-shipped as a recipe; it's mapped here with the extension it needs, so "add a recipe" stays honest
(a recipe should run through the interpreter, not lie about it).

| method | run path · op(s) | shipped? | notes |
|---|---|---|---|
| freecontrol | default · `replace_q` | ✅ recipe (block-parity) | — |
| appearance | default · Redux | ✅ recipe (spike) | — |
| appearance_mix | default · multi-Redux | ⚠️ recipe (expressible) | NEW multi-donor — mix material/style from 2+ refs, per-donor sweepable scales |
| structure_appearance | default · `replace_q` + Redux | ✅ recipe (spike) | — |
| structure_regional | **composed** · `replace_q` + `bias` | ⚠️ recipe (expressible) | NEW composition — reference layout + per-region prompts in one pass; validate before trusting |
| reference_echo | **residual** · `inject_feat` | ✅ recipe (spike) | NEW seam — block-output feature hook (forward-hooks, not attention). Strong reference reconstruction; the seam identity/PuLID plugs into |
| regional | regional · `bias` | ✅ recipe (spike) | — |
| story-diffusion | **batch** · `share` | ✅ recipe (spike) | returns a list of frames |
| story_reference | **batch** · `share` + Redux | ⚠️ recipe (expressible) | NEW — a character from a REFERENCE IMAGE across scenes; Redux appearance-level (look), not face-lock |
| kv-edit | **edit** · `substitute` | ✅ recipe (spike) | RF-invert → substitute bg K/V via the shared pre-rope hook |
| consistedit | **edit** · `blend` | ✅ recipe (spike) | **α (`consistency_strength`) dial: α≈0 = shape-change edit (validated); higher α progressively SUPPRESSES the edit** (output → source), it does **not** restyle. Default 0.0; use low α. |
| stitch | regional · `bias` + cutout | ⏳ | a cutout/composite **post-op** after the region-bind bias (needs a non-attention step) |
| flowedit | — (custom ODE) | ✗ | inversion-free velocity trick, **not** an attention op — stays a standalone pipeline |

7 recipes across 4 run paths — all A100-validated (`spike`/`block-parity`). The two unshipped are honestly *not* pure-attention recipes: `stitch` needs a
cutout/composite **post-op** on top of the region-bind bias, and `flowedit` is a custom velocity ODE, not an
attention payload at all.

The non-attention pipelines (`hrdit`, `dype`, `pulid`, `catvton`, `panorama`) use pos-embed swaps / LoRA compose /
custom loops, not the attention payload — they aren't recipe-shaped and remain standalone.

## Adding a recipe

1. Copy an existing YAML, set `site` / `capture` / `condition` / `ops` / `params`, mark `validated: expressible`.
2. If it needs an op the interpreter doesn't handle yet (`bias`, `substitute`, `append`/`share`, `blend`), wire
   that op into `RecipeRunner.run` (they already exist in `flux_modular/attention.py`; the interpreter just needs to
   route them — keep each addition small and fire-proofed).
3. Validate in `explore.ipynb` (op-fired count == target blocks; eyeball the grid), then upgrade `validated`.
