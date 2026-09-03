# Recipes — training-free FLUX interventions as configs

Each `*.yaml` here is a **recipe**: a declarative `(site, schedule, op, params)` config that
[`FluxLens`](../flux_modular/lens.py) interprets into `flux_modular` payloads. The idea: most of our shipped
`pipelines/` are the *same move* — install `FluxIntervention`, capture/replace/share/bias some `q/k/v` at some
blocks for some steps — so we make that move **data**. A method becomes a row; a new configuration (including
compositions the source papers never tried) is a new row; exploring the neighbourhood is a `sweep`.

```python
from flux_modular.lens import FluxLens
from flux_modular.recipes import load_recipes

lens = FluxLens()                      # loads FLUX.1-dev (Redux lazily)
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
| `condition` | optional conditioning added to the prompt: `{kind: redux, source, mask_floor}` |
| `ops` | the intervention: `[{op: replace_q, tokens: image}, ...]` |
| `params` | default knobs (`S` = structure-inject step cutoff, `redux_scale`, `guidance`, `last_n`, `sigma`) |
| `validated` | **honesty flag** — see below |
| `pipeline` / `paper` | links to the shipped turnkey block + source method |

### `validated` flag — don't overclaim

- **`block-parity`** — the recipe reproduces the hand-written `pipelines/<name>` block's metric on an A100
  (e.g. `freecontrol` config depth-corr **0.883** vs the shipped block's **0.89**).
- **`spike`** — validated by a GPU spike but not yet cross-checked against a standalone block
  (`appearance`, `structure_appearance`).
- **`expressible`** — the ops exist and the recipe *should* work, but it hasn't been GPU-validated through the
  lens yet. **Validate via `explore.ipynb` before trusting it.** (No `expressible` recipes are shipped yet — add
  the remaining attention-methods here as they're validated.)

## Scope — this is the FLUX adapter

The **op menu is model-agnostic**; the **adapter** (token layout, RoPE, plumbing, block enumeration) is
per-checkpoint. An SD3.5 spike showed the adapter *ports* (a ~40-line `SD3Intervention` captured/replaced Q on
all 19 target blocks) but **`replace_q` itself did not transfer** — it destabilises SD3.5 (Q-replace leans on
FLUX's rope-separable position). So: **method transfer is verified per `(method × model)`, never assumed.** The
position-agnostic ops (`bias`/mask, K/V-`share`) are the better cross-architecture bet and are the next thing to
test. Until then these recipes are **FLUX-family**.

## Adding a recipe

1. Copy an existing YAML, set `site` / `capture` / `condition` / `ops` / `params`, mark `validated: expressible`.
2. If it needs an op the interpreter doesn't handle yet (`bias`, `substitute`, `append`/`share`, `blend`), wire
   that op into `FluxLens.run` (they already exist in `flux_modular/attention.py`; the interpreter just needs to
   route them — keep each addition small and fire-proofed).
3. Validate in `explore.ipynb` (op-fired count == target blocks; eyeball the grid), then upgrade `validated`.
