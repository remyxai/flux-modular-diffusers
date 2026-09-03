"""Recipe loading — a recipe is a declarative ``(site, schedule, op, params)`` config a :class:`RecipeRunner`
interprets. Shipped example recipes live in the repo-root ``recipes/`` directory (one YAML per method).

A recipe dict:
    name         str
    description  str
    requires     list[str]   capabilities the adapter must provide (e.g. rope, single_stream, redux)
    inputs       list[str]   required run() inputs (prompt + ref_* image keys)
    site         dict        {stream: single|both, last_n|edge: int}
    capture      dict|None   {kind: lcd_q, sigma, timestep, source}   -- a pre-pass that banks reference Q
    condition    dict|None   {kind: redux, source, mask_floor}        -- conditioning added to the prompt
    ops          list[dict]  [{op: replace_q, tokens: image}, ...]
    params       dict        default knobs (S, redux_scale, guidance, last_n, sigma)
    validated    str         "block-parity" | "spike" | "expressible"  -- honesty flag; see recipes/README.md
"""

import os

_REQUIRED = ("name", "ops")


def _recipes_dir():
    # repo-root recipes/ (this file is flux_modular/recipes.py)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recipes")


def load_recipe(path):
    import yaml
    with open(path) as f:
        r = yaml.safe_load(f)
    for k in _REQUIRED:
        if k not in r:
            raise ValueError(f"recipe {path}: missing required key '{k}'")
    return r


def load_recipes(directory=None):
    """Return {name: recipe} for every ``*.yaml`` in ``recipes/`` (or ``directory``)."""
    directory = directory or _recipes_dir()
    out = {}
    for fn in sorted(os.listdir(directory)):
        if fn.endswith((".yaml", ".yml")):
            r = load_recipe(os.path.join(directory, fn))
            out[r["name"]] = r
    return out
