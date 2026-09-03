"""Codegen — turn a validated recipe into a standalone Modular-Diffusers pipeline dir.

    from flux_modular.recipes import load_recipes
    from flux_modular.codegen import generate_pipeline
    generate_pipeline(load_recipes()["freecontrol"], "pipelines/freecontrol-gen")

Emits ``block.py`` (thin — builds a :class:`ComponentsAdapter` and calls ``run_recipe`` on the baked-in recipe),
``modular_config.json``, ``modular_model_index.json``, the flat ``flux_modular.py`` (via :mod:`bundle`), and a
README stub. So a method that's a validated row becomes a runnable ``trust_remote_code`` pipeline with no
hand-written denoise loop — the interpreter is shared, not copied.
"""

import json
import os

from .bundle import build as _bundle

_FLUX = "black-forest-labs/FLUX.1-dev"
_REDUX = "black-forest-labs/FLUX.1-Redux-dev"

# components a redux-conditioned recipe additionally needs
_REDUX_COMPONENTS = [
    ("image_encoder", "transformers", "SiglipVisionModel", _REDUX, "image_encoder"),
    ("feature_extractor", "transformers", "SiglipImageProcessor", _REDUX, "feature_extractor"),
    ("image_embedder", "diffusers", "ReduxImageEncoder", _REDUX, "image_embedder"),
]
_FLUX_COMPONENTS = [
    ("text_encoder", "transformers", "CLIPTextModel", "text_encoder"),
    ("tokenizer", "transformers", "CLIPTokenizer", "tokenizer"),
    ("text_encoder_2", "transformers", "T5EncoderModel", "text_encoder_2"),
    ("tokenizer_2", "transformers", "T5TokenizerFast", "tokenizer_2"),
    ("transformer", "diffusers", "FluxTransformer2DModel", "transformer"),
    ("vae", "diffusers", "AutoencoderKL", "vae"),
    ("scheduler", "diffusers", "FlowMatchEulerDiscreteScheduler", "scheduler"),
]


def _cls_name(recipe):
    return "".join(p.capitalize() for p in recipe["name"].replace("-", "_").split("_")) + "Block"


def _needs_redux(recipe):
    return "redux" in recipe.get("requires", [])


def _block_py(recipe):
    cls = _cls_name(recipe)
    needs_redux = _needs_redux(recipe)
    inputs = list(recipe.get("inputs", ["prompt"]))
    params = recipe.get("params", {})

    comp_lines = []
    for name, lib, klass, sub in _FLUX_COMPONENTS:
        comp_lines.append(f'            ComponentSpec("{name}", {klass}, pretrained_model_name_or_path=_FLUX, subfolder="{sub}"),')
    if needs_redux:
        for name, lib, klass, repo, sub in _REDUX_COMPONENTS:
            comp_lines.append(f'            ComponentSpec("{name}", {klass}, pretrained_model_name_or_path=_REDUX, subfolder="{sub}"),')

    input_lines = [f'            InputParam("{k}", required=True),' for k in inputs]
    for k, v in params.items():
        input_lines.append(f'            InputParam("{k}", default={v!r}),')
    for k, v in (("num_inference_steps", 28), ("height", 1024), ("width", 1024), ("seed", 0)):
        input_lines.append(f'            InputParam("{k}", default={v!r}),')

    imports_redux = ""
    if needs_redux:
        imports_redux = ("\nfrom transformers import SiglipVisionModel, SiglipImageProcessor"
                         "\nfrom diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder")

    recipe_json = json.dumps(recipe, indent=0).replace("\\", "\\\\")
    return f'''\
"""{cls} — GENERATED from the "{recipe["name"]}" recipe by flux_modular.codegen. Do not hand-edit.

{recipe.get("description", "")}
Paper: {recipe.get("paper", "-")}  ·  Source pipeline: {recipe.get("pipeline", "-")}
Training-free on FLUX.1-dev (non-commercial). The denoise logic lives in the shared, vendored
flux_modular interpreter (run_recipe); this block only wires Modular-Diffusers components to it.
"""

import json
import torch

from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast{imports_redux}

from .flux_modular import run_recipe, ComponentsAdapter

_FLUX = "{_FLUX}"
_REDUX = "{_REDUX}"
RECIPE = json.loads(r"""{recipe_json}""")


class {cls}(ModularPipelineBlocks):
    model_name = "{recipe['name']}"

    @property
    def description(self):
        return RECIPE.get("description", "")

    @property
    def expected_components(self):
        return [
{chr(10).join(comp_lines)}
        ]

    @property
    def inputs(self):
        return [
{chr(10).join(input_lines)}
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        a = ComponentsAdapter(components, steps=int(bs.num_inference_steps), height=int(bs.height), width=int(bs.width))
        inp = {{k: getattr(bs, k) for k in RECIPE.get("inputs", [])}}
        params = {{k: getattr(bs, k) for k in RECIPE.get("params", {{}})}}
        out = run_recipe(a, RECIPE, inp, seed=int(bs.seed), **params)
        bs.images = out if isinstance(out, list) else [out]
        self.set_block_state(state, bs)
        return components, state
'''


def _model_index(recipe):
    cls = _cls_name(recipe)
    idx = {"_class_name": "ModularPipeline", "_diffusers_version": "0.41.0.dev0", "_blocks_class_name": cls}
    for name, lib, klass, sub in _FLUX_COMPONENTS:
        idx[name] = [None, None, {"pretrained_model_name_or_path": _FLUX, "subfolder": sub,
                                  "type_hint": [lib, klass], "revision": None, "variant": None}]
    if _needs_redux(recipe):
        for name, lib, klass, repo, sub in _REDUX_COMPONENTS:
            idx[name] = [None, None, {"pretrained_model_name_or_path": repo, "subfolder": sub,
                                      "type_hint": [lib, klass], "revision": None, "variant": None}]
    return idx


def _readme(recipe):
    cls = _cls_name(recipe)
    ins = ", ".join(f"`{k}`" for k in recipe.get("inputs", []))
    return f'''\
# {recipe["name"]} (generated) — training-free FLUX pipeline

**Generated** from the `{recipe["name"]}` recipe by `flux_modular.codegen` — the denoise logic is the shared
vendored `flux_modular` interpreter, not hand-written. Source method: [{recipe.get("paper","-")}]({recipe.get("paper","")}).

## Usage
```python
import torch
from diffusers import ModularPipeline
pipe = ModularPipeline.from_pretrained("<repo>", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16); pipe.to("cuda")
img = pipe({ins.replace("`","")}=..., output="images").images[0]
```

## How it works
`{cls}` builds a `ComponentsAdapter` over the Modular-Diffusers components and calls `run_recipe(RECIPE, ...)`.
The recipe (`(site, schedule, op, params)`) is baked into `block.py`; edit `RECIPE` there or regenerate.

## Attribution
Method credit to the source authors ({recipe.get("paper","-")}). Generated + validated by Remyx AI with AI
assistance (Claude). Uses FLUX.1-dev (non-commercial); this derivative inherits that license.

## References
{recipe.get("paper","-")}
'''


def generate_pipeline(recipe, out_dir):
    """Write block.py + configs + flat flux_modular.py + README into ``out_dir``. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    cls = _cls_name(recipe)
    open(os.path.join(out_dir, "block.py"), "w").write(_block_py(recipe))
    json.dump({"_class_name": cls, "_diffusers_version": "0.41.0.dev0",
               "auto_map": {"ModularPipelineBlocks": f"block.{cls}"}},
              open(os.path.join(out_dir, "modular_config.json"), "w"), indent=2)
    json.dump(_model_index(recipe), open(os.path.join(out_dir, "modular_model_index.json"), "w"), indent=2)
    open(os.path.join(out_dir, "flux_modular.py"), "w").write(_bundle())
    open(os.path.join(out_dir, "README.md"), "w").write(_readme(recipe))
    return out_dir
