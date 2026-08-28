# Modular Diffusers Community Pipeline — Conventions (house style for Outrider briefs)

Distilled from the four shipped pipelines — **HRDiT, DyPE, PuLID, CatVTON** (`remyxai/*-flux-modular`).
Every brief in `briefs/` assumes this file. Outrider drafts `block.py` + smoke/e2e notebooks *to this spec*.

## Repo layout (a runnable custom-block Hub repo)
- **`block.py`** — the ONLY dynamically-loaded file. Must be **FLAT**: `trust_remote_code` fetches flat sibling
  `.py` referenced by relative import; it does **not** resolve subdirectory packages (proven with a spike).
  Vendor helpers as flat modules, OR fetch a vendored package at runtime
  (`snapshot_download(_REPO_ID, allow_patterns=["pkg/*"])` + `sys.path.insert`) — the eva_clip pattern (PuLID).
- **`modular_config.json`** — `{"_class_name":"<Block>","_diffusers_version":"0.41.0.dev0","auto_map":{"ModularPipelineBlocks":"block.<Block>"}}`
- **`modular_model_index.json`** — `_blocks_class_name`, `_class_name":"ModularPipeline"`, one entry per component:
  `[null,null,{pretrained_model_name_or_path, subfolder, type_hint:[lib,cls], revision:null, variant:null}]`.
  Per-component repo is allowed (CatVTON: transformer from FLUX.1-Fill-dev, the rest from FLUX.1-dev).
- **`README.md`** + **`assets/`** (LFS images, relative `assets/…` paths).

## Block skeleton (`ModularPipelineBlocks`)
```python
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
```
- `expected_components` → the 7 FLUX comps as ComponentSpec(name, cls, pretrained_model_name_or_path=<repo>, subfolder=...):
  text_encoder(CLIPTextModel)/tokenizer(CLIPTokenizer)/text_encoder_2(T5EncoderModel)/tokenizer_2(T5TokenizerFast)/
  transformer(FluxTransformer2DModel)/vae(AutoencoderKL)/scheduler(FlowMatchEulerDiscreteScheduler).
- `inputs` → `InputParam("prompt", required=True)`, plus optional knobs defaulted to the **reference's** values.
- `intermediate_outputs` → `[OutputParam("images")]`.
- `__call__(self, components, state)` (decorate `@torch.no_grad()`): `bs=self.get_block_state(state)` → work →
  `bs.images=[...]` → `self.set_block_state(state, bs)` → `return components, state`.
- `_encode_prompt` helper (copy from any shipped block): CLIP pooled + T5 embeds + `text_ids=zeros[seq,3]`.
- Lazy-load heavy weights/preprocessors **once** (`if getattr(self,"_x",None) is None`).
- Restore any transformer mutation (pos_embed swap, hooks) in a `finally`.

## Injection patterns (choose per method)
- **forward-hook on blocks** (DyPE, PuLID): both `transformer_blocks[i]` and `single_transformer_blocks[i]`
  return `(encoder_hidden_states, hidden_states)`; the image stream is `out[1]`. Inject `out[1] += w*f(...)`;
  reset a per-forward counter with `register_forward_pre_hook`. MUST be a **bit-exact no-op when off** (w=0).
- **pos_embed swap + forward_pre_hook** (DyPE): replace `transformer.pos_embed`, feed timestep/state per step.
- **attention-processor swap via `joint_attention_kwargs`** (HRDiT): custom FluxAttnProcessor reads state from a
  **named** kwarg (unnamed kwargs are dropped by transformer_flux); no module globals (concurrency-safe).
- **compose a diffusers pipeline from components + LoRA** (CatVTON): build the target pipeline
  (`FluxFillPipeline(vae=..., transformer=..., ...)`) from components, `load_lora_into_transformer`, run — when
  re-deriving the loop is risky.
- **custom denoise loop** (HRDiT base / DyPE): reimplement the FLUX loop when the method modifies velocity/latents
  per step (e.g. editing ODEs).

## Weights / deps
- pretrained weights: `hf_hub_download`; split state_dict by module prefix + `load_state_dict`.
- heavy preprocessors (face/parse/segment): lazy `import` inside the method (block imports without them installed).
- vendored packages: runtime `snapshot_download` + `sys.path` (packages don't load via trust_remote_code).

## Testing — two notebooks per pipeline
- **`smoke.ipynb`** (cheap/fast): install diffusers main + deps → auth → publish **PRIVATE** (block+configs) →
  `ModularPipeline.from_pretrained(repo, trust_remote_code=True)` → assert `type(pipe.blocks).__name__` →
  `load_components` → a small/low-res/few-step run → assert an image of the expected shape, no error.
- **`e2e.ipynb`** (full validation with a **quantitative** signal where possible): Δ vs the reference impl
  (bit-exact where achievable — HRDiT/DyPE), ArcFace cosine (PuLID), FID / before-after (CatVTON), or a no-op
  control (feature-off == stock). **De-risk any unknown FIRST with a spike** (multi-file load, injection seam,
  input range, mask coverage).
- Build notebooks with a `py_compile` check on code cells; copy to `~/Downloads` for the human to run (Colab A100).

## Publish flow (private-first, human-gated public)
1. `create_repo(private=True)` → upload block.py + configs (+ vendored files).
2. Run smoke → then e2e; fix + re-upload (still private).
3. **Human confirms results** → flip public (`update_repo_settings(private=False)`).
4. README: hero/demo asset → 3-line usage → how-it-works → key params → deps → attribution + citation
   (**verify authors** via `curl -s arxiv.org/abs/<id>`) → non-commercial note if FLUX-dev/Fill-dev.
5. Colab badge added **after** a public demo link exists → add to the umbrella collection.

## Hard-won gotchas (check every time)
- trust_remote_code loads **flat .py only** — no subdir packages.
- current `FluxFillPipeline` expects image tensors in **[0,1]** (passing [-1,1] corrupts conditioning).
- VTON agnostic mask: include **arms** (labels 14/15) so sleeves form; exclude lower body; close+dilate+fill to
  a solid region. A raw segformer-clothes union is wrong.
- issue **OPEN ≠ unaddressed** and **CLOSED ≠ declined** — READ THE COMMENTS ("already done" / "added in #X").
  (Detail-Daemon was already in Modular Diffusers; OminiControl closed "seems already done".)
- **verify paper authors** via arxiv before writing citations.
- `ModularPipeline` has NO cpu-offload method (use ComponentsManager); NF4 low-VRAM via
  `update_components(transformer=..., text_encoder_2=...)` + bnb BitsAndBytesConfig.
- FLUX.1-dev / Fill-dev = **non-commercial** license → derivatives inherit it; state it.
- AI-assist disclosure in README + commit message (not reply comments).
- HF token lacks Manage-Collections scope → collection create/add is manual UI (or grant the scope).

## Pre-flight vetting checklist (a brief ships only if all pass)
1. base model is FLUX / MMDiT / DiT (image, single-A100-validatable)?
2. training-free OR open-weight with a permissive license (MIT/Apache); else clean-room from the paper?
3. NOT saturated in core (guidance + caching are) AND not already added — check community dir + code + **issue/PR comments**?
4. modular-fit: maps to one of the injection patterns above?
5. real gap + demand (an issue, or a clearly under-served workflow)?
