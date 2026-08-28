# Contributing a FLUX Modular Diffusers pipeline

A contribution is **one PR** adding a self-contained `pipelines/<name>/` directory. This guide defines what that
PR must contain (implementation, testing, documentation) and how it's reviewed and published. Deep technical
detail — block layout, injection patterns, hard-won gotchas — lives in [`CONVENTIONS.md`](CONVENTIONS.md); read it first.

> **For automated contributors (Outrider):** a brief in `briefs/<name>.md` is the spec. Produce the files below and
> open the PR. **Do NOT gate finalization on GPU validation** — you cannot run FLUX in CI, and that is expected: the
> `e2e.ipynb` is run by a human reviewer, not by you. Emitting the pipeline files with a runnable `e2e.ipynb` is the
> deliverable; **do not route to an Issue for lack of GPU**. Static checks (below) are what CI enforces.

## What a pipeline PR contains

```
pipelines/<name>/
  block.py                    # the ModularPipelineBlocks custom block (flat, self-contained)
  modular_config.json         # {"_class_name","auto_map":{"ModularPipelineBlocks":"block.<Class>"}}
  modular_model_index.json    # component specs (FLUX.1-dev / Fill-dev), _blocks_class_name
  README.md                   # the model card (published verbatim to the Hub repo)
  e2e.ipynb                   # end-to-end: publish PRIVATE -> load -> run -> validate (reviewer runs this)
  assets/                     # (optional) hero / demo images referenced by README.md
```

Copy [`template/`](template) to start.

## 1 · Implementation standards (`block.py`)

Follow [`CONVENTIONS.md`](CONVENTIONS.md). In short:
- **Flat, single file.** `trust_remote_code` loads only flat sibling `.py`; vendor helpers flat or fetch a package
  at runtime (`snapshot_download` + `sys.path`, the PuLID pattern). No subdirectory packages loaded via import.
- **`ModularPipelineBlocks`** with `expected_components` (the 7 FLUX comps), `inputs` (defaults = the reference's),
  `intermediate_outputs = [OutputParam("images")]`, and `__call__(components, state)` using
  `get_block_state`/`set_block_state`.
- **Injection = one of the proven seams** (see CONVENTIONS): forward-hook on blocks · pos_embed swap ·
  attention-processor via `joint_attention_kwargs` · compose a diffusers pipeline (FLUX-Fill) · custom denoise loop.
- **Bit-exact no-op when the feature is off**; restore any transformer mutation (hooks/processors/pos_embed) in `finally`.
- **Lazy-load** heavy deps/weights once; **training-free or permissive open-weight** (MIT/Apache) — else clean-room
  from the paper. State the FLUX **non-commercial** license.

## 2 · Testing standards (`e2e.ipynb`)

The notebook a reviewer runs (Colab / A100) to verify the claim. It must:
1. install (`diffusers` main + deps) and authenticate;
2. **publish the pipeline PRIVATE** (`create_repo(private=True)`, upload `block.py` + configs);
3. **load via `ModularPipeline.from_pretrained(repo, trust_remote_code=True)`** and assert the block class;
4. run the pipeline and **show a result** + a **quantitative signal** where possible — Δ vs the reference impl,
   a metric (ArcFace / CLIP / FID / DreamSim), or a no-op control (feature-off == stock);
5. **de-risk any unknown first** with a spike (multi-file load, injection seam, input range, mask).

Keep it cheap where you can (small res / few steps for the smoke path, full settings for the headline result).

## 3 · Documentation standards (`README.md` = the model card)

- **Hero/demo image** (`assets/…`, relative path) + one-paragraph intro.
- **3-line usage**, **how it works**, **key parameters** table, **dependencies**.
- **Verified citation** — confirm authors via `curl -s arxiv.org/abs/<id>` before writing the BibTeX (never
  hallucinate). If a technique has no single paper, give accurate **lineage** + a related reference.
- **Attribution & AI assistance** — credit the method authors + the reference repo/license; disclose AI assistance
  (in the card + commit message, not reply comments). Note the **non-commercial** FLUX license.
- The Colab badge is added **after** a public demo link exists.

## PR lifecycle

1. **Brief** (`briefs/<name>.md`) specifies the candidate (vetting, mechanism, injection pattern, tests, license).
2. **Draft PR** adds `pipelines/<name>/` (Outrider or a contributor).
3. **CI (static)** checks: `block.py` compiles, configs are valid JSON with a matching `auto_map`, required files
   present, README has the sections above. *(CI does not run FLUX.)*
4. **Maintainer verifies** by running `pipelines/<name>/e2e.ipynb` — confirms it loads and produces the claimed result.
5. **Merge**, then **publish** to `remyxai/<name>-flux-modular` (private → public), **link the public Colab** in the
   card, and add it to the *Training-Free FLUX* collection.

## Author checklist
- [ ] `block.py` compiles; no-op when the feature is off; mutations restored in `finally`
- [ ] configs valid; `auto_map` → `block.<Class>`; components resolve
- [ ] `e2e.ipynb` publishes-private → loads via trust_remote_code → runs → shows a metric/result
- [ ] README has hero, usage, how-it-works, params, deps, **verified** citation, non-commercial note, AI disclosure
- [ ] training-free or permissive license (else clean-room); method authors credited

## Reviewer checklist
- [ ] static CI green (compile, configs, structure)
- [ ] ran `e2e.ipynb`: loads as the expected block, produces the claimed result, metric holds
- [ ] license/attribution correct; citation authors verified
- [ ] on merge: publish Hub repo (private→public), link Colab, add to collection
