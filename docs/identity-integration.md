# Scope — PuLID identity integration (face-lock via the residual seam)

**Goal.** An `identity` recipe (face-lock from a reference photo) driven through the `flux_residual` hook, and the
three-way composition **identity ⊕ structure ⊕ appearance** — now *declarative* because the residual hook and
multi-donor Redux are first-class. This upgrades `story_reference` from appearance-level to true identity.

**Not training-free.** Unlike every other recipe, this uses **PuLID's released weights** (`guozinan/PuLID`,
`pulid_flux_v0.9.1.safetensors`) + an ID-encoder stack (InsightFace ArcFace, facexlib align/parse, EVA-CLIP). So
it is an **open-weight integration**, marked as such in the honesty system, with FLUX-dev + PuLID licenses inherited.

## The good news — PuLID's injection already IS the seam

From `pipelines/pulid/block.py` (shipped + validated): the ID injection is exactly a `flux_residual`-shaped hook.

- **ID encoder** (`_PuLIDEncoder`): InsightFace ArcFace (`antelopev2/glintr100.onnx`) → identity embed; facexlib
  (retinaface + bisenet) aligns/parses the face; EVA-CLIP (`EVA02-CLIP-L-14-336`) → face features; `IDFormer`
  (Perceiver resampler, 5 id tokens) fuses them → `id_embedding`. `encoder.get_id_embedding(face) -> id_embedding`.
- **Injection** (`_install_pulid`): `pulid_ca` = `ModuleList[PerceiverAttentionCA]`; a forward-hook on double blocks
  (`i % 2 == 0`) and single blocks (`i % 4 == 0`) does `out[1] += id_weight * pulid_ca[k](id_embedding, out[1])`,
  where `k` is a per-forward sequential counter (reset by a pre-hook).
- **Weights**: the `pulid_ca` + `IDFormer` params load from the PuLID checkpoint; `eva_clip` is vendored (fetched
  from `remyxai/pulid-flux-modular` at runtime). Deps: `insightface`, `facexlib`, `onnxruntime`.

**So the whole ID stack exists.** The integration is: route PuLID's injection through the shared `flux_residual`
hook and expose it as a recipe. No new ID research.

## Integration architecture

1. **Reuse `_PuLIDEncoder`** (extract from `pipelines/pulid/` into a loadable module, or import). It produces
   `id_embedding` from a face image and holds the trained `pulid_ca` cross-attn layers.
2. **A residual content op** `op_identity(encoder, id_embedding, id_weight, camap)`:
   ```python
   def fn(h, bid):
       return h + id_weight * encoder.pulid_ca[camap[bid]](id_embedding, h) if bid in camap else h
   ```
   plugged straight into `flux_residual(tr, set(camap), fn)`.
3. **Replace the per-forward counter with an explicit `camap = {block_id: k}`** — precompute by iterating the
   injection blocks in order (double `i%2==0`, then single `i%4==0`). This is the one subtle correctness item:
   `pulid_ca[k]` must line up with the block it was trained for. The counter works because blocks fire in order;
   the id-keyed map must reproduce that exact order (double blocks first, then single, ascending index).
4. **`_run_identity`**: load encoder (lazy, heavy), `id_embedding = encoder.get_id_embedding(inputs["id_image"])`,
   `with flux_residual(tr, ids, op_identity(...)): _denoise(...)`. Mirrors `_run_residual` but with the trained fn.
5. **Recipe** `identity` — `run: residual` (content: pulid), `requires: [pulid]`, `inputs: [id_image, prompt]`,
   `params: {id_weight: 1.0, guidance: 4.0}`.

## The three-way (the payoff)

`identity ⊕ structure ⊕ appearance` in one denoise — three independent mechanisms that already coexist:
- **identity** — `flux_residual` with the PuLID fn (block-output residual),
- **structure** — `flux_intervention` with `replace_q` (freecontrol Q-replace, attention pre-rope), soft-scheduled,
- **appearance** — Redux tokens prepended to the encoder (multi-donor supported).

A `_run_identity_composed` installs all three around one `_denoise`. Declarative recipe:
`{capture: lcd_q (structure), condition: {kind: redux, sources: [...]} (appearance), residual: pulid (identity)}`.
Inputs: `{id_image, ref_structure, ref_appearance, prompt}`.

## Risks / open questions (spike before trusting)

1. **camap ordering** (correctness): wrong block↔`pulid_ca[k]` mapping silently breaks identity. Phase-0 spike must
   confirm the id-keyed map reproduces the shipped `pulid` block's output.
2. **Three-way interference**: the residual identity is applied on the same late blocks freecontrol Q-replaces.
   Identity injection is targeted (id-only, scaled ~1.0) so gentler than the feature-echo, but the face region may
   fight the structure lock. Needs the structure_appearance-style interference spike + the soft schedule to
   arbitrate (release structure late so identity can set the face).
3. **Heavy env**: `insightface` (onnx), `facexlib`, `onnxruntime`, `antelopev2` face models, PuLID checkpoint,
   vendored `eva_clip`. Colab setup friction; the encoder load is slow. Not a light recipe.
4. **Codegen breaks the thin-block model**: an `identity` block can't be a ~30-line `ComponentsAdapter` block — it
   needs the PuLID encoder code + eva_clip bootstrap + face deps (like `pipelines/pulid/` does today). So identity
   likely stays a **richer generated block** (vendor the PuLID stack), or codegen grows a "heavy recipe" path. Flag,
   don't pretend it's thin.
5. **Metric**: identity needs a real face-similarity metric — **ArcFace cosine** between the reference face and the
   generated face (InsightFace is already loaded), not depth/CLIP. Eyeball as always.

## Phases

- **Phase 0 — seam spike** (de-risk mapping): reuse `_PuLIDEncoder`, wire its injection through `flux_residual`
  with the id-keyed `camap`, and confirm on A100 it reproduces the shipped `pulid` block's identity (ArcFace-sim
  parity). This proves the counter→map translation and that identity survives the shared seam.
- **Phase 1 — `identity` recipe**: `_run_identity` + component/dep loading + recipe. Ship `expressible` (open-weight
  flag). Validate ArcFace-sim vs the shipped block.
- **Phase 2 — three-way**: `_run_identity_composed` + `identity_structure_appearance` recipe + interference spike
  (id_weight × structure S × redux_scale sweep, ArcFace-sim + depth-corr + appearance-CLIP + eyeball). The marquee.
- **Phase 3 — codegen**: decide thin-vs-heavy block for identity; likely vendor the PuLID stack into the generated
  block (a "heavy recipe" codegen path).

## Effort

Phase 0/1 are mostly plumbing existing code (the ID stack + weights exist; the seam exists) — the real work is the
`camap` correctness spike + the heavy-dep loader. Phase 2's value (the declarative three-way) is high; its risk is
the interference, mitigated by the soft schedule we just shipped. Recommend **Phase 0 first** — one spike settles
whether the shared seam preserves PuLID identity before building the recipe layer on top.

## The garment axis — CatVTON, NOT Redux (validated 2026-09-03, `notebooks/dress_the_story.ipynb`)

To put a *specific worn garment* on the story character, the right tool is a **try-on specialist, not the generator's
appearance channel**. Redux transfers the reference's GLOBAL content, so a flat product-shot garment gets **cloned
wholesale** — a floating jacket, `no-face` (measured: `redux_scale=0.9` → look-CLIP 0.97, ArcFace no-face). Redux
dresses nobody.

**CatVTON does.** Two-stage, two specialists (chained shipped pipelines, not one denoise):
1. **`identity_story`** (FLUX.1-dev) → face-locked character across panels *(undressed)*.
2. Free the generator, load **`remyxai/catvton-flux-modular`** (FLUX.1-Fill + catvton LoRA + segformer auto-masker);
   for each panel `vton(person_image=panel, garment_image=garment)`. CatVTON inpaints **only** the clothing region
   (auto agnostic mask covers upper garment + arms), so the **face is untouched** → identity survives the edit and
   the garment is genuinely worn.

Measured (auburn-haired character + leather aviator jacket): ArcFace through the edit **0.74–0.75** (vs 0.77–0.80
before — face essentially unchanged), jacket-CLIP rises before→after; the fur collar transfers cleanly on clear
half-body framing (clifftop panel = showcase). **Limitation** — the segformer auto-mask needs a clear standing/half-
body torso: a large foreground occluder (a ship's wheel) or an unusual base garment (strappy overalls) yields a
partial mask and a weak/recolored result (sailboat panel: jacket-CLIP flat 0.45→0.46). Fix = framing or a manual
`mask`. This is the [[grounded_operands]] principle at the composition layer: a worn garment is a grounded operand
the try-on model produces, not something the appearance channel can fabricate. Replaces the Redux "look" attempt in
the story-compositions notebook.
