# Brief — StoryDiffusion → FLUX (training-free consistent-character generation)

**Target repo (private until validated):** `remyxai/story-diffusion-flux-modular`
**Block class:** `StoryDiffusionFluxBlock` · **Workflow axis:** consistency (character/subject) · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`. Shares the joint-attention spike with `style-aligned-flux.md` (reuse it).

## One-liner
Generate a **set** of images of the **same character** across different scenes/poses with a consistent identity,
training-free, via **Consistent Self-Attention** — ported to FLUX MMDiT. The storytelling/comic/brand-character
workflow; no diffusers equivalent.

## Scope
**v1 = the training-free IMAGE consistency only** (Consistent Self-Attention) + the comic compositor. Out of scope:
- the reference's **motion predictor for long video** (video, heavy) — do not port it.
- the demo's optional **PhotoMaker** mode (real-face identity from an uploaded photo) — a nice **v2** that ties to
  our PuLID work (`remyxai/pulid-flux-modular`); note it as a follow-up, don't build in v1.

## Vetting (done)
- **Base:** reference is SD1.5/SDXL → port to **FLUX** (MMDiT). ✓ image, single-A100.
- **License/repo:** `HVision-NKU/StoryDiffusion` — **Apache-2.0**, ⭐6453 (NeurIPS 2024 spotlight). ✓ strong signal.
- **Not in diffusers:** code hits = mentions only, no pipeline → **gap**. ✓
- **In-flight:** none. ✓
- **Demand:** consistent character across a set is a marquee, widely-wanted workflow; the ⭐ backs it.
- Paper: "StoryDiffusion: Consistent Self-Attention for Long-Range Image and Video Generation"
  (arXiv 2405.01434 — **[VERIFY] id + authors via arxiv**).

## Mechanism (from the ref `SpatialAttnProcessor2_0`; demo: HF Space `YupengZhou/StoryDiffusion/app.py`)
**Consistent Self-Attention** is a two-phase attention processor, training-free:
- **Write phase:** generate the first `id_length` "identity" images and **cache their self-attention hidden states**.
- **Inference phase:** for each following story frame, **concatenate the cached identity features into the
  frame's self-attention K/V** → the subject stays consistent across frames (no training, no reference net).
- Applied by feature-map resolution with a **paired-attention degree** `sa32`/`sa64` (default **0.7** at 32²/64²),
  via masks (`mask1024`/`mask4096`); masking probability **decays over steps** (~0.3 early → 0.1 late).

Params (default to the ref): `sa32=0.7`, `sa64=0.7`, `id_length=3` (range 2–4), `total_length=id_length+1`.
Port target = a **FLUX joint-attention** processor doing the same write/cache + inference/concat on the image stream.

## Modular block design
- **Injection pattern:** *attention-processor swap* (HRDiT pattern) — a FLUX joint-attention processor with the
  write/cache + inference/concat behavior above. Restore in `finally`; **no-op when disabled** (== independent gens).
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `character_prompt`(required — subject description with a trigger word, e.g. `"a woman img"`),
  `scene_prompts`(required, list — one per frame; a `"#caption"` suffix sets that panel's caption; a `"[NC]"`
  prefix generates a scene **without** the character), `sa32`(0.7), `sa64`(0.7), `id_length`(3),
  `guidance_scale`(3.5), `num_inference_steps`(28), `height`/`width`(1024), `comic_layout`(None|"four-panel"|"classic"),
  `generator`, `output_type`.
- `__call__`: compose full prompts (`character_prompt` + each scene) → **write phase** on the first `id_length`
  (cache identity features) → **inference phase** for the story frames (concat cached features) → decode →
  if `comic_layout`, compose panels + captions (PIL `ImageFont` typesetting) → `bs.images=[...]` (frames, and the
  comic sheet if requested).
- **Comic composition helper** (vendored, from the demo): lay panels into a grid + draw captions extracted from
  the `"#..."` suffixes — this is the marquee output.

## Tests
- **smoke.ipynb:** publish PRIVATE → load (assert `StoryDiffusionFluxBlock`) → 1 character + 2 scenes at low
  res/few steps → returns the frames.
- **e2e.ipynb:** the **comic demo** — one character + 4–6 scene prompts with `"#captions"` → consistent-character
  panels composed into a comic sheet; **validate subject consistency** ON vs OFF with a quantitative
  subject-similarity metric across panels (DreamSim / CLIP-I, or ArcFace for faces) — ON markedly higher.
  **Spike first:** the FLUX joint-attention write/cache + inference/concat seam (shares the StyleAligned spike)
  + no-op-when-off.

## Risks / de-risk
1. **Main risk (shared with StyleAligned):** SD self-attention → FLUX **joint** attention (text+image in one
   attention). The cross-batch token sampling must inject into the image-token K/V of the joint stream — the spike gates it.
2. StoryDiffusion's **random token sampling** across batch items — determinism/seed handling in the modular loop.
3. Keep the **video motion module out** (v1 is image consistency only).

## Deliverables
`block.py`, `modular_config.json`, `modular_model_index.json`, `smoke.ipynb`, `e2e.ipynb`, `README.md` (verified citation).
