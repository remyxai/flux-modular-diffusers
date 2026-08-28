# Brief — FlowEdit → Modular Diffusers (training-free image editing)

**Target repo (private until validated):** `remyxai/flowedit-flux-modular`
**Block class:** `FlowEditBlock` · **Workflow axis:** editing · **Mode:** training-free (no extra weights)
**Prereq:** read `../CONVENTIONS.md` — build to that house style.

## One-liner
Training-free, **inversion-free** text-based image editing on FLUX: given a source image + source prompt +
target prompt, transport the image to the edit without inverting it (more faithful + faster than
inversion-based editing). Fills the editing gap — diffusers has only `pipeline_flux_rf_inversion` today.

## Vetting (done)
- **Base:** flow models — **FLUX**.1-dev / SD3 (reference is FLUX-native). ✓ image, single-A100.
- **License/repo:** `fallenshock/FlowEdit` — **MIT**, ⭐1014, active (pushed 2026-08). ✓ permissive.
- **Not in diffusers:** community-dir grep = none; code hits = 0; only `rf_inversion` exists → **gap**. ✓
- **In-flight:** no FlowEdit PR/issue in diffusers. ✓
- **Demand:** editing is the most-requested FLUX workflow; inversion-free is a real improvement. ✓
- Paper: FlowEdit (arXiv 2412.08629 — VERIFIED: Kulikov, Kleiner, Huberman-Spiegelglas, Michaeli).

## Mechanism (to encode faithfully from the ref)
FlowEdit builds an ODE that maps source→target **without inversion**: at each step it forms the guided
velocity **difference** between the target-prompt and source-prompt model predictions and integrates it from
the source latents over a step window (`n_min…n_max`), averaging `n_avg` noise draws. Key knobs (default to the
ref): `T_steps`, `n_avg`, `src_guidance_scale`, `tgt_guidance_scale`, `n_min`, `n_max`.

## Modular block design
- **Injection pattern:** *custom denoise loop* (no hooks) — two transformer calls per step (source-cond,
  target-cond), take the guided difference, integrate. Reuse `_encode_prompt` for BOTH prompts.
- `expected_components`: the 7 FLUX.1-dev comps (CONVENTIONS).
- `inputs`: `image`(required, source), `prompt`(required, = target prompt), `source_prompt`(required),
  `T_steps`(default 28), `n_avg`(1), `src_guidance_scale`(1.5), `tgt_guidance_scale`(5.5), `n_min`(0),
  `n_max`(24), `height`/`width`(auto from image), `generator`, `output_type`.
- `__call__`: encode src+tgt prompts → VAE-encode source image to latents → FlowEdit ODE loop (per step: src &
  tgt velocities via the FLUX transformer using the guidance embed, guided difference, integrate over the
  window) → decode → `bs.images=[...]`.
- **Deps:** none beyond diffusers/transformers (training-free).

## Tests
- **smoke.ipynb:** publish PRIVATE → load via trust_remote_code (assert `FlowEditBlock`) → tiny edit (small res,
  few steps) on a sample image (e.g. an HF docs image) → returns an image, no error.
- **e2e.ipynb:** a real edit (e.g. change garment colour / "cat"→"dog") at full settings; **validate structure
  preservation** (background/pose unchanged) + target change; **spike first:** reproduce ONE edit vs the
  reference `fallenshock/FlowEdit` output (visual parity / small Δ) to confirm the velocity-difference + guidance
  match before finalizing.

## Risks / de-risk
1. Exact guided-difference + src/tgt guidance handling on FLUX's guidance-distilled transformer — the spike gates this.
2. Sigma/timestep schedule + the `n_min…n_max` window mapping to FLUX's flow-match steps.

## Deliverables
`block.py`, `modular_config.json`, `modular_model_index.json`, `smoke.ipynb`, `e2e.ipynb`, `README.md` (verified citation).

---
**STATUS (2026-08-28): BUILT in-session** (Route A — Outrider-into-fork honesty-routed to Issue; see MEMORY). block.py (187L) + configs + `~/Downloads/flowedit_e2e.ipynb` published PRIVATE `remyxai/flowedit-flux-modular`. Pending: run E2E → flip public + collection.
