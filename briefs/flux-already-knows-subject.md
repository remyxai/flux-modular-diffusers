# Brief — "Flux Already Knows" → Modular Diffusers (training-free subject-driven generation)

**Target repo (private until validated):** `remyxai/flux-subject-modular`
**Block class:** `FluxSubjectBlock` · **Axis:** personalization (subject) · **Mode:** training-free (no extra weights)
**Prereq:** read `../CONVENTIONS.md`. New axis — complements PuLID (which uses trained *face* weights); this is
**generic subject-driven, training-free**.

## One-liner
Put a **reference subject** (an object/product/character from one image) into new generations — training-free,
no LoRA, no extra weights — by activating FLUX's own capabilities. A clean complement to PuLID (faces) and
CatVTON (garments): arbitrary subjects.

## Vetting (done)
- **Base:** **FLUX** (the method is FLUX-specific — "Flux Already Knows"). ✓
- **License/repo:** **[VERIFY]** locate official repo + license (arXiv "Code" link) before build.
- **Not in diffusers:** no subject-driven-training-free FLUX pipeline → **gap** (Redux is variation, not subject-preserving). ✓
- Paper: arXiv 2504.11478 — VERIFIED authors: Kang, Fotiadis, Jiang, Yan, Jia, Liu.

## Mechanism (encode from the ref)
Training-free subject injection — read the ref for specifics; likely conditions generation on the reference
image's features via attention (inject subject K/V) and/or a noise/inversion scheme, "activating" subject
fidelity without training. Maps to our K/V-sharing / reference-attention seam (attn-spike).

## Modular block design
- **Injection pattern:** *attention-processor swap threading the reference-subject K/V via `joint_attention_kwargs`*
  (the attn-spike K/V-share path, reference = the subject image's tokens). Restore in `finally`; no-op when off.
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `subject_image`(reference), `prompt`(the new scene), subject-strength / which-layers knobs (default to
  ref; expect a share_start_frac-style knob to balance subject-fidelity vs prompt-adherence, cf. StyleAligned lesson),
  `guidance_scale`, `generator`, `output_type`.
- `__call__`: encode the subject image (VAE + a reference forward to capture K/V, or the ref's scheme) → denoise
  the new prompt with subject K/V injected → decode.

## Tests
- **smoke.ipynb:** load (assert `FluxSubjectBlock`) → subject image + a prompt → image.
- **e2e.ipynb:** a subject in 2–3 new scenes; **validate subject fidelity** (DreamSim/CLIP-I to the reference)
  AND prompt-adherence (CLIP to the scene text) — both should hold. **Spike:** subject-K/V injection seam; watch for
  the StyleAligned-style over-collapse (subject fidelity ↑ but prompt ignored) → tune the layer/strength knob.

## Risks
1. **[VERIFY] repo/license** first.
2. Fidelity-vs-prompt balance (same knob lesson as StyleAligned/StoryDiffusion — expect a later-layers/strength dial).

## Deliverables
`block.py` · configs · `smoke.ipynb` · `e2e.ipynb` · `README.md` (verified citation + repo/license).
