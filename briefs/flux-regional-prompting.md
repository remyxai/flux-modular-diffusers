# Brief — FLUX Regional Prompting (training-free spatial prompt control)

**Target repo (private until validated):** `remyxai/regional-prompting-flux-modular`
**Block class:** `RegionalPromptingFluxBlock` · **Workflow axis:** control · **Mode:** training-free
**Prereq:** read `../CONVENTIONS.md`.

## One-liner
Different prompts in different regions of one image (training-free) via masked attention. The SD version ships
in diffusers (`regional_prompting_stable_diffusion.py`) — **FLUX is the gap**.

## Vetting (done)
- **Base:** FLUX (MMDiT). ✓ image, single-A100.
- **Mode/license:** training-free (attention masking); no weights → license-clean. ✓
- **Not in diffusers for FLUX:** SD community pipeline exists; no FLUX variant (`regional_prompt` code hit = the SD
  one) → **gap**. ✓
- **In-flight:** #12367 is Qwen-Image Eligen (not FLUX regional) → clear. ✓
- **Demand:** layout/region control is a recurring ask; complements (not duplicates) ControlNet.
- Method: prompt-region masking of attention (no single canonical paper — implement the standard masked-attention
  approach; cite the SD community pipeline + relevant prior art, **[VERIFY]** any paper cited).

## Mechanism
Assign each region (bbox or mask) its own prompt; during attention, mask the text→image attention so image
tokens in region R attend only to region-R's prompt tokens (plus an optional base prompt). For FLUX MMDiT the
mask is applied within the **joint** attention (text+image tokens together).

## Modular block design
- **Injection pattern:** *attention-processor swap threading region masks via `joint_attention_kwargs`* (HRDiT
  pattern — named kwarg so it isn't dropped); restore in `finally`. No-op when a single region/no mask is given.
- `expected_components`: 7 FLUX.1-dev comps.
- `inputs`: `prompts`(required, list, one per region), `regions`(required, list of bbox `[x0,y0,x1,y1]` or masks),
  `base_prompt`(optional, global), `region_weights`(optional), `guidance_scale`(3.5), `num_inference_steps`(28),
  `height`/`width`(1024), `generator`, `output_type`.
- `__call__`: encode each region prompt (+ base) → rasterize region masks to the latent-token grid → install the
  masked-attention processor (feed masks + per-region token spans via `joint_attention_kwargs`) → denoise →
  decode.

## Tests
- **smoke.ipynb:** publish PRIVATE → load (assert class) → 2 regions ("left: red car" / "right: blue house") at
  low res/few steps → returns an image.
- **e2e.ipynb:** a 2–3 region layout at full settings; **validate** each region's content matches its prompt
  (a CLIP score per region-crop vs its prompt is a clean quantitative signal). **Spike first:** the FLUX
  joint-attention masking seam — mapping region masks to the image-token block + restricting text spans within
  the combined attention.

## Risks / de-risk
1. **Main risk:** joint attention masking is different from SD cross-attention — the mask must cover the
   image-token↔text-token sub-block correctly; the spike gates it.
2. Latent-grid rasterization of bbox/mask regions (patchify factor) must align with FLUX's token layout.

## Deliverables
`block.py`, `modular_config.json`, `modular_model_index.json`, `smoke.ipynb`, `e2e.ipynb`, `README.md`.
