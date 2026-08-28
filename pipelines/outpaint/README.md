---
library_name: diffusers
tags:
  - modular-diffusers
  - custom-block
  - flux
  - outpainting
  - image-extension
  - image-to-image
  - training-free
license: other
license_name: flux-1-dev-non-commercial-license
license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md
---

# FLUX Outpainting — training-free canvas extension (Modular Diffusers custom block)

Extend an image beyond its borders — no fine-tuning — as a
[Modular Diffusers](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks)
custom block. The source is pasted onto a larger canvas and **only the new margins** are
inpainted with **FLUX.1-Fill**, which conditions on the unmasked context so the extension
continues the scene instead of replacing it. There is no method here to train and no extra
weight to download: this is pure geometry plus the inpainting variant of FLUX, which is exactly
why diffusers ships FLUX-Fill *inpaint* but no FLUX *outpaint* pipeline — the gap this block
fills.

`left`/`right`/`top`/`bottom` take pixels or a 0–1 fraction of the source size, or give a
`target_height`/`target_width` and the source is centered. Per-side margins are honoured
exactly, and the interior is pasted back bit-exactly afterwards, so the original pixels are
guaranteed untouched no matter what the sampler does.

## Usage

```python
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("remyxai/outpaint-flux-modular", trust_remote_code=True)
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

wide = pipe(
    image="portrait.jpg",            # the image to extend
    prompt="the same beach continuing, overcast sky",   # scene for the NEW area only
    left=256, right=256,             # px, or 0.25 for a quarter of the source
    top=0, bottom=256,
    guidance_scale=30, num_inference_steps=30,
).images[0]
wide.save("wide.png")
```

Or target an output size directly — the source is centered in it:

```python
wide = pipe(image="portrait.jpg", target_width=1536, target_height=1024).images[0]
```

**Dependencies:** `pip install transformers accelerate` (torch ≥ 2.4, diffusers ≥ 0.40). On
first run it downloads FLUX.1-dev and FLUX.1-Fill-dev; **no other weights** — no LoRA, no
segmentation model, no face encoder.

## How it works

- **Compose, don't hook** — the injection seam is the CatVTON one: a `FluxFillPipeline` is built
  from the loaded components (`transformer` from **FLUX.1-Fill-dev**, text encoders / tokenizers /
  VAE / scheduler from **FLUX.1-dev**) and called directly. No forward hooks, no LoRA, no custom
  denoise loop — the only thing this block adds is the canvas and the mask.
- **One keep-box defines everything** — the source box eroded by `mask_feather` px. That single
  box is where the mask is 0 (keep), and it is also the region pasted back from the original
  afterwards, so the two can never disagree. Everything outside it — the new margins plus a thin
  band of the old border — is masked 1 (repaint).
- **`mask_feather` is the seam control** (brief risk #1). At `0` the original border is kept
  verbatim and the seam can read as a hard edge. A few px lets Fill repaint a thin strip of the
  old border so the extension blends into it; the interior is still restored bit-exactly, so
  feathering never costs interior fidelity.
- **Bit-exact interior by construction** — the sampler's output inside the keep-box is discarded
  and overwritten with the source pixels, so "the original region is untouched" is enforced in
  code rather than expected of the model. Canvas edges are rounded up to the VAE's 8px factor,
  with any slack landing on the right/bottom so requested margins are not re-split.

## Key parameters

| arg | default | meaning |
|---|---|---|
| `image` | — | the source image (PIL / numpy RGB / path) |
| `prompt` | `None` | scene for the **new** area; a neutral continuation default if omitted |
| `left`, `right`, `top`, `bottom` | `None` | margin to add, in px or as a 0–1 fraction of the source |
| `target_width`, `target_height` | `None` | alternative to margins: output size, source centered |
| `mask_feather` | 8 | px of the old border also repainted, to hide the seam |
| `guidance_scale` | 30 | FLUX-Fill's recommended scale |
| `num_inference_steps` | 30 | denoise steps |
| `max_sequence_length` | 512 | T5 prompt length |
| `generator` | `None` | seed / generator for reproducible margins |

Passing both margins and a target size raises, as does a target smaller than the source —
outpainting only extends. All-zero margins raise rather than silently returning the input.

## Attribution & AI assistance

Training-free outpainting built directly on **FLUX.1-Fill**
([`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)),
composed the same way as the CatVTON block in this repo
([arXiv:2407.15886](https://arxiv.org/abs/2407.15886), reference implementation
[nftblackmagic/catvton-flux](https://github.com/nftblackmagic/catvton-flux), MIT). Clean-room
implementation from the brief in `briefs/flux-outpainting.md`: the low-star
`alexgenovese/flux-outpainting` sketch was noted as evidence of the gap and was **not** ported.
No method-specific weights are used. The Modular-Diffusers adaptation was authored with AI
assistance (Claude) and validated by the Remyx AI team. Uses FLUX.1-dev / FLUX.1-Fill-dev under
their **non-commercial** license; this derivative inherits it.

## References

No single paper describes this block — it is geometry over an existing inpainting model, so
rather than a citation that would imply a method that does not exist, here is the accurate
lineage:

- **FLUX.1-Fill** — the inpainting variant that does all the generative work:
  [`black-forest-labs/FLUX.1-Fill-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
  (non-commercial license).
- **Outpainting as margin-inpainting** — the framing of pasting onto a larger canvas and
  inpainting only the border is standard practice; this block is a clean-room implementation of
  that idea for FLUX, per the brief.
- **CatVTON** ([arXiv:2407.15886](https://arxiv.org/abs/2407.15886)) — source of the
  compose-a-`FluxFillPipeline` injection pattern and the per-component repo split reused here.
- **Modular Diffusers custom blocks** —
  [the pattern this repo packages](https://huggingface.co/docs/diffusers/main/en/modular_diffusers/custom_blocks).
