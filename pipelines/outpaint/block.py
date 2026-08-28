# FLUX Outpainting — training-free canvas extension as a Modular Diffusers custom block.
#
# Method: paste the source image onto a larger canvas and inpaint only the new margins with
# FLUX.1-Fill (which conditions on the unmasked context), so the model synthesises an
# extension that continues the scene beyond the original border. No weights beyond FLUX Fill:
# this is pure plumbing around the inpainting variant, not a trained method — hence the brief
# vets it as a gap (diffusers ships FLUX-Fill *inpaint*, but no FLUX outpaint pipeline).
# Clean-room build; the low-star `alexgenovese/flux-outpainting` sketch was noted in the brief
# only as evidence of the gap, not ported.
#
# This is the Modular-Diffusers adaptation, and it reuses the CatVTON component setup /
# injection seam verbatim: a FluxFillPipeline is composed from the loaded components
# (transformer from FLUX.1-Fill-dev, everything else from FLUX.1-dev) — no hooks, no LoRA.
# The block's own work is geometry: place the source at (left, top) on an enlarged canvas,
# build a margin mask that keeps the original pixels and repaints only the border, and finally
# paste the original region back bit-exactly so the interior can never drift.
#
# Uses FLUX.1-dev / FLUX.1-Fill-dev (non-commercial research license).
#
# Authored with AI assistance (Claude), validated by the Remyx AI team.

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from diffusers import (
    FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline,
)
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FILL = "black-forest-labs/FLUX.1-Fill-dev"      # transformer source (inpainting variant, gated)
_BASE = "black-forest-labs/FLUX.1-dev"           # vae / text-encoders / scheduler source
_VAE_SCALE = 8                                   # FLUX VAE spatial down factor (canvas must be divisible)
_DEFAULT_PROMPT = ("seamlessly continue this scene beyond the frame, consistent lighting and "
                   "perspective, high resolution")


def _round_to_step(v, step=_VAE_SCALE, minimum=_VAE_SCALE):
    """Round a pixel size up to a multiple of `step` (the VAE's spatial down factor), so the
    padded canvas survives encode/decode without a size error or a silent resize."""
    return max(minimum, int(np.ceil(v / step)) * step)


class OutpaintBlock(ModularPipelineBlocks):
    """Training-free outpainting: extend an image by `left`/`right`/`top`/`bottom` margins by
    inpainting only the new border with FLUX.1-Fill (composed from components — no LoRA)."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FILL = _FILL
    _BASE = _BASE

    @property
    def expected_components(self):
        return [
            ComponentSpec("text_encoder", CLIPTextModel, pretrained_model_name_or_path=_BASE, subfolder="text_encoder"),
            ComponentSpec("tokenizer", CLIPTokenizer, pretrained_model_name_or_path=_BASE, subfolder="tokenizer"),
            ComponentSpec("text_encoder_2", T5EncoderModel, pretrained_model_name_or_path=_BASE, subfolder="text_encoder_2"),
            ComponentSpec("tokenizer_2", T5TokenizerFast, pretrained_model_name_or_path=_BASE, subfolder="tokenizer_2"),
            ComponentSpec("transformer", FluxTransformer2DModel, pretrained_model_name_or_path=_FILL, subfolder="transformer"),
            ComponentSpec("vae", AutoencoderKL, pretrained_model_name_or_path=_BASE, subfolder="vae"),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, pretrained_model_name_or_path=_BASE, subfolder="scheduler"),
        ]

    @property
    def inputs(self):
        return [
            InputParam("image", required=True),            # the source (PIL / numpy RGB / path)
            InputParam("prompt", default=None),             # scene for the NEW area; sensible default if omitted
            InputParam("left", default=None),               # px to add, or a 0-1 fraction of the source size
            InputParam("right", default=None),
            InputParam("top", default=None),
            InputParam("bottom", default=None),
            InputParam("target_height", default=None),      # OR an explicit output size; the source is centered
            InputParam("target_width", default=None),
            InputParam("mask_feather", default=8),          # px of mask dilation over the old border (seam relief)
            InputParam("num_inference_steps", default=30),
            InputParam("guidance_scale", default=30.0),     # FLUX-Fill's recommended scale
            InputParam("max_sequence_length", default=512),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    # ---------- geometry (pure, CPU-only — unit-tested without GPU) ----------

    @staticmethod
    def _to_pil(x):
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.fromarray(np.asarray(x)).convert("RGB")

    @staticmethod
    def _resolve_margin(m, size):
        """A margin is pixels (`int`/`float` > 1) or a 0-1 fraction of `size`. `None` -> 0."""
        if m is None:
            return 0
        m = float(m)
        if 0 < m <= 1:
            return int(round(m * size))
        return max(0, int(m))

    @classmethod
    def _canvas_geometry(cls, w, h, left, right, top, bottom, target_width, target_height):
        """Resolve margins (px or fraction) OR a target size into (canvas_w, canvas_h, x, y) —
        the paste origin of the source on the enlarged canvas. Sizes are rounded up to the VAE's
        8px down-factor. With no margins and no target at all, the geometry is the source itself
        (nothing to outpaint), which the caller reports rather than silently no-op'ing."""
        if target_width is not None or target_height is not None:
            if any(m is not None for m in (left, right, top, bottom)):
                raise ValueError("Pass either margins (left/right/top/bottom) or a target size, not both.")
            if (target_width is not None and target_width < w) or (target_height is not None and target_height < h):
                raise ValueError(f"target {target_width}x{target_height} is smaller than the source "
                                 f"{w}x{h} — cannot crop; outpainting only extends.")
            tw = _round_to_step(target_width if target_width is not None else w, minimum=w)
            th = _round_to_step(target_height if target_height is not None else h, minimum=h)
            return tw, th, (tw - w) // 2, (th - h) // 2

        # Per-side margins are honoured exactly: the paste origin IS the left/top margin, so
        # `left=256` really does put 256px of new content on the left and none on the right. Any
        # px needed to reach the VAE's 8px down-factor lands on the trailing (right/bottom) side,
        # never by re-splitting the requested margins.
        ml, mr = cls._resolve_margin(left, w), cls._resolve_margin(right, w)
        mt, mb = cls._resolve_margin(top, h), cls._resolve_margin(bottom, h)
        cw = _round_to_step(w + ml + mr, minimum=w)
        ch = _round_to_step(h + mt + mb, minimum=h)
        return cw, ch, ml, mt

    @staticmethod
    def _keep_box(x, y, w, h, feather):
        """The region FLUX-Fill must leave alone: the source box eroded by `feather` px, so a thin
        band of the *old* border is also repainted and the seam blends (brief risk #1). Clamped so
        the box always keeps at least a 2px core. This one box defines both the mask's zeros and
        the bit-exact paste-back below — they cannot disagree."""
        f = min(max(0, int(feather)), max(0, (min(w, h) - 2) // 2))
        return x + f, y + f, w - 2 * f, h - 2 * f

    @staticmethod
    def _build(canvas_w, canvas_h, box, keep_box, src_t):
        """Assemble the [0,1] canvas + mask FLUX-Fill consumes.

        `canvas` carries the *whole* source pasted at `box` — Fill needs it as context — while
        `mask` is 0 (keep) only over `keep_box` (the source eroded by the feather) and 1 (repaint)
        elsewhere, i.e. the new margins plus a thin band of the old border."""
        kx, ky, kw, kh = keep_box
        x, y, w, h = box
        canvas = torch.full((3, canvas_h, canvas_w), 0.5)
        mask = torch.ones(1, canvas_h, canvas_w)
        mask[:, ky:ky + kh, kx:kx + kw] = 0
        canvas[:, y:y + h, x:x + w] = src_t
        return canvas, mask

    # ---------- the FLUX-Fill seam (CatVTON's, minus the LoRA) ----------

    def _get_pipe(self, components):
        if getattr(self, "_pipe", None) is None:
            self._pipe = FluxFillPipeline(
                vae=components.vae, text_encoder=components.text_encoder, tokenizer=components.tokenizer,
                text_encoder_2=components.text_encoder_2, tokenizer_2=components.tokenizer_2,
                transformer=components.transformer, scheduler=components.scheduler)
        return self._pipe

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        src = self._to_pil(bs.image)
        w, h = src.size

        if all(m is None or float(m) == 0 for m in (bs.left, bs.right, bs.top, bs.bottom)) \
                and bs.target_width is None and bs.target_height is None:
            raise ValueError("Nothing to outpaint: all margins are zero and no target size was given.")

        cw, ch, x, y = self._canvas_geometry(
            w, h, bs.left, bs.right, bs.top, bs.bottom, bs.target_width, bs.target_height)

        to01 = transforms.ToTensor()                                    # [0,1]; FluxFillPipeline's image_processor
        src_t = to01(src)                                               # normalizes to [-1,1] internally (current diffusers
        kx, ky, kw, kh = self._keep_box(x, y, w, h, bs.mask_feather)    # expects [0,1]; passing [-1,1] corrupts conditioning)
        canvas, mask = self._build(cw, ch, (x, y, w, h), (kx, ky, kw, kh), src_t)
        if float(mask.sum()) == 0.0:
            raise ValueError("mask_feather covers the whole canvas — nothing would be repainted.")

        pipe = self._get_pipe(components)
        prompt = bs.prompt if bs.prompt not in (None, "") else _DEFAULT_PROMPT
        result = pipe(
            height=ch, width=cw, image=canvas, mask_image=mask,
            num_inference_steps=int(bs.num_inference_steps), guidance_scale=float(bs.guidance_scale),
            generator=bs.generator, max_sequence_length=int(bs.max_sequence_length), prompt=prompt,
        ).images[0]

        # Bit-exact interior: whatever Fill produced inside the keep-box is replaced by the
        # original pixels. The e2e claim is "interior pixel-equal" — enforced, not hoped for.
        out = np.asarray(result).copy()
        out[ky:ky + kh, kx:kx + kw] = np.asarray(src)[(ky - y):(ky - y) + kh, (kx - x):(kx - x) + kw]

        bs.images = [out if bs.output_type != "pil" else Image.fromarray(out)]
        self.set_block_state(state, bs)
        return components, state
