# CatVTON for FLUX — training-free virtual try-on as a Modular Diffusers custom block.
#
# Method: CatVTON ("Concatenation Is All You Need for Virtual Try-On", arXiv:2407.15886) on FLUX.1-Fill:
# the garment and person are concatenated side-by-side into one latent canvas, and the person's clothing
# region is inpainted conditioned on the garment half — so the reference garment is injected into the
# diffusion *context* by latent concatenation (fidelity that a cross-attention feed can't match).
# Reference: nftblackmagic/catvton-flux (MIT); LoRA weights: xiaozaa/catvton-flux-lora-alpha.
#
# This is the Modular-Diffusers adaptation: a FluxFillPipeline is composed from the loaded components with
# the CatVTON LoRA applied to the transformer; the block builds the concat canvas + extended mask and crops
# the try-on half. v1 takes a supplied agnostic mask (auto-masking is a follow-up). Uses FLUX.1-dev /
# FLUX.1-Fill-dev (non-commercial research license).
#
# Authored with AI assistance (Claude), validated by the Remyx AI team; method credit to the CatVTON authors.

import numpy as np
import torch
from torchvision import transforms

from diffusers import (
    FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxFillPipeline,
)
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FILL = "black-forest-labs/FLUX.1-Fill-dev"      # transformer source (inpainting variant, gated)
_BASE = "black-forest-labs/FLUX.1-dev"           # vae / text-encoders / scheduler source
_LORA_REPO = "xiaozaa/catvton-flux-lora-alpha"
_LORA_FILE = "pytorch_lora_weights.safetensors"
_PROMPT = ("The pair of images highlights a clothing and its styling on a model, high resolution, 4K, 8K; "
           "[IMAGE1] Detailed product shot of a clothing "
           "[IMAGE2] The same cloth is worn by a model in a lifestyle setting.")


class CatVTONFluxBlock(ModularPipelineBlocks):
    """Training-free virtual try-on: concatenate garment+person into one canvas and inpaint the person's
    clothing region (CatVTON on FLUX.1-Fill with the catvton LoRA)."""

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
            InputParam("person_image", required=True),   # PIL / numpy RGB / path
            InputParam("garment_image", required=True),  # PIL / numpy RGB / path
            InputParam("mask", default=None),            # optional agnostic mask (white=replace); auto-generated if omitted
            InputParam("height", default=768),
            InputParam("width", default=576),            # per-side; the canvas is 2×width
            InputParam("num_inference_steps", default=30),
            InputParam("guidance_scale", default=30.0),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    @staticmethod
    def _to_pil(x):
        from PIL import Image
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        return Image.fromarray(np.asarray(x)).convert("RGB")

    def _auto_mask(self, person_pil, device):
        """Agnostic upper-body mask via segformer clothes-parsing: cover the garment region + arms
        (so sleeves can form), exclude the lower body (keep the person's pants), fill to a solid region."""
        import numpy as np, cv2, torch
        from PIL import Image
        if getattr(self, "_masker", None) is None:
            from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
            proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device).eval()
            self._masker = (proc, model)
        proc, model = self._masker
        W, H = person_pil.size
        with torch.no_grad():
            inp = proc(images=person_pil, return_tensors="pt").to(device)
            logits = model(**inp).logits
            up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            labels = up.argmax(1)[0].cpu().numpy()
        UPPER, ARMS = {4, 7}, {14, 15}                       # upper-clothes/dress + arms
        region = np.isin(labels, list(UPPER | ARMS)).astype(np.uint8) * 255
        region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        region = cv2.dilate(region, np.ones((15, 15), np.uint8), iterations=2)
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        region = cv2.drawContours(np.zeros_like(region), cnts, -1, 255, thickness=-1)
        return Image.fromarray(region).convert("RGB")

    def _get_pipe(self, components):
        if getattr(self, "_pipe", None) is None:
            tr = components.transformer
            sd, alphas = FluxFillPipeline.lora_state_dict(
                pretrained_model_name_or_path_or_dict=_LORA_REPO, weight_name=_LORA_FILE, return_alphas=True)
            if not all(("lora" in k or "dora_scale" in k) for k in sd.keys()):
                raise ValueError("Unexpected CatVTON LoRA checkpoint format.")
            FluxFillPipeline.load_lora_into_transformer(state_dict=sd, network_alphas=alphas, transformer=tr)
            self._pipe = FluxFillPipeline(
                vae=components.vae, text_encoder=components.text_encoder, tokenizer=components.tokenizer,
                text_encoder_2=components.text_encoder_2, tokenizer_2=components.tokenizer_2,
                transformer=tr, scheduler=components.scheduler)
        return self._pipe

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        w, h = int(bs.width), int(bs.height)
        device = components.transformer.device
        person = self._to_pil(bs.person_image).resize((w, h))
        garment = self._to_pil(bs.garment_image).resize((w, h))
        mask = (self._to_pil(bs.mask).resize((w, h)) if bs.mask is not None
                else self._auto_mask(person, device))

        to01 = transforms.ToTensor()                                    # [0,1]; FluxFillPipeline's image_processor
        person_t = to01(person)                                         # normalizes to [-1,1] internally (current diffusers
        garment_t = to01(garment)                                       # expects [0,1]; passing [-1,1] corrupts conditioning)
        mask_t = to01(mask)[:1]                                         # single channel, [0,1]

        canvas = torch.cat([garment_t, person_t], dim=2)                 # [garment | person] along width
        ext_mask = torch.cat([torch.zeros_like(mask_t), mask_t], dim=2)  # keep garment half, inpaint person half

        pipe = self._get_pipe(components)
        result = pipe(
            height=h, width=w * 2, image=canvas, mask_image=ext_mask,
            num_inference_steps=int(bs.num_inference_steps), guidance_scale=float(bs.guidance_scale),
            generator=bs.generator, max_sequence_length=512, prompt=_PROMPT,
        ).images[0]

        tryon = result.crop((w, 0, w * 2, h))                           # right half = the try-on
        bs.images = [tryon if bs.output_type == "pil" else np.array(tryon)]
        self.set_block_state(state, bs)
        return components, state
