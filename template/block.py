# <Method> for FLUX — <one-line description> (Modular Diffusers custom block).
#
# Method: <paper title> (arXiv:<id>, <authors>). Reference: <repo> (<license>).
# See ../../CONTRIBUTING.md and ../../CONVENTIONS.md. Authored with AI assistance (Claude), validated by Remyx AI.
# Uses FLUX.1-dev (non-commercial license).

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"   # or a per-component mix (e.g. transformer from FLUX.1-Fill-dev)


class MethodBlock(ModularPipelineBlocks):
    """<one-line>. See CONVENTIONS.md for the injection patterns; keep any feature a bit-exact no-op when off."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FLUX = _FLUX

    @property
    def expected_components(self):
        F_ = self._FLUX
        return [
            ComponentSpec("text_encoder", CLIPTextModel, pretrained_model_name_or_path=F_, subfolder="text_encoder"),
            ComponentSpec("tokenizer", CLIPTokenizer, pretrained_model_name_or_path=F_, subfolder="tokenizer"),
            ComponentSpec("text_encoder_2", T5EncoderModel, pretrained_model_name_or_path=F_, subfolder="text_encoder_2"),
            ComponentSpec("tokenizer_2", T5TokenizerFast, pretrained_model_name_or_path=F_, subfolder="tokenizer_2"),
            ComponentSpec("transformer", FluxTransformer2DModel, pretrained_model_name_or_path=F_, subfolder="transformer"),
            ComponentSpec("vae", AutoencoderKL, pretrained_model_name_or_path=F_, subfolder="vae"),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, pretrained_model_name_or_path=F_, subfolder="scheduler"),
        ]

    @property
    def inputs(self):
        return [
            InputParam("prompt", required=True),
            # InputParam("image", required=True),          # for editing/processing pipelines
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=3.5),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
            # ... method-specific knobs (default to the reference's values) ...
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    def _encode_prompt(self, components, prompt, prompt_2, max_sequence_length, device, dtype):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt if prompt_2 is None else ([prompt_2] if isinstance(prompt_2, str) else prompt_2)
        tok, te = components.tokenizer, components.text_encoder
        clip_ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True,
                       return_overflowing_tokens=False, return_length=False, return_tensors="pt").input_ids
        pooled = te(clip_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=te.dtype, device=device)
        tok2, te2 = components.tokenizer_2, components.text_encoder_2
        t5_ids = tok2(prompt_2, padding="max_length", max_length=max_sequence_length, truncation=True,
                      return_length=False, return_overflowing_tokens=False, return_tensors="pt").input_ids
        prompt_embeds = te2(t5_ids.to(device), output_hidden_states=False)[0].to(dtype=te2.dtype, device=device)
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
        return prompt_embeds, pooled, text_ids

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        # TODO: implement the method (see CONVENTIONS.md "Injection patterns").
        #   - install any hook/processor; restore it in `finally`
        #   - keep the feature a bit-exact no-op when disabled
        #   - standard FLUX plumbing: _pack_latents / _prepare_latent_image_ids / calculate_shift / retrieve_timesteps
        raise NotImplementedError("implement __call__")
        bs.images = [...]
        self.set_block_state(state, bs)
        return components, state
