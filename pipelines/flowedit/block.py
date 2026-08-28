# FlowEdit for FLUX — training-free, inversion-free text-based image editing as a Modular Diffusers block.
#
# Method: FlowEdit ("Inversion-Free Text-Based Editing Using Pre-Trained Flow Models", arXiv:2412.08629,
# Kulikov, Kleiner, Huberman-Spiegelglas, Michaeli). Reference: fallenshock/FlowEdit (MIT).
# Instead of inverting the source image, FlowEdit builds an ODE that transports it directly to the edit:
# per step it forms the guided velocity DIFFERENCE between the target-prompt and source-prompt predictions
# and integrates it into the running edit latent (over an n_max..n_min step window, averaged over n_avg).
# No inversion, no training, no extra weights — more faithful + faster than inversion-based editing.
#
# Modular-Diffusers adaptation: a self-contained custom denoise loop (two transformer calls per step).
# Authored with AI assistance (Claude), validated by the Remyx AI team; method credit to the FlowEdit authors.
# Uses FLUX.1-dev (non-commercial license).

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"


class FlowEditBlock(ModularPipelineBlocks):
    """Training-free, inversion-free text-based editing: transport a source image from its source prompt to a
    target prompt via FlowEdit's guided velocity-difference ODE (single self-contained denoise loop)."""

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
            InputParam("image", required=True),           # source image: PIL / numpy RGB / path
            InputParam("prompt", required=True),           # TARGET prompt (the edit)
            InputParam("source_prompt", required=True),    # describes the source image
            InputParam("prompt_2", default=None),
            InputParam("source_prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=None),            # default: source image size (snapped to /16)
            InputParam("width", default=None),
            InputParam("T_steps", default=28),
            InputParam("n_avg", default=1),
            InputParam("src_guidance_scale", default=1.5),
            InputParam("tar_guidance_scale", default=5.5),
            InputParam("n_min", default=0),
            InputParam("n_max", default=24),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
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

    @staticmethod
    def _to_pil(x):
        from PIL import Image
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        import numpy as _np
        return Image.fromarray(_np.asarray(x)).convert("RGB")

    def _calc_v(self, tr, latents, prompt_embeds, pooled, text_ids, img_ids, guidance, t, dtype):
        return tr(hidden_states=latents, timestep=(t.expand(latents.shape[0]) / 1000).to(dtype),
                  guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                  txt_ids=text_ids, img_ids=img_ids, joint_attention_kwargs=None, return_dict=False)[0]

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)     # 8
        quant = vsf * 2                                          # 16
        num_channels_latents = tr.config.in_channels // 4       # 16
        T_steps, n_avg = int(bs.T_steps), int(bs.n_avg)
        n_min, n_max = int(bs.n_min), int(bs.n_max)
        guidance_embeds = tr.config.guidance_embeds

        # --- source image -> latents ---
        src = self._to_pil(bs.image)
        H = int(bs.height) if bs.height else (src.height // quant) * quant
        W = int(bs.width) if bs.width else (src.width // quant) * quant
        img_proc = VaeImageProcessor(vae_scale_factor=vsf)
        pixels = img_proc.preprocess(src, height=H, width=W).to(device=device, dtype=vae.dtype)
        x_src = vae.encode(pixels).latent_dist.sample(generator=bs.generator)
        x_src = (x_src - vae.config.shift_factor) * vae.config.scaling_factor
        x_src = x_src.to(dtype)
        batch = x_src.shape[0]
        lh, lw = x_src.shape[2], x_src.shape[3]
        x_src_packed = FluxPipeline._pack_latents(x_src, batch, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        # --- prompts (source + target) ---
        src_embeds, src_pooled, src_text_ids = self._encode_prompt(
            components, bs.source_prompt, bs.source_prompt_2, int(bs.max_sequence_length), device, dtype)
        tar_embeds, tar_pooled, tar_text_ids = self._encode_prompt(
            components, bs.prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)

        if guidance_embeds:
            src_g = torch.full([1], float(bs.src_guidance_scale), device=device, dtype=torch.float32).expand(batch)
            tar_g = torch.full([1], float(bs.tar_guidance_scale), device=device, dtype=torch.float32).expand(batch)
        else:
            src_g = tar_g = None

        # --- timesteps (capped mu, same schedule as generation) ---
        sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
        cfg = scheduler.config
        mu = calculate_shift(x_src_packed.shape[1], cfg.get("base_image_seq_len", 256),
                             cfg.get("max_image_seq_len", 4096), cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, sigmas=sigmas, mu=mu)

        # --- FlowEdit ODE (inversion-free) ---
        zt_edit = x_src_packed.clone()
        xt_tar = None
        for i, t in enumerate(timesteps):
            if T_steps - i > n_max:
                continue
            scheduler._init_step_index(t)
            t_i = scheduler.sigmas[scheduler.step_index]
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]

            if T_steps - i > n_min:
                V_delta = torch.zeros_like(x_src_packed)
                for _ in range(n_avg):
                    noise = randn_tensor(x_src_packed.shape, generator=bs.generator, device=device, dtype=dtype)
                    zt_src = (1 - t_i) * x_src_packed + t_i * noise
                    zt_tar = zt_edit + zt_src - x_src_packed
                    Vt_src = self._calc_v(tr, zt_src, src_embeds, src_pooled, src_text_ids, img_ids, src_g, t, dtype)
                    Vt_tar = self._calc_v(tr, zt_tar, tar_embeds, tar_pooled, tar_text_ids, img_ids, tar_g, t, dtype)
                    V_delta += (1.0 / n_avg) * (Vt_tar - Vt_src)
                zt_edit = (zt_edit.to(torch.float32) + (t_im1 - t_i) * V_delta).to(dtype)
            else:  # last n_min steps: SDEdit-style regular sampling of the target
                if i == T_steps - n_min:
                    noise = randn_tensor(x_src_packed.shape, generator=bs.generator, device=device, dtype=dtype)
                    xt_src = t_i * noise + (1.0 - t_i) * x_src_packed
                    xt_tar = zt_edit + xt_src - x_src_packed
                Vt_tar = self._calc_v(tr, xt_tar, tar_embeds, tar_pooled, tar_text_ids, img_ids, tar_g, t, dtype)
                xt_tar = (xt_tar.to(torch.float32) + (t_im1 - t_i) * Vt_tar).to(dtype)

        out = zt_edit if n_min == 0 else xt_tar

        # --- decode ---
        if bs.output_type == "latent":
            image = out
        else:
            lat = FluxPipeline._unpack_latents(out, H, W, vsf)
            lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
            image = img_proc.postprocess(image, output_type=bs.output_type)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
