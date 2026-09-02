# Regional Prompting for FLUX — training-free spatial prompt control as a Modular Diffusers block.
#
# Different prompts in different image regions, training-free, by masking the joint attention: each image
# token attends to a shared base prompt + only its own region's prompt tokens (never other regions'). The
# SD version ships in diffusers (regional_prompting_stable_diffusion.py); this is the FLUX (MMDiT) port.
#
# Modular-Diffusers adaptation: a FLUX joint-attention processor (delegates to stock when no mask = bit-exact
# no-op; seam validated in attn_spike.ipynb) that applies a per-region key mask threaded via
# joint_attention_kwargs. Authored with AI assistance (Claude), validated by Remyx AI. FLUX.1-dev (non-commercial).

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Shared FLUX-modular attention primitive (vendored flat beside this file for trust_remote_code).
# The region mask is a plain additive attention bias -> the primitive's `bias` op; no payload -> stock.
from .flux_modular import flux_intervention, PAYLOAD_KEY

_FLUX = "black-forest-labs/FLUX.1-dev"


class RegionalPromptingFluxBlock(ModularPipelineBlocks):
    """Training-free regional prompting: base prompt + per-region prompts, routed by a joint-attention mask
    so each image region follows its own prompt."""

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
            InputParam("base_prompt", required=True),          # global scene prompt
            InputParam("regions", required=True),               # list of {"prompt": str, "bbox": [x0,y0,x1,y1] normalized}
            InputParam("region_seq_len", default=128),          # T5 tokens per prompt (kept short to bound the concat)
            InputParam("region_exclusive", default=True),        # assigned tokens see ONLY their region prompt (region dominates the base)
            InputParam("region_isolate_strength", default=0.0),  # cross-region image-attention penalty; 0=off (cleanest look, default); raise to 1-3 only if objects fuse (higher can add a boundary seam)
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=3.5),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    def _encode(self, components, prompt, L, device, dtype):
        tok, te = components.tokenizer, components.text_encoder
        clip_ids = tok([prompt], padding="max_length", max_length=tok.model_max_length, truncation=True,
                       return_tensors="pt").input_ids
        pooled = te(clip_ids.to(device), output_hidden_states=False).pooler_output.to(dtype=te.dtype, device=device)
        tok2, te2 = components.tokenizer_2, components.text_encoder_2
        t5 = tok2([prompt], padding="max_length", max_length=L, truncation=True, return_tensors="pt").input_ids
        emb = te2(t5.to(device), output_hidden_states=False)[0].to(dtype=te2.dtype, device=device)
        return emb, pooled  # (1,L,4096), (1,768)

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1); quant = vsf * 2
        num_channels_latents = tr.config.in_channels // 4
        H, W = int(bs.height), int(bs.width)
        if H % quant or W % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        nsteps = int(bs.num_inference_steps); L = int(bs.region_seq_len)
        regions = list(bs.regions)

        # encode base + each region prompt; concat text
        base_emb, base_pooled = self._encode(components, bs.base_prompt, L, device, dtype)
        embs = [base_emb]; spans = [(0, base_emb.shape[1])]
        off = base_emb.shape[1]
        for rg in regions:
            e, _ = self._encode(components, rg["prompt"], L, device, dtype)
            embs.append(e); spans.append((off, off + e.shape[1])); off += e.shape[1]
        prompt_embeds = torch.cat(embs, dim=1)                  # (1, n_txt, 4096)
        pooled = base_pooled
        n_txt = prompt_embeds.shape[1]
        text_ids = torch.zeros(n_txt, 3, device=device, dtype=dtype)

        lh, lw = 2 * (H // quant), 2 * (W // quant)
        gh, gw = lh // 2, lw // 2                                # packed image-token grid
        n_img = gh * gw
        latents = randn_tensor((1, num_channels_latents, lh, lw), generator=bs.generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, 1, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, gh, gw, device, dtype)

        # region assignment per image token + joint mask
        region_of = np.full(n_img, -1, dtype=np.int64)          # -1 = base only
        ys = (np.arange(n_img) // gw + 0.5) / gh
        xs = (np.arange(n_img) % gw + 0.5) / gw
        for r, rg in enumerate(regions):
            x0, y0, x1, y1 = rg["bbox"]
            inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
            region_of[inside] = r                               # later region wins on overlap
        joint = n_txt + n_img
        base_s = spans[0]
        NEG = -1e4
        bias = torch.zeros(joint, joint, dtype=torch.float32)
        # image-query -> text-key routing (HARD): each token sees its region's prompt (+ base if allowed)
        text_allow = torch.zeros(n_img, n_txt, dtype=torch.bool)
        assigned = torch.from_numpy(region_of >= 0)
        if bool(bs.region_exclusive):
            text_allow[~assigned, base_s[0]:base_s[1]] = True   # only unassigned tokens fall back to base
        else:
            text_allow[:, base_s[0]:base_s[1]] = True           # everyone also sees base
        for r, (s0, s1) in enumerate(spans[1:]):
            text_allow[torch.from_numpy(region_of == r), s0:s1] = True
        bias[n_txt:, :n_txt] = torch.where(text_allow, 0.0, NEG)
        # image-query -> image-key: SOFT cross-region penalty (separate objects without a hard seam)
        strength = float(bs.region_isolate_strength)
        if strength > 0:
            ro = torch.from_numpy(region_of)
            img_allow = torch.zeros(n_img, n_img, dtype=torch.bool)
            for r in range(len(regions)):
                sel = ro == r
                img_allow |= (sel.unsqueeze(1) & sel.unsqueeze(0))   # within own region
            un = ro == -1
            img_allow |= un.unsqueeze(0)                              # all may attend to background
            img_allow |= un.unsqueeze(1)                              # background attends to all
            img_allow |= torch.eye(n_img, dtype=torch.bool)          # self
            bias[n_txt:, n_txt:] = torch.where(img_allow, 0.0, -strength)
        mask = bias.unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)   # (1,1,joint,joint) additive

        guidance = (torch.full([1], bs.guidance_scale, device=device, dtype=torch.float32) if tr.config.guidance_embeds else None)
        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        cfg = scheduler.config
        mu = calculate_shift(latents.shape[1], cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)

        region_bias = lambda q, k, n_txt, attn, pl: mask   # precomputed (joint x joint) additive mask, every step
        with flux_intervention(tr):                        # FluxIntervention on all blocks; payload drives it
            for t in timesteps:
                noise_pred = tr(hidden_states=latents, timestep=t.expand(1).to(dtype) / 1000, guidance=guidance,
                                pooled_projections=pooled, encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                                img_ids=img_ids, joint_attention_kwargs={PAYLOAD_KEY: {"bias": region_bias}},
                                return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        lat = FluxPipeline._unpack_latents(latents, H, W, vsf)
        lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
        from diffusers.image_processor import VaeImageProcessor
        image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
