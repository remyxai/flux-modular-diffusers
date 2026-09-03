"""FreeControlBlock — training-free reference-STRUCTURAL control on FLUX.1-dev.

Clean-room implementation of "FreeControl: Efficient, Training-Free Structural Control via One-Step
Attention Extraction" (arXiv:2511.05219, NeurIPS 2025; Lin, Chen, Wu, Zhang, Zhang, Wang, Tang, Wang,
Yang, Yi). No reference code was released — implemented from the paper, GPU-validated by a Colab spike.

Given a REFERENCE image (structure) + a target PROMPT, generate an image that follows the prompt but keeps
the reference's spatial structure. Mechanism (no inversion, no gradient loop):
  1. Latent-Condition Decoupling (LCD): build a noise-free reference latent  x~ = (1 - sigma) * x0.
  2. ONE transformer step at the key timestep t*=661 on x~, capturing the self-attention Query at the last
     N single-stream blocks (op_capture_q).
  3. Generate the target prompt, REPLACING the image-token Query with the captured reference Query in those
     blocks for the first `structure_strength` fraction of steps (K/V + text-Q stay dynamic -> geometry from
     the reference, content from the prompt). Injecting every step over-locks on FLUX; the step cutoff is the
     structure<->content dial (spike: cutoff 0.3 keeps ref layout while the prompt renders).

Bit-exact stock FLUX when `reference_image is None`. FLUX.1-dev is non-commercial; derivative inherits it.
Built on the shared `flux_modular` primitive (vendored flat beside this file).
"""

import numpy as np
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from .flux_modular import (
    pack_latents,
    unpack_latents,
    prepare_latent_image_ids,
    calculate_shift,
    flux_intervention,
    last_single_attn_ids,
    op_capture_q,
    op_replace_q,
    PAYLOAD_KEY,
)

_FLUX = "black-forest-labs/FLUX.1-dev"
_T_STAR = 661   # paper's key timestep for one-step Q extraction


class FreeControlBlock(ModularPipelineBlocks):
    """Training-free structural control on FLUX.1-dev via one-step Query extraction (arXiv:2511.05219)."""

    model_name = "freecontrol-flux"

    @property
    def description(self):
        return ("Transfer a reference image's structure/layout onto a target prompt, training-free on "
                "FLUX.1-dev, by replacing self-attention Queries in the late single blocks. "
                "reference_image=None -> stock FLUX.")

    @property
    def expected_components(self):
        F_ = _FLUX
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
            InputParam("prompt", required=True),               # target prompt (content)
            InputParam("reference_image", default=None),       # structure source; None -> stock FLUX
            InputParam("structure_strength", default=0.3),     # step-cutoff dial: inject Q for the first frac of steps
                                                               #   (spike: 0.3 = ref layout + prompt content; higher = tighter lock)
            InputParam("sigma", default=0.35),                 # LCD scale: x~=(1-sigma)x0 (0.25-0.5)
            InputParam("key_timestep", default=_T_STAR),       # t* for the one-step Q capture
            InputParam("inject_last_n", default=25),           # last-N single blocks to inject
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=6.5),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    def _improc(self):
        if getattr(self, "_vip", None) is None:
            self._vip = VaeImageProcessor(vae_scale_factor=8)
        return self._vip

    def _encode_prompt(self, c, prompt, L, device, dtype):
        te, te2 = c.text_encoder, c.text_encoder_2
        ci = c.tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        pooled = te(ci.to(te.device), output_hidden_states=False).pooler_output.to(device, dtype)
        t5 = c.tokenizer_2([prompt], padding="max_length", max_length=L, truncation=True, return_tensors="pt").input_ids
        emb = te2(t5.to(te2.device))[0].to(device, dtype)
        text_ids = torch.zeros(emb.shape[1], 3, device=device, dtype=dtype)
        return emb, pooled, text_ids

    def _encode_ref_x0(self, c, ref, H, W, device):
        x = self._improc().preprocess(ref, height=H, width=W).to(c.vae.device, c.vae.dtype)
        z = c.vae.encode(x).latent_dist.mode()
        z = (z - c.vae.config.shift_factor) * c.vae.config.scaling_factor
        return pack_latents(z, 1, c.vae.config.latent_channels, 2 * (H // 16), 2 * (W // 16)).to(device)

    def _decode(self, c, latents, H, W, output_type):
        z = unpack_latents(latents, H, W, 8).to(c.vae.device)
        z = z / c.vae.config.scaling_factor + c.vae.config.shift_factor
        img = c.vae.decode(z.to(c.vae.dtype), return_dict=False)[0]
        return self._improc().postprocess(img, output_type=output_type)

    @torch.no_grad()
    def __call__(self, components, state):
        c = components
        bs = self.get_block_state(state)
        tr, vae, sched = c.transformer, c.vae, c.scheduler
        device, dtype = tr.device, tr.dtype
        H, W = int(bs.height), int(bs.width)
        L = int(bs.max_sequence_length)
        steps = int(bs.num_inference_steps)
        lh, lw = 2 * (H // 16), 2 * (W // 16)
        gh, gw = lh // 2, lw // 2
        n_img = gh * gw
        guidance = (torch.full((1,), float(bs.guidance_scale), device=device, dtype=dtype)
                    if tr.config.guidance_embeds else None)
        pe, pooled, text_ids = self._encode_prompt(c, bs.prompt, L, device, dtype)
        img_ids = prepare_latent_image_ids(gh, gw, device, dtype)

        cfg = sched.config
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        mu = calculate_shift(n_img, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        sched.set_timesteps(sigmas=sigmas, mu=mu, device=device)
        timesteps = sched.timesteps

        def vel(latents, t, jkw):
            return tr(hidden_states=latents, timestep=(t.expand(latents.shape[0]) / 1000).to(dtype),
                      guidance=guidance, pooled_projections=pooled, encoder_hidden_states=pe,
                      txt_ids=text_ids, img_ids=img_ids, joint_attention_kwargs=jkw, return_dict=False)[0]

        latents = pack_latents(
            randn_tensor((1, vae.config.latent_channels, lh, lw), generator=bs.generator, device=device, dtype=dtype),
            1, vae.config.latent_channels, lh, lw)

        # ---- no-op path: no reference -> stock FLUX (bit-exact) ----
        if bs.reference_image is None:
            for t in timesteps:
                latents = sched.step(vel(latents, t, None), t, latents, return_dict=False)[0]
            bs.images = self._decode(c, latents, H, W, bs.output_type)
            self.set_block_state(state, bs)
            return components, state

        # ---- FreeControl: LCD one-step Q capture, then cutoff Q-replace generation ----
        ids = last_single_attn_ids(tr, int(bs.inject_last_n))
        xt = ((1.0 - float(bs.sigma)) * self._encode_ref_x0(c, bs.reference_image, H, W, device)).to(dtype)
        cap_pe, cap_pooled, cap_tids = self._encode_prompt(c, "", L, device, dtype)   # empty capture prompt (structure)
        cap_g = (torch.full((1,), 3.5, device=device, dtype=dtype) if tr.config.guidance_embeds else None)
        cap_ts = torch.full((1,), float(bs.key_timestep) / 1000, device=device, dtype=dtype)
        bank = {}
        with flux_intervention(tr):
            tr(hidden_states=xt, timestep=cap_ts, guidance=cap_g, pooled_projections=cap_pooled,
               encoder_hidden_states=cap_pe, txt_ids=cap_tids, img_ids=img_ids,
               joint_attention_kwargs={PAYLOAD_KEY: {"pre_rope": op_capture_q(bank, ids), "n_txt": L}},
               return_dict=False)

        cutoff = int(float(bs.structure_strength) * len(timesteps))   # inject Q for the first `cutoff` steps
        repl = {PAYLOAD_KEY: {"pre_rope": op_replace_q(bank, ids), "n_txt": L}}
        with flux_intervention(tr):
            for i, t in enumerate(timesteps):
                latents = sched.step(vel(latents, t, repl if i < cutoff else None), t, latents, return_dict=False)[0]

        bs.images = self._decode(c, latents, H, W, bs.output_type)
        self.set_block_state(state, bs)
        return components, state
