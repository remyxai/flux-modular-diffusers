"""AppearanceTransferBlock — training-free reference appearance/texture transfer on FLUX.1-Depth.

Clean-room implementation of "A Training-Free Framework for High-Fidelity Appearance Transfer via
Diffusion Transformers" (arXiv:2603.26767; Gu, Y. Wang, Wu, Ma, Q. Wang, L. Wang, Yi). No reference
code was read or copied — implemented from the paper, validated by a Colab A100 spike (2026-09).

Given a source image (structure to keep) and a reference image (appearance to transfer), the block:
  1. inverts the source along the FLUX.1-Depth flow (matched RK2 solver) -> a content-rich trajectory,
  2. encodes the reference through a **mask-weighted Redux** (suppresses reference shape, keeps texture)
     as a global appearance embedding, and captures its image-token K/V as a local appearance library,
  3. generates by replaying the source trajectory for the first `blend_k` fraction of steps (structure
     lock), then free synthesis conditioned on depth + Redux with the reference K/V concatenated into the
     attention at the first-2 / last-2 blocks of both streams (appearance).

Bit-exact identity when `reference_image is None` (returns the source unchanged).
FLUX.1-Depth-dev + FLUX.1-Redux-dev are **non-commercial**; this derivative inherits that license.
"""

import numpy as np
import torch
import torch.nn.functional as F

from diffusers.modular_pipelines import (
    ModularPipelineBlocks,
    ComponentSpec,
    InputParam,
    OutputParam,
)
from diffusers.image_processor import VaeImageProcessor

# Shared FLUX-modular abstractions (vendored flat beside this file for trust_remote_code). Canonical
# source: flux_modular/ in the monorepo. Replaces the per-block FLUX plumbing + hand-rolled attention.
from .flux_modular import (
    pack_latents,
    unpack_latents,
    prepare_latent_image_ids,
    calculate_shift,
    flux_intervention,
    edge_blocks,
    op_append,
    op_capture_image_kv,
    PAYLOAD_KEY,
)


# --------------------------------------------------------------------------- the block
class AppearanceTransferBlock(ModularPipelineBlocks):
    """Training-free reference appearance transfer on FLUX.1-Depth (arXiv:2603.26767)."""

    model_name = "appearance-transfer-flux"

    @property
    def description(self):
        return (
            "Transfer a reference image's appearance/texture onto a source image while preserving the "
            "source geometry, training-free on FLUX.1-Depth (+ mask-weighted Redux). "
            "reference_image=None -> returns the source unchanged."
        )

    @property
    def expected_components(self):
        depth = "black-forest-labs/FLUX.1-Depth-dev"
        redux = "black-forest-labs/FLUX.1-Redux-dev"
        from transformers import (
            CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast,
            SiglipVisionModel, SiglipImageProcessor,
        )
        from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
        return [
            ComponentSpec("text_encoder", CLIPTextModel, pretrained_model_name_or_path=depth, subfolder="text_encoder"),
            ComponentSpec("tokenizer", CLIPTokenizer, pretrained_model_name_or_path=depth, subfolder="tokenizer"),
            ComponentSpec("text_encoder_2", T5EncoderModel, pretrained_model_name_or_path=depth, subfolder="text_encoder_2"),
            ComponentSpec("tokenizer_2", T5TokenizerFast, pretrained_model_name_or_path=depth, subfolder="tokenizer_2"),
            ComponentSpec("transformer", FluxTransformer2DModel, pretrained_model_name_or_path=depth, subfolder="transformer"),
            ComponentSpec("vae", AutoencoderKL, pretrained_model_name_or_path=depth, subfolder="vae"),
            ComponentSpec("scheduler", FlowMatchEulerDiscreteScheduler, pretrained_model_name_or_path=depth, subfolder="scheduler"),
            ComponentSpec("image_encoder", SiglipVisionModel, pretrained_model_name_or_path=redux, subfolder="image_encoder"),
            ComponentSpec("feature_extractor", SiglipImageProcessor, pretrained_model_name_or_path=redux, subfolder="feature_extractor"),
            ComponentSpec("image_embedder", ReduxImageEncoder, pretrained_model_name_or_path=redux, subfolder="image_embedder"),
        ]

    @property
    def inputs(self):
        return [
            InputParam("source_image", required=True),        # PIL.Image — structure to keep
            InputParam("reference_image", default=None),      # PIL.Image — appearance to transfer; None -> identity
            InputParam("reference_mask", default=None),       # HxW array/PIL in [0,1]; None -> unweighted Redux
            InputParam("source_mask", default=None),          # reserved (spatial transfer region); None -> whole image
            InputParam("blend_k", default=0.25),              # fraction of steps to replay the source (structure lock);
                                                              # e2e-tuned: lower(~0.2)=more appearance, raise(0.3-0.4)=more structure
            InputParam("num_inference_steps", default=50),
            InputParam("guidance_scale", default=10.0),       # FLUX.1-Depth generation guidance
            InputParam("invert_guidance", default=1.0),       # RF-inversion guidance (spike: 1.0 best)
            InputParam("use_kv_injection", default=True),     # secondary appearance channel
            InputParam("redux_mask_floor", default=0.1),      # background patch weight floor for mask-weighted Redux
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("generator", default=None),
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_outputs(self):
        return [OutputParam("images")]

    # ---- lazy singletons (depth estimator + image processor) ----
    def _improc(self):
        if getattr(self, "_vip", None) is None:
            self._vip = VaeImageProcessor(vae_scale_factor=8)
        return self._vip

    def _depth(self, device):
        if getattr(self, "_dep", None) is None:
            from transformers import pipeline as hf_pipeline
            idx = device.index if getattr(device, "index", None) is not None else 0
            self._dep = hf_pipeline("depth-estimation",
                                    model="depth-anything/Depth-Anything-V2-Small-hf",
                                    device=idx if device.type == "cuda" else -1)
        return self._dep

    # ---- helpers ----
    def _encode_empty_prompt(self, c, tdev, dtype):
        te = c.text_encoder.device
        tok = c.tokenizer([""], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(te)
        pooled = c.text_encoder(tok.input_ids, output_hidden_states=False).pooler_output
        te2 = c.text_encoder_2.device
        tok2 = c.tokenizer_2([""], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(te2)
        prompt_embeds = c.text_encoder_2(tok2.input_ids)[0]
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=tdev, dtype=dtype)
        return prompt_embeds.to(tdev, dtype), pooled.to(tdev, dtype), text_ids

    def _redux_embeds(self, c, ref_img, ref_mask, tdev, dtype, floor):
        ie = c.image_encoder.device
        fe = c.feature_extractor(images=ref_img, return_tensors="pt").to(ie)
        siglip = c.image_encoder(pixel_values=fe.pixel_values.to(c.image_encoder.dtype)).last_hidden_state
        if ref_mask is not None:  # mask-weighted: down-weight background patches -> texture, not shape
            p = siglip.shape[1]
            g = int(round(p ** 0.5))
            if g * g == p:
                m = torch.as_tensor(np.asarray(ref_mask), dtype=torch.float32)
                if m.ndim == 3:
                    m = m.mean(-1)
                m = m[None, None] / (m.max() + 1e-6)
                m = F.interpolate(m, size=(g, g), mode="area").reshape(1, p, 1).to(ie, siglip.dtype)
                siglip = siglip * (floor + (1.0 - floor) * m)
        emb_dev = next(c.image_embedder.parameters()).device
        return c.image_embedder(siglip.to(emb_dev)).image_embeds.to(tdev, dtype)

    def _encode_image(self, c, img, H, W, tdev):
        vd = c.vae.device
        x = self._improc().preprocess(img, height=H, width=W).to(vd, c.vae.dtype)
        z = c.vae.encode(x).latent_dist.mode()   # deterministic; avoids generator/device coupling
        z = (z - c.vae.config.shift_factor) * c.vae.config.scaling_factor
        packed = pack_latents(z, 1, c.vae.config.latent_channels, 2 * (H // 16), 2 * (W // 16))
        return packed.to(tdev)

    def _control_latent(self, c, img, H, W, tdev):
        depth = self._depth(tdev)(img)["depth"].convert("RGB").resize((W, H))
        return self._encode_image(c, depth, H, W, tdev)

    def _decode(self, c, packed, H, W, output_type):
        vd = c.vae.device
        z = unpack_latents(packed, H, W, 8).to(vd)
        z = z / c.vae.config.scaling_factor + c.vae.config.shift_factor
        img = c.vae.decode(z.to(c.vae.dtype), return_dict=False)[0]
        return self._improc().postprocess(img, output_type=output_type)

    @torch.no_grad()
    def __call__(self, components, state):
        c = components
        bs = self.get_block_state(state)
        tdev = c.transformer.device            # compute device for the denoise loop
        dtype = c.transformer.dtype
        H, W = int(bs.height), int(bs.width)

        # identity fast-path
        if bs.reference_image is None:
            bs.images = [bs.source_image] if bs.output_type == "pil" else bs.source_image
            self.set_block_state(state, bs)
            return components, state

        prompt_embeds, pooled, text_ids = self._encode_empty_prompt(c, tdev, dtype)

        # combined generation conditioning = [mask-weighted Redux appearance ; empty prompt]
        redux = self._redux_embeds(c, bs.reference_image, bs.reference_mask, tdev, dtype, bs.redux_mask_floor)
        enc_gen = torch.cat([redux, prompt_embeds], dim=1)
        tids_gen = torch.zeros(enc_gen.shape[1], 3, device=tdev, dtype=dtype)

        # depth control latents (source drives structure; reference used only for its K/V capture pass)
        ctrl_src = self._control_latent(c, bs.source_image, H, W, tdev)
        ctrl_ref = self._control_latent(c, bs.reference_image, H, W, tdev)
        n_img = ctrl_src.shape[1]
        img_ids = prepare_latent_image_ids(H // 16, W // 16, tdev, dtype)

        # sigma schedule (mu-shifted), matched RK2 both directions
        sig_lin = np.linspace(1.0, 1.0 / bs.num_inference_steps, bs.num_inference_steps)
        mu = calculate_shift(n_img, c.scheduler.config.base_image_seq_len, c.scheduler.config.max_image_seq_len,
                             c.scheduler.config.base_shift, c.scheduler.config.max_shift)
        c.scheduler.set_timesteps(sigmas=sig_lin, mu=mu, device=tdev)
        sigs = [float(s) for s in c.scheduler.sigmas.cpu().numpy()]   # descending, ends 0.0
        N = len(sigs) - 1

        def vel(latent, sigma, gval, enc, tids, ctrl, jkw=None):
            hs = torch.cat([latent, ctrl], dim=2)
            ts = torch.full((1,), float(sigma), device=tdev, dtype=dtype)
            gd = torch.full((1,), float(gval), device=tdev, dtype=dtype)
            return c.transformer(hidden_states=hs, timestep=ts, guidance=gd, pooled_projections=pooled,
                                 encoder_hidden_states=enc, txt_ids=tids, img_ids=img_ids,
                                 joint_attention_kwargs=jkw, return_dict=False)[0]

        # Install one FluxIntervention on the edge blocks for the whole call; drive per-pass via the
        # joint_attention_kwargs payload. No payload -> stock (bit-exact), so inversion is unaffected.
        bank = {}
        cap_pl = {PAYLOAD_KEY: {"post_rope": op_capture_image_kv(bank, n_img)}}   # record ref image K/V
        inj_pl = {PAYLOAD_KEY: {"post_rope": op_append(bank)}}                    # append it onto source K/V

        with flux_intervention(c.transformer, edge_blocks(c.transformer, n=2)):
            # (1) capture the reference's image-token K/V at a structured (low-sigma) state
            xref = self._encode_image(c, bs.reference_image, H, W, tdev)
            _ = vel(xref, sigs[-2], bs.invert_guidance, prompt_embeds, text_ids, ctrl_ref, jkw=cap_pl)

            # (2) invert the source (RK2 ascending) into a full trajectory (no payload -> stock attention)
            x0 = self._encode_image(c, bs.source_image, H, W, tdev)
            asc = sigs[::-1]
            traj = [x0.clone()]
            x = x0.clone()
            for i in range(N):
                s, s1 = asc[i], asc[i + 1]
                dt = s1 - s
                v1 = vel(x, s, bs.invert_guidance, prompt_embeds, text_ids, ctrl_src)
                v2 = vel(x + 0.5 * dt * v1, s + 0.5 * dt, bs.invert_guidance, prompt_embeds, text_ids, ctrl_src)
                x = x + dt * v2
                traj.append(x.clone())

            # (3) generate: replay to blend_k (structure lock), then free RK2 + Redux (+ K/V append)
            k_idx = int(float(bs.blend_k) * N)
            gpl = inj_pl if bs.use_kv_injection else None
            x = traj[N].clone()
            for i in range(N):
                if i < k_idx:
                    x = traj[N - i - 1].clone()
                    continue
                s, s1 = sigs[i], sigs[i + 1]
                dt = s1 - s
                v1 = vel(x, s, bs.guidance_scale, enc_gen, tids_gen, ctrl_src, jkw=gpl)
                v2 = vel(x + 0.5 * dt * v1, s + 0.5 * dt, bs.guidance_scale, enc_gen, tids_gen, ctrl_src, jkw=gpl)
                x = x + dt * v2

        bs.images = self._decode(c, x, H, W, bs.output_type)
        self.set_block_state(state, bs)
        return components, state
