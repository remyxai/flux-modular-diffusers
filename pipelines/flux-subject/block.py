# "Flux Already Knows" for FLUX — training-free subject-driven generation as a Modular Diffusers block.
#
# Method: LatentUnfold / "Flux Already Knows — Activating Subject-Driven Image Generation without
# Training" (arXiv:2504.11478 — Kang, Fotiadis, Jiang, Yan, Jia, Liu, Chong, Lu). Reference:
# bytedance/LatentUnfold (Apache-2.0). Put a reference subject (an object / product / character)
# into a brand-new scene with NO training, no LoRA and no extra weights: FLUX already has the
# capability, it just has to be laid out so it can use it.
#
# The brief guessed "inject the subject's K/V"; the reference does something simpler and stronger —
# the subject image is REPLICATED as the tiles of a mosaic latent and the scene is generated in the
# remaining tile, so the subject tokens are simply *present in the sequence*. Two additions make it
# work well: (a) cascade attention — the generation tile's attention onto the subject tiles is
# boosted by an average-pooled, re-roped attention map; (b) meta prompting — one [IMAGEk] caption
# per tile, [IMAGE1] being the user's scene. Complements PuLID (trained face weights) and CatVTON
# (garments): this one is arbitrary subjects, zero weights.
#
# Modular-Diffusers adaptation: the reference monkeypatches FluxAttnProcessor2_0.__call__ with
# module-global state (self.aug_att / self.grid_shape / ...). Here the same math is a processor
# subclass threaded through `joint_attention_kwargs['subject_cascade']` (the HRDiT seam — named
# kwarg, no globals, concurrency-safe), installed for the denoise and restored in `finally`.
# `subject_strength=0.0` (or a non-divisible/off layer) falls through to stock FLUX attention —
# bit-exact no-op.
#
# Deviations from the reference, both deliberate and flagged for the GPU reviewer:
#   [VERIFY-encode] the reference encodes the subject tiles with RAW VAE latents but decodes through
#     `(z/scaling)+shift`, so its conditioning latents are off-distribution. Here the tiles are
#     normalized with the VAE shift/scaling factors (the space FLUX.1-dev was trained in), making
#     encode/decode inverses; the smoke notebook round-trips a tile to check exactly this.
#   [VERIFY-idx] the reference indexes tiles with `m * mosaic_shape[0] + n` (rows as the stride);
#     identical for the square grids it ships, wrong for e.g. (2,3). Here the stride is the column
#     count. Equal on every square grid.
#   [VERIFY-attn-memory] the cascade path computes the FULL (n_txt + gh*gw)^2 attention matrix
#     (torch.softmax, then @ value) instead of SDPA, exactly as the reference does — at 3x3/512px
#     that is ~9.7k^2 x 24 heads x bf16 ~ 4.3GB of activation. If that OOMs on the reviewer's card,
#     drop to 384px tiles or grid_shape=(2,2); flagged in the README.
#   RMBG-2.0 background removal is OPT-IN (`remove_background=True`): the brief's headline is
#     "no extra weights", and RMBG-2.0 is itself a non-commercial checkpoint. Pass a pre-cut-out
#     image (or RGBA) to stay weight-free.
#   Meta prompting uses a deterministic template built from `subject_prompt` (the reference's GPT-4o
#     path needs API keys). A `prompt` that already contains "[IMAGE" is used verbatim as an escape
#     hatch for reproducing the reference's exact prompts.
#
# STATUS: follows the PROVEN HRDiT attention seam; the cascade/pooling math is a transcription of
# latent_unfold.py's flux_attn_call2_0 + pool_kq + scaled_dot_product_attention_cascade and is
# gated by the smoke spike + the e2e strength sweep. Publish PRIVATE to remyxai first, then E2E on
# GPU before making public. Authored with AI assistance (Claude), validated by the Remyx AI team;
# method credit to the LatentUnfold authors. Uses FLUX.1-dev (non-commercial license).

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_FLUX = "black-forest-labs/FLUX.1-dev"

# Per-tile captions for the meta prompt. The reference's are written per subject by GPT-4o; these
# are a fixed rotation of view descriptions covering what the tiles actually show.
_META_VIEWS = (
    "captures the subject head-on, showing its overall form and main details",
    "displays the subject in profile, emphasizing its silhouette and edge details",
    "presents a close-up of the subject's texture and material finish",
    "shows the subject from above, highlighting its shape and proportions",
    "captures the subject from behind, revealing its rear details",
    "features the subject's color scheme and finish, a comprehensive view",
    "focuses on the subject's structure and construction details",
    "emphasizes the subject's surface highlights and fine detail",
)


def build_meta_prompt(scene_prompt: str, subject_prompt: Optional[str], num_tiles: int) -> str:
    """The reference's meta prompt: one caption per grid tile, `[IMAGE1]` = the user's scene."""
    if num_tiles <= 1:                       # no reference tiles -> nothing to caption
        return scene_prompt
    subject = (subject_prompt or "the reference subject").strip().rstrip(".")
    parts = [
        f"This set of full-frame photos captures an identical {subject} subject firmly positioned "
        f"in the real scene, highlighting its design and features from various perspectives "
        f"(cinematic, epic, 4K, high quality). [IMAGE1] {scene_prompt.strip()}"
    ]
    parts += [f"[IMAGE{i}]" + _META_VIEWS[(i - 2) % len(_META_VIEWS)] for i in range(2, num_tiles + 1)]
    return " ".join(parts)


def _pool_tokens(x: torch.Tensor, factor: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Average-pool the image token grid of `(B, grid_h*grid_w, heads, D)` by `factor`
    (the reference's `pool_kq`, in plain torch). Returns `(B, (h/f)*(w/f), heads, D)`."""
    b, _, heads, dim = x.shape
    v = x.reshape(b, grid_h, grid_w, heads, dim).permute(0, 3, 4, 1, 2).reshape(b * heads * dim, 1, grid_h, grid_w)
    v = F.avg_pool2d(v, kernel_size=factor, stride=factor)
    nh, nw = v.shape[-2:]
    return v.view(b, heads, dim, nh, nw).permute(0, 1, 3, 4, 2).reshape(b, nh * nw, heads, dim)


class SubjectCascadeFluxAttnProcessor(FluxAttnProcessor):
    """FLUX joint attention with LatentUnfold's cascade attention, driven by
    `joint_attention_kwargs['subject_cascade']`.

    Payload: strength, step, injection_steps, cascade (pooling factors), ropes (one RoPE per factor,
    built on the downsampled id grid), grid (packed-token mosaic rows/cols) and tile (packed-token
    size of one tile). Inactive layers, `strength <= 0`, `step >= injection_steps` and every
    single-stream block (`encoder_hidden_states is None`, as in the reference) take the stock
    FluxAttnProcessor path — a bit-exact no-op.

    Cascade: the image-stream K/Q are average-pooled by each factor, rotated with the matching
    downsampled RoPE, and the pooled attention map is bilinearly upsampled and ADDED (scaled by
    strength/factor/len(cascade)) to the generation tile's attention over the subject tiles —
    up-weighting attention from the region being generated onto the reference-subject tokens
    without touching a single weight.
    """

    active = True

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None, subject_cascade=None):
        if (subject_cascade is None or not self.active or encoder_hidden_states is None
                or subject_cascade["strength"] <= 0.0
                or subject_cascade["step"] >= subject_cascade["injection_steps"]):
            return super().__call__(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

        sc = subject_cascade
        gh, gw = sc["grid"]
        th, tw = sc["tile"]
        n_txt = encoder_hidden_states.shape[1]
        batch = encoder_hidden_states.shape[0]
        scale = (hidden_states.shape[-1] // attn.heads) ** -0.5

        # projections, pre-RoPE. hidden_states = the image stream, encoder_hidden_states = text.
        query, key, value, e_q, e_k, e_v = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        q = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))                     # (B, S_img, h, D)
        k = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        v = value.unflatten(-1, (attn.heads, -1))
        eq = attn.norm_added_q(e_q.unflatten(-1, (attn.heads, -1)))
        ek = attn.norm_added_k(e_k.unflatten(-1, (attn.heads, -1)))
        ev = e_v.unflatten(-1, (attn.heads, -1))

        def _prob(qf, kf, rope):
            qf = apply_rotary_emb(qf, rope, sequence_dim=1).transpose(1, 2)        # (B, h, S, D)
            kf = apply_rotary_emb(kf, rope, sequence_dim=1).transpose(1, 2)
            return torch.softmax(qf @ kf.transpose(-2, -1) * scale, dim=-1)

        attn_prob = _prob(torch.cat([eq, q], dim=1), torch.cat([ek, k], dim=1), image_rotary_emb)
        # attn_prob is the FULL joint (n_txt + n_img)^2 matrix; the cascade up-weights only the
        # image->image block. That block is a strided (non-contiguous) slice, so it can't be
        # `.view`-reshaped to the (gh, gw, gh, gw) grid in place. Work on a contiguous copy, then
        # fold it back into attn_prob below so the output projection sees the modified probabilities.
        img_prob = attn_prob[:, :, n_txt:, n_txt:].contiguous()
        prob_grid = img_prob.view(batch, attn.heads, gh, gw, gh, gw)
        for i, factor in enumerate(sc["cascade"]):
            pq = _pool_tokens(q, factor, gh, gw)
            pk = _pool_tokens(k, factor, gh, gw)
            c_prob = _prob(torch.cat([eq, pq], dim=1), torch.cat([ek, pk], dim=1), sc["ropes"][i])
            # NOTE[VERIFY-upsample]: transcribed verbatim — the pooled image-image block is upsampled
            # in FLATTENED token index ((h/f)*(w/f) -> h*w), not on the true spatial grid.
            up = F.interpolate(c_prob[:, :, n_txt:, n_txt:], size=(gh * gw, gh * gw),
                               mode="bilinear", align_corners=False).view(batch, attn.heads, gh, gw, gh, gw)
            weight = sc["strength"] / factor / len(sc["cascade"])
            prob_grid[:, :, :th, :tw, :th, tw:] += up[:, :, :th, :tw, :th, tw:] * weight
            prob_grid[:, :, :th, :tw, th:, :] += up[:, :, :th, :tw, th:, :] * weight

        attn_prob[:, :, n_txt:, n_txt:] = img_prob      # fold the cascade back into the joint matrix
        del img_prob

        out = (attn_prob @ torch.cat([ev, v], dim=1).transpose(1, 2)).transpose(1, 2).flatten(2, 3).to(q.dtype)
        enc, out = out.split_with_sizes([n_txt, out.shape[1] - n_txt], dim=1)
        hidden_states = attn.to_out[1](attn.to_out[0](out.contiguous()))
        return hidden_states, attn.to_add_out(enc.contiguous())


class FluxSubjectBlock(ModularPipelineBlocks):
    """Training-free subject-driven generation: replicate a reference subject as the tiles of a
    mosaic latent, denoise the scene into the remaining tile with cascade attention up-weighting
    the subject tiles, decode just that tile. No training, no LoRA, no extra weights."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FLUX = _FLUX
    _RMBG = "briaai/RMBG-2.0"          # opt-in subject cutout (non-commercial checkpoint)

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
            InputParam("subject_image", required=True),   # reference subject: PIL / numpy / path (list = multi-view)
            InputParam("prompt", required=True),          # the NEW scene ([IMAGE1]); verbatim if it holds "[IMAGE"
            InputParam("subject_prompt", default=None),   # one short description of the subject (meta prompt)
            InputParam("prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=512),            # size of the OUTPUT tile (reference's image_shape)
            InputParam("width", default=512),
            InputParam("grid_shape", default=(3, 3)),     # mosaic rows/cols; (1,1) = no reference tiles = stock FLUX
            InputParam("num_inference_steps", default=28),
            InputParam("guidance_scale", default=7.0),    # the reference's value (not FLUX's usual 3.5)
            InputParam("subject_strength", default=0.05), # cascade weight (reference's aug_att); 0.0 = pure mosaic
            InputParam("cascade", default=(2, 3)),        # pooling factors; () disables cascade attention
            InputParam("injection_steps", default=14),    # apply cascade attention only over the first N steps
            InputParam("cascade_start_frac", default=0.0),# skip the first frac of dual-stream layers (0.0 = reference)
            InputParam("remove_background", default=False),  # True = opt-in RMBG-2.0 cutout (extra weights)
            InputParam("seed", default=0),                # tile-pick RNG (reference's `seed`)
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
        img = Image.open(x) if isinstance(x, str) else x
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(x))
        if img.mode in ("RGBA", "LA"):                    # a pre-cut-out subject -> white background
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            return bg
        return img.convert("RGB")

    def _remove_background(self, img, device):
        """The reference's utils/seg_utils.rmbg (RMBG-2.0, lazy-loaded once), cutout onto white."""
        if getattr(self, "_rmbg", None) is None:
            from transformers import AutoModelForImageSegmentation
            self._rmbg = AutoModelForImageSegmentation.from_pretrained(
                self._RMBG, trust_remote_code=True).to(device).eval()
        from PIL import Image
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = torch.from_numpy(np.asarray(img.resize((1024, 1024)))).permute(2, 0, 1).unsqueeze(0)
        x = (x.to(device=device, dtype=torch.float32) / 255.0 - mean) / std
        with torch.no_grad():
            pred = self._rmbg(x)[-1].sigmoid()[0, 0].cpu()
        mask = Image.fromarray((pred.numpy() * 255).astype(np.uint8)).resize(img.size)
        out = Image.new("RGB", img.size, (255, 255, 255))
        out.paste(img.convert("RGB"), (0, 0), mask)
        return out

    def _prepare_subject(self, x, size, remove_background, device):
        """Reference's utils/image_utils.resize: optional cutout -> centre-crop square -> tile size."""
        img = self._to_pil(x)
        if remove_background:
            img = self._remove_background(img, device)
        w, h = img.size
        m = min(w, h)
        return img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize(size)

    @staticmethod
    def _white_tile(width, height):
        from PIL import Image
        return Image.new("RGB", (width, height), (255, 255, 255))

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)          # 8
        quant = vsf * 2                                               # 16
        num_channels_latents = tr.config.in_channels // 4            # 16
        guidance_embeds = tr.config.guidance_embeds

        # --- geometry: one output tile of (height, width), replicated into a rows x cols mosaic ---
        height, width = int(bs.height), int(bs.width)
        if height % quant or width % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        rows, cols = (int(bs.grid_shape[0]), int(bs.grid_shape[1]))
        if rows < 1 or cols < 1:
            raise ValueError(f"grid_shape must be >= (1, 1), got {bs.grid_shape}.")
        lh, lw = 2 * (height // quant), 2 * (width // quant)          # latent size of ONE tile
        mh, mw = lh * rows, lw * cols                                 # latent size of the mosaic
        gh, gw = mh // 2, mw // 2                                     # packed-token mosaic grid
        th, tw = lh // 2, lw // 2                                     # packed tokens of ONE tile

        cascade = tuple(int(s) for s in (bs.cascade or ()))
        strength, inj_steps = float(bs.subject_strength), int(bs.injection_steps)
        if strength > 0.0 and cascade and any(gh % s or gw % s for s in cascade):
            raise ValueError(f"every cascade factor must divide the token grid ({gh}x{gw}), got {cascade}.")

        # --- subject -> mosaic latent. Tile (0,0) is the generation region, the rest are references ---
        subs = bs.subject_image if isinstance(bs.subject_image, (list, tuple)) else [bs.subject_image]
        subjects = [self._prepare_subject(s, (width, height), bool(bs.remove_background), device) for s in subs]
        img_proc = VaeImageProcessor(vae_scale_factor=vsf)

        def _encode(img):
            px = img_proc.preprocess(img, height=height, width=width).to(device=device, dtype=vae.dtype)
            z = vae.encode(px).latent_dist.mode()                    # argmax, as in the reference
            return ((z - vae.config.shift_factor) * vae.config.scaling_factor).to(dtype)   # [VERIFY-encode]

        refs = [_encode(im) for im in subjects][: max(rows * cols - 1, 0)]
        white = _encode(self._white_tile(width, height))
        mosaic = white.repeat(1, 1, rows, cols)
        rng = random.Random(int(bs.seed))                            # local RNG: no global mutation
        for m in range(rows):
            for n in range(cols):
                if m == 0 and n == 0:
                    continue
                idx = m * cols + n - 1                               # [VERIFY-idx] row-major stride
                idx = idx if idx < len(refs) else rng.randint(0, len(refs) - 1)
                if refs:
                    mosaic[:, :, m * lh:(m + 1) * lh, n * lw:(n + 1) * lw] = refs[idx]
        mask = torch.zeros((1, 1, mh, mw), device=device, dtype=dtype)
        mask[:, :, :lh, :lw] = 1.0                                   # 1 = denoise (the generation tile)

        # --- meta prompt (verbatim if the caller already wrote [IMAGEk] captions) ---
        prompt = bs.prompt if "[IMAGE" in str(bs.prompt) else build_meta_prompt(
            bs.prompt, bs.subject_prompt, rows * cols)
        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = (torch.full([1], float(bs.guidance_scale), device=device, dtype=torch.float32)
                    if guidance_embeds else None)

        # --- schedule (mu from the TILE seq len, as in the reference) ---
        nsteps = int(bs.num_inference_steps)
        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        cfg = scheduler.config
        mu = calculate_shift((height // (vsf * 2)) * (width // (vsf * 2)), cfg.get("base_image_seq_len", 256),
                             cfg.get("max_image_seq_len", 4096), cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)

        # --- pack the mosaic; seed the latents with the noised mosaic (strength = 1.0) ---
        def pack(x):
            return FluxPipeline._pack_latents(x, 1, num_channels_latents, mh, mw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, gh, gw, device, dtype)
        noise = pack(randn_tensor((1, num_channels_latents, mh, mw), generator=bs.generator,
                                  device=device, dtype=dtype))
        mosaic_p, mask_p = pack(mosaic), pack(mask.repeat(1, num_channels_latents, 1, 1))
        latents = scheduler.scale_noise(mosaic_p, timesteps[:1], noise).to(dtype)

        # --- one RoPE per cascade factor, built on the downsampled id grid (once, not per step) ---
        ropes: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if strength > 0.0 and cascade:
            grid_h, grid_w = int(img_ids[:, 1].max().item()) + 1, int(img_ids[:, 2].max().item()) + 1
            for s in cascade:
                ids = torch.tensor([[0.0, float(i), float(j)] for i in range(grid_h // s)
                                    for j in range(grid_w // s)], device=device, dtype=dtype)
                ropes.append(tr.pos_embed(torch.cat([text_ids, ids], dim=0)))

        def _install(start_frac):
            """Swap in the cascade processor; `start_frac` disarms the early dual-stream layers."""
            orig, start = {}, int(float(start_frac) * len(tr.transformer_blocks))
            for i, blk in enumerate(tr.transformer_blocks):
                orig[f"transformer_blocks.{i}.attn"] = blk.attn.processor
                proc = SubjectCascadeFluxAttnProcessor()
                proc.active = i >= start
                blk.attn.processor = proc
            # The single-stream blocks never cascade (encoder_hidden_states is None), but they still
            # need a processor that ACCEPTS the kwarg: blocks forward **joint_attention_kwargs into
            # attn(), and a stock FluxAttnProcessor would raise TypeError on 'subject_cascade'. It
            # falls straight through to super(), so those blocks stay bit-exact.
            for i, blk in enumerate(tr.single_transformer_blocks):
                orig[f"single_transformer_blocks.{i}.attn"] = blk.attn.processor
                blk.attn.processor = SubjectCascadeFluxAttnProcessor()
            return orig

        orig = _install(bs.cascade_start_frac)
        try:
            for i, t in enumerate(timesteps):
                jak = None
                # (1,1) has no reference tiles, so there is nothing to up-weight: leave jak None and
                # every processor takes the stock path — genuinely bit-exact stock FLUX, and it skips
                # the full-matrix attention the cascade path needs.
                if strength > 0.0 and cascade and rows * cols > 1:
                    jak = {"subject_cascade": {"strength": strength, "step": i, "injection_steps": inj_steps,
                                               "cascade": cascade, "ropes": ropes, "grid": (gh, gw),
                                               "tile": (th, tw)}}
                noise_pred = tr(hidden_states=latents, timestep=t.expand(latents.shape[0]).to(dtype) / 1000,
                                guidance=guidance, pooled_projections=pooled,
                                encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=img_ids,
                                joint_attention_kwargs=jak, return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                # hold the reference tiles at the right noise level; only tile (0,0) actually denoises
                held = mosaic_p if i == len(timesteps) - 1 else scheduler.scale_noise(
                    mosaic_p, timesteps[i + 1:i + 2], noise).to(dtype)
                latents = (1 - mask_p) * held + mask_p * latents
        finally:
            for name, mod in tr.named_modules():
                if name in orig:
                    mod.processor = orig[name]

        # --- decode just the generation tile ---
        if bs.output_type == "latent":
            image = latents
        else:
            lat = FluxPipeline._unpack_latents(latents, mh * vsf, mw * vsf, vsf)   # unpack expects PIXEL dims
            lat = ((lat / vae.config.scaling_factor) + vae.config.shift_factor)[:, :, :lh, :lw]
            image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
            image = img_proc.postprocess(image, output_type=bs.output_type)

        bs.images = image if isinstance(image, list) else [image]
        self.set_block_state(state, bs)
        return components, state
