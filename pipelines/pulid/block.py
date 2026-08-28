# PuLID for FLUX — training-free identity personalization as a Modular Diffusers custom block.
#
# Method: PuLID (https://github.com/ToTheBeginning/PuLID, Apache-2.0) — an ID embedding from a
# reference face (InsightFace ArcFace + facexlib align/parse + EVA-CLIP -> an IDFormer resampler)
# is injected as an additive cross-attention residual into the image stream after every 2nd double
# block and every 4th single block of FLUX. Weights: guozinan/PuLID (Apache-2.0). EVA-CLIP and
# InsightFace/facexlib are the ID-encoder dependencies.
#
# This file is the Modular-Diffusers adaptation: the injection is applied via forward hooks on
# diffusers' FluxTransformer2DModel (no changes to the base weights), and the vendored `eva_clip`
# package is fetched at runtime + added to sys.path (subdirectory packages are not resolved by the
# trust_remote_code loader, which fetches only flat sibling .py files).
#
# Authored with AI assistance (Claude) and validated by the Remyx AI team; all method credit to the
# original PuLID / EVA-CLIP / InsightFace authors.

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from diffusers.utils.torch_utils import randn_tensor
from diffusers import FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.modular_pipelines import ModularPipelineBlocks, ComponentSpec, InputParam, OutputParam
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

_REPO_ID = "remyxai/pulid-flux-modular"          # this repo — runtime source of the vendored eva_clip/
_PULID_REPO = "guozinan/PuLID"
_PULID_FILE = "pulid_flux_v0.9.1.safetensors"

# --------------------------------------------------------------------------------------
# runtime bootstrap of the vendored eva_clip package (fetched as data, not via relative import)
# --------------------------------------------------------------------------------------
_EVA_READY = False


def _ensure_eva_clip():
    global _EVA_READY
    if _EVA_READY:
        return
    from huggingface_hub import snapshot_download
    local = snapshot_download(_REPO_ID, allow_patterns=["eva_clip/*"])
    if local not in sys.path:
        sys.path.insert(0, local)
    _EVA_READY = True


# --------------------------------------------------------------------------------------
# vendored ID-encoder modules  (ToTheBeginning/PuLID/pulid/encoders_transformer.py, Apache-2.0)
# --------------------------------------------------------------------------------------
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttentionCA(nn.Module):
    def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """x: image features (b, n1, D) as KV; latents: query (b, n2, D)."""
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, seq_len, _ = latents.shape
        q = self.to_q(latents)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, seq_len, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
        return self.to_out(out)


class IDFormer(nn.Module):
    """Perceiver-resampler ID encoder: arcface id + query tokens as latents, cross-attending
    multi-scale EVA-CLIP features (each scale -> two IDFormer layers)."""

    def __init__(self, dim=1024, depth=10, dim_head=64, heads=16, num_id_token=5,
                 num_queries=32, output_dim=2048, ff_mult=4):
        super().__init__()
        self.num_id_token = num_id_token
        self.dim = dim
        self.num_queries = num_queries
        assert depth % 5 == 0
        self.depth = depth // 5
        scale = dim ** -0.5
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
        self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ]))
        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(
                nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                nn.Linear(1024, dim),
            ))
        self.id_embedding_mapping = nn.Sequential(
            nn.Linear(1280, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
            nn.Linear(1024, dim * num_id_token),
        )

    def forward(self, x, y):
        latents = self.latents.repeat(x.size(0), 1, 1)
        num_duotu = x.shape[1] if x.ndim == 3 else 1
        x = self.id_embedding_mapping(x)
        x = x.reshape(-1, self.num_id_token * num_duotu, self.dim)
        latents = torch.cat((latents, x), dim=1)
        for i in range(5):
            vit_feature = getattr(self, f'mapping_{i}')(y[i])
            ctx_feature = torch.cat((x, vit_feature), dim=1)
            for attn, ff in self.layers[i * self.depth: (i + 1) * self.depth]:
                latents = attn(ctx_feature, latents) + latents
                latents = ff(latents) + latents
        latents = latents[:, :self.num_queries]
        latents = latents @ self.proj_out
        return latents


# --------------------------------------------------------------------------------------
# vendored image utils  (ToTheBeginning/PuLID/pulid/utils.py, Apache-2.0)
# --------------------------------------------------------------------------------------
def resize_numpy_image_long(image, resize_long_edge=1024):
    import cv2
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    image = cv2.resize(image, (int(w * k), int(h * k)), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(img, bgr2rgb=True, float32=True):
    import cv2
    if img.shape[2] == 3 and bgr2rgb:
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    if float32:
        img = img.float()
    return img


# --------------------------------------------------------------------------------------
# ID encoder (mirrors PuLIDPipeline.get_id_embedding; face stack built lazily)
# --------------------------------------------------------------------------------------
class _PuLIDEncoder(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.device = device
        self.weight_dtype = dtype
        _ensure_eva_clip()
        import insightface  # noqa
        from insightface.app import FaceAnalysis
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from huggingface_hub import snapshot_download
        from eva_clip import create_model_and_transforms
        from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

        self.pulid_encoder = IDFormer().to(device, dtype)
        self.pulid_ca = nn.ModuleList(
            [PerceiverAttentionCA().to(device, dtype) for _ in range(20)]
        )

        self.face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model='retinaface_resnet50', save_ext='png', device=device)
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=device)

        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        self.clip_vision_model = model.visual.to(device, dtype)
        mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        self.eva_transform_mean = mean if isinstance(mean, (list, tuple)) else (mean,) * 3
        self.eva_transform_std = std if isinstance(std, (list, tuple)) else (std,) * 3

        snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if str(device).startswith('cuda') \
            else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name='antelopev2', root='.', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx', providers=providers)
        self.handler_ante.prepare(ctx_id=0)

        self._load_weights()

    def _load_weights(self):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        ckpt = hf_hub_download(_PULID_REPO, _PULID_FILE)
        sd = load_file(ckpt)
        buckets = {}
        for k, v in sd.items():
            mod = k.split('.')[0]
            buckets.setdefault(mod, {})[k[len(mod) + 1:]] = v
        for mod, d in buckets.items():
            getattr(self, mod).load_state_dict(d, strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        return x.repeat(1, 3, 1, 1)

    @torch.no_grad()
    def get_id_embedding(self, image):
        """image: numpy RGB uint8 [0,255]. Returns id embedding (1, 32, 2048)."""
        import cv2
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import normalize, resize

        self.face_helper.clean_all()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face_info = self.app.get(image_bgr)
        if len(face_info) > 0:
            face_info = sorted(
                face_info,
                key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
            id_ante_embedding = face_info['embedding']
        else:
            id_ante_embedding = None

        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib failed to align a face in the reference image.')
        align_face = self.face_helper.cropped_faces[0]

        if id_ante_embedding is None:
            id_ante_embedding = self.handler_ante.get_feat(align_face)
        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        inp = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        inp = inp.to(self.device)
        parsing_out = self.face_helper.face_parse(
            normalize(inp, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white = torch.ones_like(inp)
        face_features_image = torch.where(bg, white, self.to_gray(inp))

        face_features_image = resize(
            face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False)
        id_cond_vit = torch.div(id_cond_vit, torch.norm(id_cond_vit, 2, 1, True))

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)
        return self.pulid_encoder(id_cond, id_vit_hidden)


# --------------------------------------------------------------------------------------
# injection: forward hooks on FluxTransformer2DModel (uniform for double + single blocks)
# --------------------------------------------------------------------------------------
def _install_pulid(transformer, encoder, id_embedding, id_weight,
                   double_interval=2, single_interval=4):
    pulid_ca = encoder.pulid_ca
    dtype = transformer.dtype
    ide = id_embedding.to(device=transformer.device, dtype=dtype)
    ctr = {"k": 0}

    def _reset(module, args, kwargs):
        ctr["k"] = 0
        return None

    def _hook(module, inp, out):
        enc, hid = out                       # (encoder_hidden_states, image hidden_states)
        add = id_weight * pulid_ca[ctr["k"]](ide, hid.to(dtype))
        ctr["k"] += 1
        return (enc, hid + add.to(hid.dtype))

    handles = [transformer.register_forward_pre_hook(_reset, with_kwargs=True)]
    for i, blk in enumerate(transformer.transformer_blocks):
        if i % double_interval == 0:
            handles.append(blk.register_forward_hook(_hook))
    for i, blk in enumerate(transformer.single_transformer_blocks):
        if i % single_interval == 0:
            handles.append(blk.register_forward_hook(_hook))
    return handles


# --------------------------------------------------------------------------------------
# the modular block
# --------------------------------------------------------------------------------------
class PuLIDFluxBlock(ModularPipelineBlocks):
    """Training-free identity personalization: builds a PuLID id embedding from a reference face and
    injects it as a cross-attention residual into FLUX during a single-pass denoise (fake-CFG)."""

    _requirements = {"diffusers": ">=0.40.0", "torch": ">=2.4.0"}
    _FLUX = "black-forest-labs/FLUX.1-dev"

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
            InputParam("id_image", required=True),      # reference face: PIL / numpy RGB / path
            InputParam("id_weight", default=1.0),        # 0..3 (PuLID recommends ~1.0)
            InputParam("prompt_2", default=None),
            InputParam("max_sequence_length", default=512),
            InputParam("height", default=1024),
            InputParam("width", default=1024),
            InputParam("num_inference_steps", default=20),
            InputParam("guidance_scale", default=4.0),
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

    def _get_encoder(self, device, dtype):
        if getattr(self, "_encoder", None) is None:
            self._encoder = _PuLIDEncoder(device, dtype)
        return self._encoder

    @staticmethod
    def _to_numpy_rgb(img):
        if isinstance(img, str):
            from PIL import Image
            return np.array(Image.open(img).convert("RGB"))
        try:
            from PIL import Image
            if isinstance(img, Image.Image):
                return np.array(img.convert("RGB"))
        except Exception:
            pass
        return np.asarray(img)

    @torch.no_grad()
    def __call__(self, components, state):
        bs = self.get_block_state(state)
        tr, vae, scheduler = components.transformer, components.vae, components.scheduler
        device, dtype = tr.device, tr.dtype
        vsf = 2 ** (len(vae.config.block_out_channels) - 1)
        quant = vsf * 2
        num_channels_latents = tr.config.in_channels // 4
        height, width = int(bs.height), int(bs.width)
        if height % quant or width % quant:
            raise ValueError(f"height/width must be multiples of {quant}.")
        nsteps = int(bs.num_inference_steps)
        guidance_embeds = tr.config.guidance_embeds
        batch_size = 1

        prompt_embeds, pooled, text_ids = self._encode_prompt(
            components, bs.prompt, bs.prompt_2, int(bs.max_sequence_length), device, dtype)
        guidance = (torch.full([1], bs.guidance_scale, device=device, dtype=torch.float32).expand(batch_size)
                    if guidance_embeds else None)

        lh, lw = 2 * (height // quant), 2 * (width // quant)
        latents = randn_tensor((batch_size, num_channels_latents, lh, lw),
                               generator=bs.generator, device=device, dtype=dtype)
        latents = FluxPipeline._pack_latents(latents, batch_size, num_channels_latents, lh, lw)
        img_ids = FluxPipeline._prepare_latent_image_ids(None, lh // 2, lw // 2, device, dtype)

        image_seq_len = latents.shape[1]
        sigmas = np.linspace(1.0, 1 / nsteps, nsteps)
        cfg = scheduler.config
        mu = calculate_shift(image_seq_len, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(scheduler, nsteps, device, sigmas=sigmas, mu=mu)

        # build the ID embedding from the reference face, then install the injection hooks
        encoder = self._get_encoder(device, dtype)
        id_np = resize_numpy_image_long(self._to_numpy_rgb(bs.id_image), 1024)
        id_embedding = encoder.get_id_embedding(id_np)
        handles = _install_pulid(tr, encoder, id_embedding, float(bs.id_weight))
        try:
            for t in timesteps:
                noise_pred = tr(hidden_states=latents, timestep=t.expand(latents.shape[0]).to(dtype) / 1000,
                                guidance=guidance, pooled_projections=pooled, encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids, img_ids=img_ids, return_dict=False)[0]
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if bs.output_type == "latent":
                image = latents
            else:
                lat = FluxPipeline._unpack_latents(latents, height, width, vsf)
                lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
                image = vae.decode(lat.to(vae.dtype), return_dict=False)[0]
                from diffusers.image_processor import VaeImageProcessor
                image = VaeImageProcessor(vae_scale_factor=vsf).postprocess(image, output_type=bs.output_type)
        finally:
            for h in handles:
                h.remove()

        bs.images = image
        self.set_block_state(state, bs)
        return components, state
