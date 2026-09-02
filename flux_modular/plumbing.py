"""FLUX latent-packing / scheduling helpers.

These mirror the private staticmethods duplicated across ``FluxPipeline`` / ``FluxControlPipeline`` /
``FluxFillPipeline`` / Kontext — and re-copied into ~12/13 of our custom blocks. Kept public here so a
block imports them instead of re-deriving (and re-mis-deriving) them. Candidate for upstream as public
``diffusers.pipelines.flux`` utilities.
"""

import torch


def pack_latents(latents, batch, channels, height, width):
    """(B, C, H, W) -> (B, (H/2)*(W/2), C*4). ``height``/``width`` are LATENT-grid dims (e.g. 128 at 1024px)."""
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch, (height // 2) * (width // 2), channels * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    """Inverse of :func:`pack_latents`. ``height``/``width`` are PIXEL dims."""
    b, _, ch = latents.shape
    h = 2 * (height // (vae_scale_factor * 2))
    w = 2 * (width // (vae_scale_factor * 2))
    latents = latents.view(b, h // 2, w // 2, ch // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, ch // 4, h, w)


def prepare_latent_image_ids(height, width, device, dtype):
    """FLUX positional ids for a ``height`` x ``width`` packed grid (row-major)."""
    ids = torch.zeros(height, width, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(height)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width)[None, :]
    return ids.reshape(height * width, 3).to(device=device, dtype=dtype)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.15):
    """FlowMatch ``mu`` shift for a packed image-token sequence length (defaults = FLUX scheduler config)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)
