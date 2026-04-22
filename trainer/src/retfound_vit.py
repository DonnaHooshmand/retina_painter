"""
RETFound ViT-Large encoder.

Architecture matches the MAE ViT-Large used in:
  RETFound: A foundation model for generalizable disease detection
  from retinal images (Zhou et al., 2023)
  https://github.com/rmaphoh/RETFound_MAE

Uses timm building blocks (>=0.9.0) for the transformer blocks and patch
embedding, with the sin-cos positional embeddings from the original MAE repo.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn

try:
    from timm.layers import PatchEmbed
except ImportError:
    from timm.models.layers import PatchEmbed  # timm <0.9 fallback

try:
    from timm.models.vision_transformer import Block
except ImportError:
    raise ImportError("timm>=0.9.0 is required: pip install 'timm>=0.9.0'")


# ---------------------------------------------------------------------------
# Sin-cos 2D positional embedding (identical to MAE / RETFound)
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim, positions):
    """positions: 1-D array of grid indices."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)          # (D/2,)
    positions = positions.reshape(-1)        # (M,)
    out = np.einsum("m,d->md", positions, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (M, D)


def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Return sin-cos positional embeddings for a (grid_size × grid_size) grid.

    Returns shape (grid_size**2, embed_dim) without cls_token, or
    (1 + grid_size**2, embed_dim) with cls_token (first row is zeros).
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)   # w first to match MAE
    grid = np.stack([grid_w, grid_h], axis=0)        # (2, G, G)

    half = embed_dim // 2
    emb_w = _get_1d_sincos_pos_embed(half, grid[0].reshape(-1))  # (G*G, D/2)
    emb_h = _get_1d_sincos_pos_embed(half, grid[1].reshape(-1))  # (G*G, D/2)
    emb = np.concatenate([emb_w, emb_h], axis=1)                 # (G*G, D)

    if cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return emb


# ---------------------------------------------------------------------------
# ViT-Large matching RETFound's checkpoint format
# ---------------------------------------------------------------------------

class RETFoundViT(nn.Module):
    """
    ViT-Large (patch_size=16, embed_dim=1024, depth=24, num_heads=16).

    The key public method is ``forward_features(x)`` which returns
    the (B, num_patches, embed_dim) patch token sequence after the final
    LayerNorm — the cls token is dropped.  This is the input expected by
    the segmentation decoder in retfound_model.py.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed is fixed (sin-cos); not updated by optimizer
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # Fixed sin-cos positional embedding
        grid_size = int(self.patch_embed.num_patches ** 0.5)
        pos_embed = _get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], grid_size, cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Patch projection: init like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.normal_(self.cls_token, std=1e-6)

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor, values in [0, 1] range
               (ImageNet normalisation is applied upstream in RETFoundSeg).
        Returns:
            patch_tokens: (B, num_patches, embed_dim)
        """
        x = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)     # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                      # (B, N+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]   # drop cls token → (B, N, D)

    # ------------------------------------------------------------------
    def forward_multi_features(self, x: torch.Tensor, indices=(5, 11, 17, 23)):
        """
        Run the encoder and return intermediate block outputs at the given
        indices, for use as U-Net-style skip connections (RFA-U-Net).

        The final output in the returned list has the block output at
        ``indices[-1]`` (not the post-LayerNorm tokens used by
        ``forward_features``) — this matches the RFA-U-Net reference
        implementation.

        Args:
            x: (B, 3, H, W) image tensor in [0, 1] range.
            indices: block indices to capture (0-indexed).  Default
                ``(5, 11, 17, 23)`` corresponds to layers Z6/Z12/Z18/Z24.

        Returns:
            List of tensors, each (B, num_patches, embed_dim); cls token dropped.
        """
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        wanted = set(indices)
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in wanted:
                outputs.append(x[:, 1:, :])  # drop cls token
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retfound_vit(checkpoint_path=None) -> RETFoundViT:
    """
    Instantiate RETFound ViT-Large.

    If *checkpoint_path* is given the pretrained weights are loaded.
    MAE decoder weights (keys starting with ``decoder_`` or
    ``mask_token``) are silently dropped; remaining mismatches are
    reported via ``strict=False`` and printed to stdout.
    """
    model = RETFoundViT(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if checkpoint_path is not None:
        print(f"Loading RETFound weights from {checkpoint_path} ({checkpoint_path.stat().st_size / 1e9:.1f} GB — please wait)...", flush=True)
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # RETFound checkpoints use a 'model' key at the top level
        state_dict = raw.get("model", raw)
        # Drop MAE decoder / mask-token keys
        encoder_sd = {
            k: v for k, v in state_dict.items()
            if not k.startswith("decoder") and k != "mask_token"
        }
        msg = model.load_state_dict(encoder_sd, strict=False)
        missing = [k for k in msg.missing_keys if "decoder" not in k]
        print("  Weights loaded successfully.", flush=True)
        if missing:
            print(f"  Missing keys after loading: {missing}", flush=True)
        if msg.unexpected_keys:
            print(f"  Unexpected keys (ignored): {len(msg.unexpected_keys)} decoder keys dropped.", flush=True)

    return model
