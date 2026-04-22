"""
RETFoundSegRFA: RFA-U-Net segmentation head on the RETFound ViT-Large backbone.

This is an alternative to ``retfound_model.RETFoundSeg`` that replaces the
plain 4-stage transposed-convolution decoder with a U-Net-style decoder that
consumes four intermediate ViT feature maps (Z6, Z12, Z18, Z24) as skip
connections, passes each through a progressive upsampling pyramid, and fuses
them via attention gates at every decoder stage.

Architecture follows:
  Hayati, A. et al. (2025). RFA-U-Net: A Foundation Model-Driven Approach for
  Accurate Choroid Segmentation in OCT Imaging.  medRxiv 2025.05.03.25326923.
  https://doi.org/10.1101/2025.05.03.25326923

Reference implementation: https://github.com/Alirezahayatimedtech/RFA-U-Net

Shared with ``retfound_model.py``:
  - Encoder weights are the RETFound MAE ViT-Large checkpoint from
    ``~/.cache/retina_painter/RETFound_oct.pth``.
  - Input tiles are expected in the [0, 1] range; ImageNet normalisation is
    applied internally.
  - Input/output resolution is 224×224 (in_w == out_w; no valid-conv crop).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from retfound_vit import build_retfound_vit


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Two 3×3 convs with BatchNorm + ReLU (standard U-Net block)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    Given a gating signal ``g`` (from the decoder path) and a skip signal
    ``s`` (from the encoder), returns ``s`` modulated by a learned spatial
    attention map.
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.w_s = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, g, s):
        wg = self.w_g(g)
        ws = self.w_s(s)
        if wg.shape[-2:] != ws.shape[-2:]:
            ws = F.interpolate(ws, size=wg.shape[-2:], mode="bilinear", align_corners=True)
        attn = self.psi(self.relu(wg + ws))
        if attn.shape[-2:] != s.shape[-2:]:
            s = F.interpolate(s, size=attn.shape[-2:], mode="bilinear", align_corners=True)
        return attn * s


class _DecoderBlock(nn.Module):
    """
    Upsample the decoder feature map by 2×, reduce skip channels, apply an
    attention gate, then fuse by concatenation + two 3×3 convs.

    Args:
        decoder_in:  channels of the incoming decoder feature
        skip_in:     channels of the skip feature
        out_channels: output channels after fusion
    """

    def __init__(self, decoder_in: int, skip_in: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(decoder_in, out_channels, kernel_size=2, stride=2)
        self.reduce_skip = nn.Conv2d(skip_in, out_channels, kernel_size=1)
        self.attn = _AttentionGate(out_channels, out_channels, out_channels)
        self.fuse = _ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, s):
        x = self.up(x)
        s = self.reduce_skip(s)
        if s.shape[-2:] != x.shape[-2:]:
            s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=True)
        s = self.attn(x, s)
        return self.fuse(torch.cat([x, s], dim=1))


class _FinalUp(nn.Module):
    """Last upsample stage with no skip (112 → 224)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_channels, out_channels)

    def forward(self, x):
        return self.conv(self.up(x))


def _make_skip_pyramid(in_ch: int, out_ch: int, up_steps: int) -> nn.Module:
    """
    Progressive 2× upsampling pyramid, channel-reducing as it goes.

    All ViT skips start at 14×14; this projects them to 28, 56, or 112 px as
    required by the decoder stages they feed into.
    """
    layers = []
    cur = in_ch
    # channel schedule: halve each step, but never below out_ch
    schedule = []
    for step in range(up_steps):
        nxt = max(out_ch, cur // 2)
        if step == up_steps - 1:
            nxt = out_ch
        schedule.append(nxt)
        cur = nxt
    cur = in_ch
    for nxt in schedule:
        layers.append(nn.ConvTranspose2d(cur, nxt, kernel_size=2, stride=2))
        layers.append(nn.BatchNorm2d(nxt))
        layers.append(nn.ReLU(inplace=True))
        cur = nxt
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class _RFASegDecoder(nn.Module):
    """
    RFA-U-Net decoder operating on four 14×14 ViT feature maps.

    Decoder path:
        z24 (14², 1024) ─d1→ (28², 512) with skip_proj(z18)
                         ─d2→ (56², 256) with skip_proj(z12)
                         ─d3→ (112², 128) with skip_proj(z6)
                         ─d4→ (224², 64)    no skip
        out: 1×1 conv → num_classes
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 2):
        super().__init__()
        # Skip projections: 14×14 ViT features upsampled + channel-reduced
        self.skip_proj_d1 = _make_skip_pyramid(in_channels, 512, up_steps=1)   # → 28², 512
        self.skip_proj_d2 = _make_skip_pyramid(in_channels, 256, up_steps=2)   # → 56², 256
        self.skip_proj_d3 = _make_skip_pyramid(in_channels, 128, up_steps=3)   # → 112², 128

        self.d1 = _DecoderBlock(decoder_in=in_channels, skip_in=512, out_channels=512)
        self.d2 = _DecoderBlock(decoder_in=512,         skip_in=256, out_channels=256)
        self.d3 = _DecoderBlock(decoder_in=256,         skip_in=128, out_channels=128)
        self.d4 = _FinalUp(in_channels=128, out_channels=64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, z6, z12, z18, z24):
        s1 = self.skip_proj_d1(z18)   # 28², 512
        s2 = self.skip_proj_d2(z12)   # 56², 256
        s3 = self.skip_proj_d3(z6)    # 112², 128

        x = self.d1(z24, s1)          # 28², 512
        x = self.d2(x,   s2)          # 56², 256
        x = self.d3(x,   s3)          # 112², 128
        x = self.d4(x)                # 224², 64
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Full segmentation model
# ---------------------------------------------------------------------------

class RETFoundSegRFA(nn.Module):
    """
    RETFound ViT-Large encoder + RFA-U-Net decoder.

    Input           : (B, 3, 224, 224) float in [0, 1]
    Output          : (B, num_classes, 224, 224) logits

    The encoder returns features at ViT blocks Z6, Z12, Z18, Z24; each is
    reshaped from (B, 196, 1024) to (B, 1024, 14, 14) before entering the
    decoder.

    Parameters
    ----------
    num_classes : int
        Number of output segmentation classes (default 2 — foreground /
        background).
    checkpoint_path : Path, optional
        Path to the RETFound ``.pth`` file.  If ``None``, encoder weights are
        random.
    skip_indices : tuple of 4 ints
        Block indices (0-indexed) to use as skip connections.  Default
        ``(5, 11, 17, 23)`` matches the RFA-U-Net paper.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, num_classes: int = 2, checkpoint_path=None,
                 skip_indices=(5, 11, 17, 23)):
        super().__init__()
        assert len(skip_indices) == 4, "skip_indices must have exactly 4 entries"
        self.skip_indices = tuple(skip_indices)

        self.encoder = build_retfound_vit(checkpoint_path=checkpoint_path)
        self.decoder = _RFASegDecoder(in_channels=1024, num_classes=num_classes)

        mean = torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean)
        self.register_buffer("_imagenet_std",  std)

    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._imagenet_mean) / self._imagenet_std

    @staticmethod
    def _tokens_to_feature_map(tokens: torch.Tensor) -> torch.Tensor:
        """(B, N, C) → (B, C, √N, √N) for square patch grids."""
        b, n, c = tokens.shape
        g = int(n ** 0.5)
        if g * g != n:
            raise RuntimeError(f"Non-square token grid: N={n}")
        return tokens.transpose(1, 2).reshape(b, c, g, g)

    # ------------------------------------------------------------------
    def freeze_encoder_blocks(self, num_blocks: int) -> None:
        """Freeze the first ``num_blocks`` transformer blocks.

        The RFA-U-Net paper freezes 21 of 24 blocks by default so that only
        the last three blocks + decoder are trainable.  ``num_blocks=0``
        leaves the entire encoder trainable; ``num_blocks=24`` freezes every
        block.  The patch embedding, cls token, and positional embedding are
        not touched by this method.
        """
        num_blocks = max(0, min(int(num_blocks), len(self.encoder.blocks)))
        for i, blk in enumerate(self.encoder.blocks):
            requires_grad = i >= num_blocks
            for p in blk.parameters():
                p.requires_grad = requires_grad

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        skips = self.encoder.forward_multi_features(x, indices=self.skip_indices)
        z6  = self._tokens_to_feature_map(skips[0])   # (B, 1024, 14, 14)
        z12 = self._tokens_to_feature_map(skips[1])
        z18 = self._tokens_to_feature_map(skips[2])
        z24 = self._tokens_to_feature_map(skips[3])
        return self.decoder(z6, z12, z18, z24)
