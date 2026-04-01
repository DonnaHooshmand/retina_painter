"""
RETFoundSeg: segmentation model built on the RETFound ViT-Large backbone.

The encoder is the RETFound foundation model (Zhou et al., 2023).
The decoder is a lightweight 4-stage convolutional upsampler that maps the
14×14 patch-token feature map back to the original 224×224 resolution.

Weight download
---------------
RETFound weights are hosted on HuggingFace Hub.  Call
``download_retfound_weights()`` to fetch them on first use; they are cached
under ``~/.cache/retina_painter/``.

If the HuggingFace Hub is unavailable (e.g. air-gapped compute), place the
checkpoint at the cache path manually and the function will skip the download.

Expected HuggingFace repository : ``rmaphoh/RETFound-MAE``
Expected filename                : ``RETFound_oct.pth``
"""

from pathlib import Path

import torch
import torch.nn as nn

from retfound_vit import build_retfound_vit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETFOUND_HF_REPO = "rmaphoh/RETFound-MAE"
RETFOUND_OCT_FILENAME = "RETFound_oct.pth"
_MANUAL_DOWNLOAD_URL = "https://github.com/rmaphoh/RETFound_MAE#usage"


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------

def download_retfound_weights(cache_dir: Path = None) -> Path:
    """
    Return a local path to the RETFound OCT checkpoint, downloading if needed.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching the checkpoint.
        Defaults to ``~/.cache/retina_painter/``.

    Returns
    -------
    Path to the local ``.pth`` file.

    Raises
    ------
    RuntimeError
        If the file is neither cached nor downloadable.
    """
    cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "retina_painter"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / RETFOUND_OCT_FILENAME

    if local_path.exists():
        print(f"Using cached RETFound weights: {local_path}")
        return local_path

    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading RETFound OCT weights from HuggingFace ({RETFOUND_HF_REPO})…")
        downloaded = hf_hub_download(
            repo_id=RETFOUND_HF_REPO,
            filename=RETFOUND_OCT_FILENAME,
            local_dir=str(cache_dir),
        )
        return Path(downloaded)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download RETFound weights automatically: {exc}\n"
            f"Please download '{RETFOUND_OCT_FILENAME}' manually from\n"
            f"  {_MANUAL_DOWNLOAD_URL}\n"
            f"and place it at:\n"
            f"  {local_path}"
        ) from exc


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class _UpBlock(nn.Sequential):
    """ConvTranspose2d upsample → GroupNorm → ReLU → Conv refinement."""

    def __init__(self, in_channels: int, out_channels: int):
        groups = min(32, out_channels)
        super().__init__(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )


class _SegDecoder(nn.Module):
    """
    Progressive 4× upsampling: 14 → 28 → 56 → 112 → 224.

    Input : (B, 1024, 14, 14)  — reshaped ViT patch tokens
    Output: (B, num_classes, 224, 224)
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 2):
        super().__init__()
        self.up1 = _UpBlock(in_channels, 512)   # 14 → 28
        self.up2 = _UpBlock(512, 256)            # 28 → 56
        self.up3 = _UpBlock(256, 128)            # 56 → 112
        self.up4 = _UpBlock(128, 64)             # 112 → 224
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.out_conv(x)


# ---------------------------------------------------------------------------
# Full segmentation model
# ---------------------------------------------------------------------------

class RETFoundSeg(nn.Module):
    """
    RETFound ViT-Large encoder + lightweight segmentation decoder.

    Input patch size : 224 × 224   (in_w == out_w == 224)
    Output           : (B, num_classes, 224, 224) logits

    The model expects image tiles in the [0, 1] range (the output of
    ``im_utils.normalize_tile``).  ImageNet normalisation is applied
    internally before passing through the ViT encoder, matching RETFound's
    pretraining distribution.
    """

    # ImageNet statistics used during RETFound pretraining
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, num_classes: int = 2, checkpoint_path=None):
        super().__init__()
        self.encoder = build_retfound_vit(checkpoint_path=checkpoint_path)
        self.decoder = _SegDecoder(in_channels=1024, num_classes=num_classes)

        # Register as buffers so they move to the right device with .to()
        mean = torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(self._STD,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean)
        self.register_buffer("_imagenet_std",  std)

    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale [0, 1] tiles to ImageNet-normalised values."""
        return (x - self._imagenet_mean) / self._imagenet_std

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) float32 tensor with values in [0, 1].

        Returns:
            logits: (B, num_classes, 224, 224)
        """
        x = self._normalize(x)
        tokens = self.encoder.forward_features(x)   # (B, 196, 1024)
        B, N, C = tokens.shape
        g = int(N ** 0.5)                            # 14
        feat = tokens.permute(0, 2, 1).reshape(B, C, g, g)
        return self.decoder(feat)                    # (B, num_classes, 224, 224)
