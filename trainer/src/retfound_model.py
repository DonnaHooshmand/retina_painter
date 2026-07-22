"""
RETFoundSeg: segmentation model built on the RETFound ViT-Large backbone.

The encoder is the RETFound foundation model (Zhou et al., 2023).
The decoder is a lightweight 4-stage convolutional upsampler that maps the
14×14 patch-token feature map back to the original 224×224 resolution.

Weight download
---------------
``download_retfound_weights()`` returns a local path to the checkpoint,
caching it at ``~/.cache/retina_painter/RETFound_oct.pth``.

Recommended path (the one that actually works for most users): run
``setup_retfound.py`` once. It downloads ``RETFound_oct.pth`` from Google
Drive and writes it to the cache path above; this loader then finds it and
skips any network access.

Automatic HuggingFace fallback (best-effort only): if the checkpoint is not
already cached, the function attempts ``hf_hub_download`` from the gated repo
``iszt/RETFound_mae_natureOCT``. Note that repo (a) is access-gated — you must
request access and call ``huggingface_hub.login()`` first — and (b) ships its
weights as ``model.safetensors``, not ``RETFound_oct.pth``, so the automatic
``.pth`` download will usually fail. Prefer ``setup_retfound.py``.

If the Hub is unavailable (e.g. air-gapped compute), place
``RETFound_oct.pth`` at the cache path manually and the download is skipped.
"""

from pathlib import Path

import torch
import torch.nn as nn

from retfound_vit import build_retfound_vit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETFOUND_HF_REPO = "iszt/RETFound_mae_natureOCT"
RETFOUND_OCT_FILENAME = "RETFound_oct.pth"
_MANUAL_DOWNLOAD_URL = "https://huggingface.co/iszt/RETFound_mae_natureOCT"


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
            f"The gated repo '{RETFOUND_HF_REPO}' ships 'model.safetensors', not "
            f"'{RETFOUND_OCT_FILENAME}', so the automatic download often fails.\n"
            f"Recommended fix: run setup_retfound.py to fetch "
            f"'{RETFOUND_OCT_FILENAME}' from Google Drive.\n"
            f"Alternatively, download it manually from\n"
            f"  {_MANUAL_DOWNLOAD_URL}\n"
            f"(request access + huggingface_hub.login() first) and place it at:\n"
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

    def freeze_encoder_blocks(self, num_blocks: int) -> None:
        """Freeze the first ``num_blocks`` RETFound transformer blocks.

        Plain RETFound and RETFound-RFA use the same partial fine-tuning
        policy so their comparison isolates the decoder architecture. The
        decoder and remaining encoder blocks stay trainable.
        """
        num_blocks = max(0, min(int(num_blocks), len(self.encoder.blocks)))
        for i, block in enumerate(self.encoder.blocks):
            requires_grad = i >= num_blocks
            for parameter in block.parameters():
                parameter.requires_grad = requires_grad

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
