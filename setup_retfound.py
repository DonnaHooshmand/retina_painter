"""
setup_retfound.py — First-time setup helper for RetinaPainter (RETFound mode)

Downloads and caches the RETFound OCT weights.
Sources tried in order:
  1. HuggingFace Hub (monish563/RETFOUND) — gated; HF account + token after accepting access
  2. Google Drive (official release, no account needed)

Run this once before using --model-type retfound.

Usage:
    python setup_retfound.py
    python setup_retfound.py --token hf_xxxx   # use a specific HuggingFace token
    python setup_retfound.py --gdrive          # skip HuggingFace, use Google Drive
"""

import argparse
import sys
from pathlib import Path

# Google Drive file ID for the official RETFound OCT weights
# Source: https://huggingface.co/open-eye/RETFound_MAE
GDRIVE_FILE_ID = "1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2"
CACHE_PATH = Path.home() / ".cache" / "retina_painter" / "RETFound_oct.pth"


def download_from_gdrive(dest: Path) -> bool:
    """Try to download from Google Drive using gdown. Returns True on success."""
    try:
        import gdown
    except ImportError:
        print("  Installing gdown for Google Drive download...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"  Downloading from Google Drive...")
    try:
        gdown.download(url, str(dest), quiet=False)
        return dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as e:
        print(f"  Google Drive download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download RETFound OCT weights for RetinaPainter."
    )
    parser.add_argument(
        "--token",
        help="HuggingFace access token (from https://huggingface.co/settings/tokens). "
             "If omitted you will be prompted.",
    )
    parser.add_argument(
        "--gdrive",
        action="store_true",
        help="Skip HuggingFace and download directly from Google Drive.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RetinaPainter — RETFound weight setup")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------ #
    # Already cached?
    # ------------------------------------------------------------------ #
    if CACHE_PATH.exists() and CACHE_PATH.stat().st_size > 1_000_000:
        print(f"Weights already cached at:\n  {CACHE_PATH}")
        print("\nNothing to do. You can run RetinaPainter in RETFound mode:")
        print("  start-trainer --syncdir <your_sync_dir> --model-type retfound")
        return

    # ------------------------------------------------------------------ #
    # Add trainer/src to path
    # ------------------------------------------------------------------ #
    trainer_src = Path(__file__).parent / "trainer" / "src"
    if str(trainer_src) not in sys.path:
        sys.path.insert(0, str(trainer_src))

    # ------------------------------------------------------------------ #
    # Option A: Google Drive (no account needed)
    # ------------------------------------------------------------------ #
    if args.gdrive:
        print("Downloading RETFound OCT weights from Google Drive (~4 GB)...")
        print("This only happens once — weights are cached for future use.\n")
        if download_from_gdrive(CACHE_PATH):
            print(f"\nDone! Weights cached at:\n  {CACHE_PATH}")
            print("\nYou can now run RetinaPainter in RETFound mode:")
            print("  start-trainer --syncdir <your_sync_dir> --model-type retfound")
        else:
            _print_manual_instructions()
            sys.exit(1)
        return

    # ------------------------------------------------------------------ #
    # Option B: HuggingFace Hub
    # ------------------------------------------------------------------ #
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Run:  pip install huggingface_hub>=0.20.0")
        sys.exit(1)

    token = args.token
    if not token:
        print("The RETFound OCT weights can be downloaded two ways:\n")
        print("  [1] HuggingFace Hub — gated; create a free account, accept access on the model page, then use a read token")
        print("        https://huggingface.co/monish563/RETFOUND")
        print()
        print("  [2] Google Drive    — no account needed")
        print(f"        https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        print()
        choice = input("Enter 1 for HuggingFace, 2 for Google Drive, or press Enter for Google Drive: ").strip()
        if choice == "1":
            token = input("\nPaste your HuggingFace token: ").strip()
        else:
            print("\nDownloading from Google Drive (~4 GB)...")
            print("This only happens once — weights are cached for future use.\n")
            if download_from_gdrive(CACHE_PATH):
                print(f"\nDone! Weights cached at:\n  {CACHE_PATH}")
                print("\nYou can now run RetinaPainter in RETFound mode:")
                print("  start-trainer --syncdir <your_sync_dir> --model-type retfound")
            else:
                _print_manual_instructions()
                sys.exit(1)
            return

    if token:
        print("\nLogging in to HuggingFace...")
        login(token=token, add_to_git_credential=False)
        try:
            user = whoami()
            print(f"Authenticated as: {user['name']}")
        except Exception:
            print("Warning: could not verify authentication, continuing anyway.")

    try:
        from retfound_model import download_retfound_weights
    except ImportError as e:
        print(f"\nERROR: Could not import retfound_model: {e}")
        print("Make sure you are running this script from the repo root.")
        sys.exit(1)

    print("\nDownloading RETFound OCT weights from HuggingFace (~4 GB)...")
    print("This only happens once — weights are cached for future use.\n")

    try:
        path = download_retfound_weights()
        print(f"\nDone! Weights cached at:\n  {path}")
        print("\nYou can now run RetinaPainter in RETFound mode:")
        print("  start-trainer --syncdir <your_sync_dir> --model-type retfound")
    except RuntimeError:
        print("\nHuggingFace download failed. Trying Google Drive as fallback...")
        if download_from_gdrive(CACHE_PATH):
            print(f"\nDone! Weights cached at:\n  {CACHE_PATH}")
            print("\nYou can now run RetinaPainter in RETFound mode:")
            print("  start-trainer --syncdir <your_sync_dir> --model-type retfound")
        else:
            _print_manual_instructions()
            sys.exit(1)


def _print_manual_instructions():
    print("\nAutomatic download failed. Download the weights manually:")
    print(f"  https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
    print(f"\nSave the file as:\n  {CACHE_PATH}")
    print("(Create the folder if it doesn't exist.)")


if __name__ == "__main__":
    main()
