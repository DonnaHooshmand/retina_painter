"""
setup_retfound.py — First-time setup helper for RetinaPainter (RETFound mode)

Downloads and caches the RETFound OCT weights from HuggingFace.
Run this once before using --model-type retfound.

Usage:
    python setup_retfound.py
    python setup_retfound.py --token hf_xxxx   # skip the prompt
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Download RETFound weights for RetinaPainter."
    )
    parser.add_argument(
        "--token",
        help="HuggingFace access token (from https://huggingface.co/settings/tokens). "
             "If omitted you will be prompted.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Check huggingface_hub is available
    # ------------------------------------------------------------------ #
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        print("ERROR: huggingface_hub is not installed.")
        print("Run:  pip install huggingface_hub>=0.20.0")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Authenticate
    # ------------------------------------------------------------------ #
    token = args.token
    if not token:
        print("=" * 60)
        print("RetinaPainter — RETFound weight setup")
        print("=" * 60)
        print()
        print("The RETFound OCT weights are hosted on HuggingFace and")
        print("require a free account + access approval.")
        print()
        print("Steps if you haven't done this yet:")
        print("  1. Create a free account at https://huggingface.co")
        print("  2. Request access at:")
        print("       https://huggingface.co/YukunZhou/RETFound_mae_natureOCT")
        print("  3. Generate a Read token at:")
        print("       https://huggingface.co/settings/tokens")
        print()
        token = input("Paste your HuggingFace token (or press Enter to skip): ").strip()
        if not token:
            print("\nNo token provided. Skipping authentication.")
            print("If weights are already cached this may still work.")

    if token:
        print("\nLogging in to HuggingFace...")
        login(token=token, add_to_git_credential=False)
        try:
            user = whoami()
            print(f"Authenticated as: {user['name']}")
        except Exception:
            print("Warning: could not verify authentication, continuing anyway.")

    # ------------------------------------------------------------------ #
    # 3. Download weights
    # ------------------------------------------------------------------ #
    # Add trainer/src to path so we can import retfound_model
    trainer_src = Path(__file__).parent / "trainer" / "src"
    if str(trainer_src) not in sys.path:
        sys.path.insert(0, str(trainer_src))

    try:
        from retfound_model import download_retfound_weights
    except ImportError as e:
        print(f"\nERROR: Could not import retfound_model: {e}")
        print("Make sure you are running this script from the repo root.")
        sys.exit(1)

    print("\nDownloading RETFound OCT weights (~330 MB)...")
    print("This only happens once — weights are cached for future use.\n")

    try:
        path = download_retfound_weights()
        print(f"\nDone! Weights cached at:\n  {path}")
        print("\nYou can now run RetinaPainter in RETFound mode:")
        print("  start-trainer --syncdir <your_sync_dir> --model-type retfound")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nCommon causes:")
        print("  - Access not yet approved (check your HuggingFace notifications)")
        print("  - Wrong or expired token")
        sys.exit(1)


if __name__ == "__main__":
    main()
