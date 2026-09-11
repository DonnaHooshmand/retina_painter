"""Create reproducible, nested image subsets for clinician annotation.

Images receive a stable pseudo-random rank from the SHA-256 digest of the
requested seed and filename.  A subset of size N contains ranks 1 through N,
so every smaller subset is exactly contained in every larger subset.

The script is deliberately non-destructive: it never deletes files, refuses
unexpected images in an existing subset, and verifies every copied image.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SIZES = (150, 200, 250, 300, 350, 400, 450, 500, 550, 600)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ALGORITHM = "sha256_seeded_filename_rank_v1"
EYE_PATTERN = re.compile(r"_(OD|OS)_", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="2026")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--exclude-dir", type=Path, action="append", default=[],
        help="Exclude source images with names found under this directory. "
             "May be specified more than once.")
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print the deterministic plan without creating files.")
    return parser.parse_args()


def image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (path for path in iterator
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (path.name.casefold(), path.name))


def assert_unique_names(paths: list[Path], label: str) -> None:
    seen: dict[str, Path] = {}
    for path in paths:
        key = path.name.casefold()
        if key in seen:
            raise ValueError(
                f"Duplicate image filename in {label}: "
                f"{seen[key]} and {path}")
        seen[key] = path


def rank_key(seed: str, filename: str) -> tuple[bytes, str, str]:
    digest = hashlib.sha256(
        seed.encode("utf-8") + b"\0" + filename.encode("utf-8")).digest()
    return digest, filename.casefold(), filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def patient_and_eye(filename: str) -> tuple[str, str]:
    patient_id = filename.split("_", 1)[0]
    match = EYE_PATTERN.search(filename)
    return patient_id, match.group(1).upper() if match else ""


def smallest_subset(rank: int, sizes: list[int]) -> int:
    return next(size for size in sizes if rank <= size)


def build_plan(args: argparse.Namespace) -> tuple[list[Path], dict]:
    source = args.source.resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")

    sizes = sorted(set(args.sizes))
    if not sizes or sizes[0] <= 0:
        raise ValueError("Subset sizes must be positive")

    source_images = image_files(source)
    assert_unique_names(source_images, "source")

    excluded_names: set[str] = set()
    exclude_directories = []
    for directory in args.exclude_dir:
        resolved = directory.resolve()
        if not resolved.is_dir():
            raise ValueError(f"Exclusion directory does not exist: {resolved}")
        exclude_directories.append(str(resolved))
        excluded_names.update(
            path.name.casefold()
            for path in image_files(resolved, recursive=True))

    eligible = [
        path for path in source_images
        if path.name.casefold() not in excluded_names
    ]
    eligible.sort(key=lambda path: rank_key(str(args.seed), path.name))
    if len(eligible) < sizes[-1]:
        raise ValueError(
            f"Need {sizes[-1]} eligible images but found {len(eligible)}")

    selected = eligible[:sizes[-1]]
    metadata = {
        "source": str(source),
        "output": str(args.output.resolve()),
        "seed": str(args.seed),
        "algorithm": ALGORITHM,
        "sizes": sizes,
        "source_image_count": len(source_images),
        "eligible_image_count": len(eligible),
        "excluded_matching_source_images": len(source_images) - len(eligible),
        "exclude_directories": exclude_directories,
    }
    return selected, metadata


def write_dataset(selected: list[Path], metadata: dict) -> None:
    output = Path(metadata["output"])
    sizes = metadata["sizes"]
    output.mkdir(parents=True, exist_ok=True)

    hashes = {path.name: sha256_file(path) for path in selected}
    manifest_rows = []
    for rank, source_path in enumerate(selected, start=1):
        patient_id, eye = patient_and_eye(source_path.name)
        manifest_rows.append({
            "rank": rank,
            "first_subset": smallest_subset(rank, sizes),
            "filename": source_path.name,
            "patient_id": patient_id,
            "eye": eye,
            "bytes": source_path.stat().st_size,
            "sha256": hashes[source_path.name],
            "source_path": str(source_path),
        })

    for size in sizes:
        subset_dir = output / str(size)
        subset_dir.mkdir(exist_ok=True)
        desired = selected[:size]
        desired_names = {path.name.casefold() for path in desired}
        existing = image_files(subset_dir)
        unexpected = [
            path.name for path in existing
            if path.name.casefold() not in desired_names
        ]
        if unexpected:
            raise RuntimeError(
                f"Refusing to alter {subset_dir}; unexpected image(s): "
                + ", ".join(unexpected[:10]))

        for source_path in desired:
            destination = subset_dir / source_path.name
            if not destination.exists():
                shutil.copy2(source_path, destination)
            if sha256_file(destination) != hashes[source_path.name]:
                raise RuntimeError(f"Copy verification failed: {destination}")

        copied = image_files(subset_dir)
        if len(copied) != size:
            raise RuntimeError(
                f"Expected {size} images in {subset_dir}, found {len(copied)}")

    manifest_path = output / "manifest.csv"
    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(manifest_path)

    metadata = dict(metadata)
    metadata["generated_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["selected_image_count"] = len(selected)
    metadata["copies_are_independent"] = True
    atomic_write_text(
        output / "dataset_metadata.json",
        json.dumps(metadata, indent=2) + "\n")

    excluded = ", ".join(metadata["exclude_directories"]) or "none"
    readme = f"""RIPL_dataset_2026 nested clinician-annotation subsets

Source: {metadata['source']}
Seed: {metadata['seed']}
Ranking algorithm: {metadata['algorithm']}
Subset sizes: {', '.join(map(str, sizes))}
Excluded directories: {excluded}
Excluded matching source images: {metadata['excluded_matching_source_images']}

The seed and each filename are hashed to create one reproducible random order.
Folder N contains ranks 1 through N. Therefore every smaller folder is an
exact subset of every larger folder. Images are independent copies, not links.

In RetinaPainter, select the numeric child folder (for example, 150 or 600),
not this parent directory. Keep manifest.csv and dataset_metadata.json with
the collection; they freeze the selection and record provenance.

The selection is image-level, not patient-grouped. Before research model
evaluation, assign train/validation/test partitions by patient_id from the
manifest so images from one patient cannot cross evaluation boundaries.
"""
    atomic_write_text(output / "README.txt", readme)


def main() -> None:
    args = parse_args()
    selected, metadata = build_plan(args)
    patients = {patient_and_eye(path.name)[0] for path in selected}
    print(json.dumps({
        **metadata,
        "selected_image_count": len(selected),
        "selected_patient_count": len(patients),
        "first_five": [path.name for path in selected[:5]],
    }, indent=2))
    if not args.plan_only:
        write_dataset(selected, metadata)
        print(f"Created and verified nested datasets at {metadata['output']}")


if __name__ == "__main__":
    main()
