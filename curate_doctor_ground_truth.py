"""Build paired images and corrected masks from RetinaPainter projects.

RetinaPainter annotations are sparse corrections rather than dense masks:

* red / channel 0 marks foreground (a false negative),
* green / channel 1 marks background (a false positive), and
* blue / channel 2 marks an unsure pixel and leaves the prediction unchanged.

For every annotation, this script starts from the same-named segmentation in
that project, applies the foreground/background corrections, and copies the
matching source image.  When multiple projects annotated the same image, later
versions receive ``_option2``, ``_option3``, and so on.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


SEGMENTATION_FOREGROUND_RGBA = np.array([0, 255, 255, 178], dtype=np.uint8)


@dataclass(frozen=True)
class ProjectInfo:
    root: Path
    name: str
    dataset: str
    model_type: str


@dataclass(frozen=True)
class Record:
    project: ProjectInfo
    annotation_path: Path
    annotation_split: str
    segmentation_path: Path
    source_image_path: Path
    output_mask_name: str
    option_number: int


def _load_project(project_root: Path) -> ProjectInfo:
    project_root = project_root.resolve()
    project_files = sorted(project_root.glob("*.seg_proj"))
    if len(project_files) != 1:
        raise ValueError(
            f"Expected exactly one .seg_proj file in {project_root}, "
            f"found {len(project_files)}"
        )
    with project_files[0].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return ProjectInfo(
        root=project_root,
        name=metadata["name"],
        dataset=metadata["dataset"],
        model_type=metadata.get("model_type", "unet"),
    )


def _index_source_images(dataset_dir: Path) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    for path in sorted(dataset_dir.iterdir()):
        if not path.is_file():
            continue
        key = path.stem.casefold()
        if key in by_stem:
            raise ValueError(
                f"Dataset has multiple files with the same stem: "
                f"{by_stem[key].name!r} and {path.name!r}"
            )
        by_stem[key] = path
    return by_stem


def _option_name(filename: str, option_number: int, suffix: str | None = None) -> str:
    path = Path(filename)
    output_suffix = path.suffix if suffix is None else suffix
    option_suffix = "" if option_number == 1 else f"_option{option_number}"
    return f"{path.stem}{option_suffix}{output_suffix}"


def _collect_records(
    projects: Iterable[ProjectInfo], dataset_dir: Path
) -> tuple[list[Record], dict[str, int]]:
    source_images = _index_source_images(dataset_dir)
    occurrences: dict[str, int] = {}
    records: list[Record] = []

    for project in projects:
        annotation_paths = sorted((project.root / "annotations").rglob("*.png"))
        names_in_project: set[str] = set()
        for annotation_path in annotation_paths:
            key = annotation_path.stem.casefold()
            if key in names_in_project:
                raise ValueError(
                    f"Duplicate annotation stem in {project.name}: "
                    f"{annotation_path.stem}"
                )
            names_in_project.add(key)

            segmentation_path = project.root / "segmentations" / annotation_path.name
            if not segmentation_path.is_file():
                raise FileNotFoundError(
                    f"No segmentation for {annotation_path}: {segmentation_path}"
                )
            if key not in source_images:
                raise FileNotFoundError(
                    f"No source image in {dataset_dir} for {annotation_path.name}"
                )

            occurrences[key] = occurrences.get(key, 0) + 1
            option_number = occurrences[key]
            source_image_path = source_images[key]
            records.append(
                Record(
                    project=project,
                    annotation_path=annotation_path,
                    annotation_split=annotation_path.parent.name,
                    segmentation_path=segmentation_path,
                    source_image_path=source_image_path,
                    output_mask_name=_option_name(
                        annotation_path.name, option_number, suffix=".png"
                    ),
                    option_number=option_number,
                )
            )

    return records, occurrences


def _load_rgba(path: Path, description: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(
            f"Expected an RGBA {description} at {path}, got shape {array.shape}"
        )
    return array


def _correct_segmentation(
    annotation_path: Path, segmentation_path: Path
) -> tuple[np.ndarray, dict[str, int | float]]:
    annotation = _load_rgba(annotation_path, "annotation")
    segmentation = _load_rgba(segmentation_path, "segmentation")
    if annotation.shape != segmentation.shape:
        raise ValueError(
            f"Shape mismatch for {annotation_path.name}: annotation "
            f"{annotation.shape}, segmentation {segmentation.shape}"
        )

    foreground = annotation[..., 0] > 0
    background = annotation[..., 1] > 0
    unsure = annotation[..., 2] > 0
    if np.any(foreground & background):
        raise ValueError(
            f"Foreground and background corrections overlap in {annotation_path}"
        )

    corrected = segmentation.copy()
    corrected[background] = 0
    corrected[foreground] = SEGMENTATION_FOREGROUND_RGBA

    defined = foreground | background
    before = segmentation[..., 3] > 0
    after = corrected[..., 3] > 0
    stats: dict[str, int | float] = {
        "width": int(annotation.shape[1]),
        "height": int(annotation.shape[0]),
        "foreground_correction_pixels": int(np.count_nonzero(foreground)),
        "background_correction_pixels": int(np.count_nonzero(background)),
        "unsure_pixels": int(np.count_nonzero(unsure)),
        "defined_fraction": float(np.count_nonzero(defined) / defined.size),
        "segmentation_foreground_pixels": int(np.count_nonzero(before)),
        "corrected_foreground_pixels": int(np.count_nonzero(after)),
    }
    return corrected, stats


def _write_readme(
    output_dir: Path,
    projects: list[ProjectInfo],
    dataset_dir: Path,
    record_count: int,
    unique_count: int,
    overlap_count: int,
) -> None:
    project_lines = "\n".join(
        f"  {index}. {project.name}: {project.root}"
        for index, project in enumerate(projects, start=1)
    )
    text = f"""RetinaPainter clinician-corrected dataset

Created (UTC): {datetime.now(timezone.utc).isoformat()}
Source dataset: {dataset_dir}
Source projects, in filename-priority order:
{project_lines}

Contents
  images/              One copy of each unique source OCT image. Duplicate
                       _option2 images are intentionally not created.
  ground_truth/        RetinaPainter RGBA masks (transparent background,
                       cyan foreground).
  ground_truth_binary/ Evaluation-friendly 8-bit masks (0 background,
                       255 foreground).
  manifest.csv         Per-mask provenance, coverage, and correction counts.
  summary.json         Machine-readable dataset summary.

Counts
  Unique training images: {unique_count}
  Doctor-derived mask records: {record_count}
  Images annotated by more than one doctor: {overlap_count}

Construction rule
  Each mask starts from the same-named segmentation saved in that doctor's
  project. Red/channel-0 corrections are set to foreground. Green/channel-1
  corrections are set to background. Blue/channel-2 unsure pixels and all
  untouched pixels retain the saved segmentation.

Naming rule
  Training images always keep the original filename and appear only once. The
  first project's mask keeps the original filename. If a later project
  annotated the same image, only its mask/reference products use _option2
  before the extension (then _option3 if ever needed).

Important interpretation note
  These are model-derived, clinician-corrected masks, not independently drawn
  dense expert masks. Where defined_fraction in manifest.csv is below 1.0,
  some pixels still come from the model prediction. Pixels marked unsure also
  retain the model prediction and should be considered separately in any
  evaluation that needs uncertainty-aware scoring.
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def build_curated_dataset(
    project_roots: list[Path], output_dir: Path, dataset_dir: Path | None = None
) -> dict[str, object]:
    projects = [_load_project(path) for path in project_roots]
    dataset_names = {project.dataset for project in projects}
    if len(dataset_names) != 1:
        raise ValueError(
            f"All projects must refer to the same dataset, got {dataset_names}"
        )

    if dataset_dir is None:
        sync_roots = {project.root.parent.parent for project in projects}
        if len(sync_roots) != 1:
            raise ValueError(
                "Projects have different sync roots; pass --dataset-dir explicitly"
            )
        dataset_dir = next(iter(sync_roots)) / "datasets" / projects[0].dataset
    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )

    records, occurrences = _collect_records(projects, dataset_dir)
    if not records:
        raise ValueError("No PNG annotations were found")

    # Validate every image/mask pair before creating the output directory.
    prepared: list[tuple[Record, dict[str, int | float]]] = []
    for record in records:
        corrected, stats = _correct_segmentation(
            record.annotation_path, record.segmentation_path
        )
        with Image.open(record.source_image_path) as image:
            source_size = image.size
        if source_size != (stats["width"], stats["height"]):
            raise ValueError(
                f"Source image size mismatch for {record.source_image_path}: "
                f"{source_size} versus {(stats['width'], stats['height'])}"
            )
        prepared.append((record, stats))

    images_dir = output_dir / "images"
    rgba_dir = output_dir / "ground_truth"
    binary_dir = output_dir / "ground_truth_binary"
    images_dir.mkdir(parents=True)
    rgba_dir.mkdir()
    binary_dir.mkdir()

    manifest_rows: list[dict[str, object]] = []
    for record, stats in prepared:
        corrected, _ = _correct_segmentation(
            record.annotation_path, record.segmentation_path
        )
        image_destination = images_dir / record.source_image_path.name
        if not image_destination.exists():
            shutil.copy2(record.source_image_path, image_destination)
        Image.fromarray(corrected, mode="RGBA").save(
            rgba_dir / record.output_mask_name
        )
        binary = np.where(corrected[..., 3] > 0, 255, 0).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(
            binary_dir / record.output_mask_name
        )
        manifest_rows.append(
            {
                "output_name": record.output_mask_name,
                "source_image_name": record.source_image_path.name,
                "project": record.project.name,
                "doctor_option": record.option_number,
                "annotation_split": record.annotation_split,
                "annotation_path": str(record.annotation_path),
                "segmentation_path": str(record.segmentation_path),
                **stats,
            }
        )

    fieldnames = list(manifest_rows[0])
    with (output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    overlap_count = sum(count > 1 for count in occurrences.values())
    summary: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_dir),
        "projects": [
            {
                "name": project.name,
                "path": str(project.root),
                "model_type": project.model_type,
                "record_count": sum(
                    record.project == project for record in records
                ),
            }
            for project in projects
        ],
        "record_count": len(records),
        "training_image_count": len(occurrences),
        "unique_source_image_count": len(occurrences),
        "overlap_source_image_count": overlap_count,
        "option2_count": sum(record.option_number == 2 for record in records),
        "records_with_unsure_pixels": sum(
            int(stats["unsure_pixels"]) > 0 for _, stats in prepared
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    _write_readme(
        output_dir,
        projects,
        dataset_dir,
        record_count=len(records),
        unique_count=len(occurrences),
        overlap_count=overlap_count,
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "projects",
        nargs="+",
        type=Path,
        help="RetinaPainter project directories, in filename-priority order",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory to create (must not already exist)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Source image directory (normally inferred from the projects)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_curated_dataset(
        args.projects, args.output_dir, dataset_dir=args.dataset_dir
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
