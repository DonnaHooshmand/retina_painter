"""Create visual audit overlays for a curated RetinaPainter dataset.

The red-only overlays show pixels that a doctor explicitly painted as
foreground.  The audit overlays distinguish those red pixels from foreground
that the corrected mask inherited from the project's saved model prediction.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


DOCTOR_RED = np.array([255, 32, 32], dtype=np.float32)
INHERITED_CYAN = np.array([0, 255, 255], dtype=np.float32)


def _blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if np.any(mask):
        rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha


def create_overlays(dataset_dir: Path) -> dict[str, int]:
    dataset_dir = dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest contains no records: {manifest_path}")

    doctor_overlay_dir = dataset_dir / "doctor_foreground_overlays"
    audit_overlay_dir = dataset_dir / "ground_truth_audit_overlays"
    doctor_mask_dir = dataset_dir / "doctor_foreground_masks"
    for path in (doctor_overlay_dir, audit_overlay_dir, doctor_mask_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing directory: {path}")
        path.mkdir()

    audit_rows: list[dict[str, object]] = []
    for row in rows:
        output_name = row["output_name"]
        # Alternate doctor references (_option2) share one unique source image.
        with Image.open(dataset_dir / "images" / row["source_image_name"]) as image:
            source = np.array(image.convert("RGB"), dtype=np.float32)
        with Image.open(row["annotation_path"]) as image:
            annotation = np.array(image)
        with Image.open(dataset_dir / "ground_truth" / output_name) as image:
            corrected = np.array(image)

        if source.shape[:2] != annotation.shape[:2] or annotation.shape != corrected.shape:
            raise ValueError(f"Shape mismatch for {output_name}")

        doctor_foreground = annotation[..., 0] > 0
        corrected_foreground = corrected[..., 3] > 0
        inherited_foreground = corrected_foreground & ~doctor_foreground

        doctor_overlay = source.copy()
        _blend(doctor_overlay, doctor_foreground, DOCTOR_RED, alpha=0.72)
        Image.fromarray(np.rint(doctor_overlay).astype(np.uint8), mode="RGB").save(
            doctor_overlay_dir / output_name
        )

        audit_overlay = source.copy()
        _blend(audit_overlay, inherited_foreground, INHERITED_CYAN, alpha=0.48)
        _blend(audit_overlay, doctor_foreground, DOCTOR_RED, alpha=0.78)
        Image.fromarray(np.rint(audit_overlay).astype(np.uint8), mode="RGB").save(
            audit_overlay_dir / output_name
        )

        doctor_mask = np.where(doctor_foreground, 255, 0).astype(np.uint8)
        Image.fromarray(doctor_mask, mode="L").save(doctor_mask_dir / output_name)

        doctor_pixels = int(np.count_nonzero(doctor_foreground))
        inherited_pixels = int(np.count_nonzero(inherited_foreground))
        corrected_pixels = int(np.count_nonzero(corrected_foreground))
        audit_rows.append(
            {
                "output_name": output_name,
                "project": row["project"],
                "doctor_foreground_pixels": doctor_pixels,
                "inherited_model_foreground_pixels": inherited_pixels,
                "corrected_foreground_pixels": corrected_pixels,
                "inherited_fraction_of_corrected_foreground": (
                    inherited_pixels / corrected_pixels if corrected_pixels else 0.0
                ),
                "defined_fraction": row["defined_fraction"],
                "unsure_pixels": row["unsure_pixels"],
            }
        )

    with (dataset_dir / "ground_truth_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    readme = """Ground-truth visual audit

doctor_foreground_overlays/
  The source OCT with explicit doctor-painted foreground shown in red. This is
  the safest visual reference for seeing exactly what the doctor highlighted.

ground_truth_audit_overlays/
  Red = explicit doctor-painted foreground.
  Cyan = foreground inherited from the project's saved model segmentation.
  Uncolored = source OCT only.

doctor_foreground_masks/
  Binary masks containing only explicit red doctor marks. A value of 255 means
  doctor-painted foreground; 0 includes explicit background and untouched
  pixels, so these files must not be described as independently dense expert
  masks without an additional grading assumption.

ground_truth_audit.csv
  Pixel counts and inherited-model fraction for every record.

The original ground_truth/ masks are retained unchanged for traceability.
"""
    (dataset_dir / "OVERLAYS_README.txt").write_text(readme, encoding="utf-8")

    return {
        "record_count": len(rows),
        "doctor_overlay_count": len(list(doctor_overlay_dir.glob("*.png"))),
        "audit_overlay_count": len(list(audit_overlay_dir.glob("*.png"))),
        "doctor_mask_count": len(list(doctor_mask_dir.glob("*.png"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    for key, value in create_overlays(args.dataset_dir).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
