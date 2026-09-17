"""Persistence and export helpers for clinician scan reviews.

RetinaPainter corrections are sparse: red adds foreground and green removes
foreground.  After training begins, untouched pixels retain the model
prediction.  Before training begins, the random prediction is not a useful
label source and only the clinician's red foreground marks are exported.
An explicit absent decision always exports an empty mask.  These helpers
snapshot the reviewed inputs and record which reconstruction rule applies so
later model activity cannot change a completed review.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REVIEW_SCHEMA_VERSION = 1
VALID_DECISIONS = ("present", "absent", "uncertain")
PREDICTION_CORRECTED = "prediction_corrected"
ANNOTATION_FOREGROUND_ONLY = "annotation_foreground_only"
DECISION_OVERRIDE_EMPTY = "decision_override_empty"
VALID_RECONSTRUCTION_MODES = (
    PREDICTION_CORRECTED,
    ANNOTATION_FOREGROUND_ONLY,
    DECISION_OVERRIDE_EMPTY,
)
DECISION_PREVIEW_LABELS = {
    "present": ("RIPL PRESENT", (255, 70, 70, 255)),
    "absent": ("RIPL ABSENT", (40, 255, 90, 255)),
    "uncertain": ("RIPL UNCERTAIN", (255, 200, 40, 255)),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records_path(project_dir: Path) -> Path:
    return project_dir / "reviews" / "review_records.json"


def _training_state_path(project_dir: Path) -> Path:
    return project_dir / "reviews" / "training_state.json"


def _relative_posix(path: Path, project_dir: Path) -> str:
    return path.relative_to(project_dir).as_posix()


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _atomic_save_png(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, destination)


def _load_preview_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow versions before scalable default fonts.
            return ImageFont.load_default()


def create_decision_audit_preview(image_path, segmentation_path,
                                  annotation_path, decision,
                                  output_path) -> Path:
    """Save a labeled visual audit copy without changing model inputs."""
    if decision not in DECISION_PREVIEW_LABELS:
        raise ValueError(f"Invalid decision for audit preview: {decision}")

    base = Image.open(image_path).convert("RGBA")
    segmentation = Image.open(segmentation_path).convert("RGBA")
    if segmentation.size != base.size:
        raise ValueError(
            "Image and segmentation dimensions do not match for review "
            f"preview: {base.size} != {segmentation.size}")
    preview = Image.alpha_composite(base, segmentation)

    if annotation_path is not None:
        annotation = Image.open(annotation_path).convert("RGBA")
        if annotation.size != base.size:
            raise ValueError(
                "Image and annotation dimensions do not match for review "
                f"preview: {base.size} != {annotation.size}")
        preview = Image.alpha_composite(preview, annotation)

    label, color = DECISION_PREVIEW_LABELS[decision]
    font = _load_preview_font(max(18, min(32, base.width // 24)))
    draw = ImageDraw.Draw(preview, "RGBA")
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    padding = max(6, font.size // 3 if hasattr(font, "size") else 8)
    box_width = right - left + padding * 2
    box_height = bottom - top + padding * 2
    draw.rounded_rectangle(
        (8, 8, 8 + box_width, 8 + box_height),
        radius=max(4, padding // 2),
        fill=(0, 0, 0, 210))
    draw.text(
        (8 + padding - left, 8 + padding - top),
        label,
        font=font,
        fill=color)

    output_path = Path(output_path)
    _atomic_save_png(preview.convert("RGB"), output_path)
    return output_path


def load_review_document(project_dir) -> dict:
    """Load a project's review document, returning an empty v1 document."""
    project_dir = Path(project_dir)
    path = _records_path(project_dir)
    if not path.is_file():
        return {"schema_version": REVIEW_SCHEMA_VERSION, "reviews": {}}
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported review schema in {path}: "
            f"{document.get('schema_version')}")
    if not isinstance(document.get("reviews"), dict):
        raise ValueError(f"Review records must be an object in {path}")
    return document


def load_review(project_dir, image_filename: str) -> dict | None:
    """Return the stored review for one source image, if present."""
    return load_review_document(project_dir)["reviews"].get(image_filename)


def completed_review_count(project_dir) -> int:
    """Return the number of scans explicitly marked review-complete."""
    reviews = load_review_document(project_dir)["reviews"].values()
    return sum(bool(review.get("review_complete")) for review in reviews)


def incomplete_warmup_reviews(project_dir, image_order, threshold) -> list[str]:
    """Return required initial filenames that are not review-complete."""
    if threshold is None:
        return []
    try:
        threshold = int(threshold)
    except (TypeError, ValueError):
        return []
    if threshold <= 0:
        return []
    reviews = load_review_document(project_dir)["reviews"]
    required = list(image_order)[:threshold]
    return [
        filename for filename in required
        if not reviews.get(filename, {}).get("review_complete")
    ]


def should_auto_start_training(completed_count, threshold,
                               has_started=False) -> bool:
    """Return whether review progress should trigger the first training run."""
    if has_started or threshold is None:
        return False
    try:
        threshold = int(threshold)
    except (TypeError, ValueError):
        return False
    return threshold > 0 and completed_count >= threshold


def mark_training_started(project_dir) -> dict:
    """Persist that this project has entered model-assisted review.

    The first timestamp is retained across stop/start cycles. Reviews saved
    before this marker use annotation-only reconstruction; reviews saved after
    it may retain correct model suggestions in untouched pixels.
    """
    project_dir = Path(project_dir)
    path = _training_state_path(project_dir)
    if path.is_file():
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    state = {"first_training_started_at_utc": _utc_now()}
    _atomic_write_json(path, state)
    return state


def training_has_started(project_dir) -> bool:
    """Return whether the project has started training at least once.

    Existing projects created before ``training_state.json`` are recognized
    by a non-empty trainer CSV so reopening them does not revert to
    annotation-only mode.
    """
    project_dir = Path(project_dir)
    if _training_state_path(project_dir).is_file():
        return True
    log_dir = project_dir / "logs"
    if not log_dir.is_dir():
        return False
    for path in log_dir.glob("*_train.csv"):
        try:
            with path.open("r", encoding="utf-8") as stream:
                next(stream, None)
                if next(stream, None) is not None:
                    return True
        except OSError:
            continue
    return False


def save_review(project_dir, image_filename: str, decision: str | None,
                review_complete: bool, segmentation_path=None,
                annotation_path=None, use_prediction=True,
                image_path=None) -> dict:
    """Save one review and snapshot its inputs when it is complete.

    Incomplete records do not need a segmentation.  A completed record must
    have one of the explicit decisions and a displayed segmentation to
    preserve.  The latest completed save replaces that image's prior
    snapshots, which supports intentional clinician revisions.
    """
    project_dir = Path(project_dir)
    if decision not in (None, *VALID_DECISIONS):
        raise ValueError(f"Invalid scan-level decision: {decision}")
    if review_complete and decision is None:
        raise ValueError("A completed review requires a scan-level decision")

    document = load_review_document(project_dir)
    previous = document["reviews"].get(image_filename, {})
    record = {
        "decision": decision,
        "review_complete": bool(review_complete),
        "updated_at_utc": _utc_now(),
    }

    if review_complete:
        if segmentation_path is None or not Path(segmentation_path).is_file():
            raise ValueError(
                "A completed review requires the displayed segmentation")

        segmentation_size = Image.open(segmentation_path).size
        red_pixels = 0
        if annotation_path is not None and Path(annotation_path).is_file():
            annotation = np.asarray(
                Image.open(annotation_path).convert("RGBA"))
            if annotation.shape[1::-1] != segmentation_size:
                raise ValueError(
                    "Annotation and segmentation dimensions do not match: "
                    f"{annotation.shape[1::-1]} != {segmentation_size}")
            red_pixels = int(np.count_nonzero(np.logical_and(
                annotation[:, :, 0] > 0,
                annotation[:, :, 3] > 0)))

        if decision == "absent" and red_pixels:
            raise ValueError(
                "RIPL absent conflicts with red foreground marks. Erase the "
                "red marks or change the scan decision before completing "
                "the review.")
        if decision == "present" and not use_prediction and not red_pixels:
            raise ValueError(
                "Before training starts, a RIPL-present review requires red "
                "foreground. Paint every RIPL red because the random model "
                "prediction is ignored.")

        png_name = Path(image_filename).stem + ".png"
        segmentation_snapshot = (
            project_dir / "reviews" / "segmentation_snapshots" / png_name)
        _atomic_copy(Path(segmentation_path), segmentation_snapshot)
        record.update({
            "completed_at_utc": _utc_now(),
            "segmentation_snapshot": _relative_posix(
                segmentation_snapshot, project_dir),
            "segmentation_sha256": _sha256(segmentation_snapshot),
        })

        if decision == "absent":
            reconstruction_mode = DECISION_OVERRIDE_EMPTY
        elif use_prediction:
            reconstruction_mode = PREDICTION_CORRECTED
        else:
            reconstruction_mode = ANNOTATION_FOREGROUND_ONLY
        record["reconstruction_mode"] = reconstruction_mode

        annotation_snapshot = None
        if annotation_path is not None and Path(annotation_path).is_file():
            annotation_snapshot = (
                project_dir / "reviews" / "annotation_snapshots" / png_name)
            _atomic_copy(Path(annotation_path), annotation_snapshot)
            record.update({
                "annotation_snapshot": _relative_posix(
                    annotation_snapshot, project_dir),
                "annotation_sha256": _sha256(annotation_snapshot),
            })
        else:
            record["annotation_snapshot"] = None
            record["annotation_sha256"] = None

        if image_path is not None:
            if not Path(image_path).is_file():
                raise ValueError(
                    f"Reviewed source image does not exist: {image_path}")
            decision_preview = (
                project_dir / "reviews" / "decision_previews" / png_name)
            create_decision_audit_preview(
                image_path,
                segmentation_snapshot,
                annotation_snapshot,
                decision,
                decision_preview)
            record.update({
                "decision_preview": _relative_posix(
                    decision_preview, project_dir),
                "decision_preview_sha256": _sha256(decision_preview),
            })
        else:
            record["decision_preview"] = None
            record["decision_preview_sha256"] = None
    else:
        # Keep snapshot references as provenance if a previously completed
        # review is deliberately reopened, but exports ignore incomplete rows.
        for key in (
                "completed_at_utc", "segmentation_snapshot",
                "segmentation_sha256", "annotation_snapshot",
                "annotation_sha256", "reconstruction_mode",
                "decision_preview", "decision_preview_sha256"):
            if key in previous:
                record[key] = previous[key]

    document["reviews"][image_filename] = record
    _atomic_write_json(_records_path(project_dir), document)
    return record


def backfill_decision_audit_previews(project_dir, dataset_dir) -> int:
    """Create missing labeled previews for already completed reviews.

    Existing review timestamps, reconstruction modes, and immutable input
    snapshots are preserved. Hashes are verified before a snapshot is used.
    """
    project_dir = Path(project_dir)
    dataset_dir = Path(dataset_dir)
    document = load_review_document(project_dir)
    created = 0
    for filename, record in document["reviews"].items():
        if not record.get("review_complete") or record.get("decision_preview"):
            continue

        image_path = dataset_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing source image for completed review: {image_path}")
        segmentation_path = project_dir / record["segmentation_snapshot"]
        if not segmentation_path.is_file():
            raise FileNotFoundError(
                f"Missing reviewed segmentation for {filename}: "
                f"{segmentation_path}")
        if _sha256(segmentation_path) != record["segmentation_sha256"]:
            raise ValueError(
                f"Reviewed segmentation hash changed for {filename}")

        annotation_rel = record.get("annotation_snapshot")
        annotation_path = (
            project_dir / annotation_rel if annotation_rel else None)
        if annotation_path is not None:
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Missing reviewed corrections for {filename}: "
                    f"{annotation_path}")
            if _sha256(annotation_path) != record["annotation_sha256"]:
                raise ValueError(
                    f"Reviewed correction hash changed for {filename}")

        preview_path = (
            project_dir / "reviews" / "decision_previews"
            / (Path(filename).stem + ".png"))
        create_decision_audit_preview(
            image_path,
            segmentation_path,
            annotation_path,
            record["decision"],
            preview_path)
        record["decision_preview"] = _relative_posix(
            preview_path, project_dir)
        record["decision_preview_sha256"] = _sha256(preview_path)
        created += 1

    if created:
        _atomic_write_json(_records_path(project_dir), document)
    return created


def reconstruct_corrected_binary(segmentation_path,
                                 annotation_path=None) -> np.ndarray:
    """Return ``prediction - green + red`` as a 2-D boolean mask."""
    segmentation = np.asarray(Image.open(segmentation_path).convert("RGBA"))
    corrected = np.logical_and(
        segmentation[:, :, 2] > 0,
        segmentation[:, :, 3] > 0)

    if annotation_path is not None:
        annotation = np.asarray(Image.open(annotation_path).convert("RGBA"))
        if annotation.shape[:2] != corrected.shape:
            raise ValueError(
                "Annotation and segmentation dimensions do not match: "
                f"{annotation.shape[:2]} != {corrected.shape}")
        visible = annotation[:, :, 3] > 0
        foreground = np.logical_and(annotation[:, :, 0] > 0, visible)
        background = np.logical_and(annotation[:, :, 1] > 0, visible)
        corrected[background] = False
        corrected[foreground] = True

    return corrected


def reconstruct_review_binary(segmentation_path, annotation_path=None,
                              decision=None,
                              reconstruction_mode=PREDICTION_CORRECTED
                              ) -> np.ndarray:
    """Reconstruct a completed review under its recorded label-source rule."""
    segmentation = np.asarray(Image.open(segmentation_path).convert("RGBA"))
    shape = segmentation.shape[:2]

    if decision == "absent" or reconstruction_mode == DECISION_OVERRIDE_EMPTY:
        return np.zeros(shape, dtype=bool)

    if reconstruction_mode == PREDICTION_CORRECTED:
        return reconstruct_corrected_binary(
            segmentation_path, annotation_path)

    if reconstruction_mode != ANNOTATION_FOREGROUND_ONLY:
        raise ValueError(
            f"Unknown review reconstruction mode: {reconstruction_mode}")

    foreground = np.zeros(shape, dtype=bool)
    if annotation_path is None:
        return foreground
    annotation = np.asarray(Image.open(annotation_path).convert("RGBA"))
    if annotation.shape[:2] != shape:
        raise ValueError(
            "Annotation and segmentation dimensions do not match: "
            f"{annotation.shape[:2]} != {shape}")
    visible = annotation[:, :, 3] > 0
    foreground = np.logical_and(annotation[:, :, 0] > 0, visible)
    return foreground


def export_completed_reviews(project_dir, image_order=None,
                             output_dir=None) -> dict:
    """Export completed scan decisions and reconstructed binary masks."""
    project_dir = Path(project_dir)
    document = load_review_document(project_dir)
    completed = {
        filename: record
        for filename, record in document["reviews"].items()
        if record.get("review_complete")
    }
    if not completed:
        raise ValueError("No completed reviews are available to export")

    if image_order is None:
        filenames = sorted(completed, key=str.casefold)
    else:
        filenames = [name for name in image_order if name in completed]
        filenames.extend(sorted(
            set(completed) - set(filenames), key=str.casefold))

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = project_dir / "review_exports" / stamp
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Export directory already exists: {output_dir}")
    masks_dir = output_dir / "binary_masks"
    masks_dir.mkdir(parents=True)
    previews_dir = output_dir / "audit_previews"

    rows = []
    inconsistent = []
    for filename in filenames:
        record = completed[filename]
        segmentation_path = project_dir / record["segmentation_snapshot"]
        if not segmentation_path.is_file():
            raise FileNotFoundError(
                f"Missing reviewed segmentation for {filename}: "
                f"{segmentation_path}")
        if _sha256(segmentation_path) != record["segmentation_sha256"]:
            raise ValueError(
                f"Reviewed segmentation hash changed for {filename}")

        annotation_rel = record.get("annotation_snapshot")
        annotation_path = project_dir / annotation_rel if annotation_rel else None
        if annotation_path is not None:
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Missing reviewed corrections for {filename}: "
                    f"{annotation_path}")
            if _sha256(annotation_path) != record["annotation_sha256"]:
                raise ValueError(
                    f"Reviewed correction hash changed for {filename}")

        preview_rel = record.get("decision_preview")
        exported_preview_rel = ""
        exported_preview_sha256 = ""
        if preview_rel:
            preview_path = project_dir / preview_rel
            if not preview_path.is_file():
                raise FileNotFoundError(
                    f"Missing decision audit preview for {filename}: "
                    f"{preview_path}")
            if _sha256(preview_path) != record["decision_preview_sha256"]:
                raise ValueError(
                    f"Decision audit preview hash changed for {filename}")
            exported_preview = previews_dir / (Path(filename).stem + ".png")
            _atomic_copy(preview_path, exported_preview)
            exported_preview_rel = (
                f"audit_previews/{exported_preview.name}")
            exported_preview_sha256 = _sha256(exported_preview)

        reconstruction_mode = record.get(
            "reconstruction_mode", PREDICTION_CORRECTED)
        mask = reconstruct_review_binary(
            segmentation_path,
            annotation_path,
            decision=record["decision"],
            reconstruction_mode=reconstruction_mode)
        mask_name = Path(filename).stem + ".png"
        mask_path = masks_dir / mask_name
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        foreground_pixels = int(mask.sum())
        decision = record["decision"]
        decision_mask_consistent = (
            decision == "uncertain"
            or (decision == "present" and foreground_pixels > 0)
            or (decision == "absent" and foreground_pixels == 0)
        )
        if not decision_mask_consistent:
            inconsistent.append(filename)
        rows.append({
            "filename": filename,
            "decision": decision,
            "review_complete": True,
            "completed_at_utc": record.get("completed_at_utc", ""),
            "foreground_pixels": foreground_pixels,
            "decision_mask_consistent": decision_mask_consistent,
            "reconstruction_mode": reconstruction_mode,
            "prediction_used": (
                reconstruction_mode == PREDICTION_CORRECTED),
            "binary_mask": f"binary_masks/{mask_name}",
            "binary_mask_sha256": _sha256(mask_path),
            "segmentation_snapshot": record["segmentation_snapshot"],
            "segmentation_sha256": record["segmentation_sha256"],
            "annotation_snapshot": annotation_rel or "",
            "annotation_sha256": record.get("annotation_sha256") or "",
            "audit_preview": exported_preview_rel,
            "audit_preview_sha256": exported_preview_sha256,
        })

    csv_path = output_dir / "reviews.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "exported_at_utc": _utc_now(),
        "completed_review_count": len(rows),
        "decision_mask_inconsistency_count": len(inconsistent),
        "decision_mask_inconsistencies": inconsistent,
        "reconstruction": {
            PREDICTION_CORRECTED: (
                "displayed prediction - green corrections + red corrections"),
            ANNOTATION_FOREGROUND_ONLY: (
                "red foreground corrections only; pre-training prediction "
                "ignored"),
            DECISION_OVERRIDE_EMPTY: (
                "empty mask from explicit RIPL-absent decision"),
        },
    }
    _atomic_write_json(output_dir / "export_metadata.json", metadata)
    return {
        "output_dir": output_dir,
        "csv_path": csv_path,
        "mask_dir": masks_dir,
        **metadata,
    }
