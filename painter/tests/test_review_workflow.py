"""Tests for save, reopen, snapshot, and export of clinician reviews."""

import csv
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), "src", "main", "python")
sys.path.insert(0, src_dir)

from review_utils import (  # noqa: E402
    ANNOTATION_FOREGROUND_ONLY,
    DECISION_OVERRIDE_EMPTY,
    PREDICTION_CORRECTED,
    backfill_decision_audit_previews,
    completed_review_count,
    export_completed_reviews,
    incomplete_warmup_reviews,
    load_review,
    load_review_document,
    mark_training_started,
    reconstruct_corrected_binary,
    reconstruct_review_binary,
    save_review,
    should_auto_start_training,
    training_has_started,
)


def _save_segmentation(path, foreground_pixels):
    array = np.zeros((5, 6, 4), dtype=np.uint8)
    for y, x in foreground_pixels:
        array[y, x] = [0, 255, 255, 180]
    Image.fromarray(array, mode="RGBA").save(path)


def _save_annotation(path, red_pixels=(), green_pixels=()):
    array = np.zeros((5, 6, 4), dtype=np.uint8)
    for y, x in red_pixels:
        array[y, x] = [255, 0, 0, 180]
    for y, x in green_pixels:
        array[y, x] = [0, 255, 0, 180]
    Image.fromarray(array, mode="RGBA").save(path)


def test_reconstruction_applies_green_then_red(tmp_path):
    seg_path = tmp_path / "seg.png"
    annot_path = tmp_path / "annot.png"
    _save_segmentation(seg_path, [(1, 1), (2, 2)])
    _save_annotation(
        annot_path, red_pixels=[(3, 3)], green_pixels=[(1, 1)])

    mask = reconstruct_corrected_binary(seg_path, annot_path)

    assert not mask[1, 1]
    assert mask[2, 2]
    assert mask[3, 3]
    assert int(mask.sum()) == 2


def test_absent_decision_wipes_residual_prediction(tmp_path):
    seg_path = tmp_path / "seg.png"
    annot_path = tmp_path / "annot.png"
    _save_segmentation(seg_path, [(1, 1), (2, 2), (4, 5)])
    _save_annotation(annot_path, green_pixels=[(1, 1)])

    mask = reconstruct_review_binary(
        seg_path,
        annot_path,
        decision="absent",
        reconstruction_mode=DECISION_OVERRIDE_EMPTY)

    assert not mask.any()


def test_pretraining_review_uses_only_red_annotation(tmp_path):
    seg_path = tmp_path / "seg.png"
    annot_path = tmp_path / "annot.png"
    _save_segmentation(seg_path, [(1, 1), (2, 2), (4, 5)])
    _save_annotation(
        annot_path, red_pixels=[(3, 3)], green_pixels=[(1, 1)])

    mask = reconstruct_review_binary(
        seg_path,
        annot_path,
        decision="present",
        reconstruction_mode=ANNOTATION_FOREGROUND_ONLY)

    assert mask[3, 3]
    assert int(mask.sum()) == 1


def test_pretraining_present_requires_red_foreground(tmp_path):
    project_dir = tmp_path / "project"
    seg_path = project_dir / "segmentations" / "scan.png"
    seg_path.parent.mkdir(parents=True)
    _save_segmentation(seg_path, [(1, 1)])

    try:
        save_review(
            project_dir,
            "scan.png",
            "present",
            True,
            segmentation_path=seg_path,
            use_prediction=False)
    except ValueError as error:
        assert "requires red foreground" in str(error)
    else:
        raise AssertionError(
            "Pre-training present review must require clinician foreground")


def test_absent_review_rejects_red_foreground(tmp_path):
    project_dir = tmp_path / "project"
    seg_path = project_dir / "segmentations" / "scan.png"
    annot_path = project_dir / "annotations" / "train" / "scan.png"
    seg_path.parent.mkdir(parents=True)
    annot_path.parent.mkdir(parents=True)
    _save_segmentation(seg_path, [])
    _save_annotation(annot_path, red_pixels=[(2, 2)])

    try:
        save_review(
            project_dir,
            "scan.png",
            "absent",
            True,
            segmentation_path=seg_path,
            annotation_path=annot_path,
            use_prediction=False)
    except ValueError as error:
        assert "conflicts with red foreground" in str(error)
    else:
        raise AssertionError("Absent review must not discard red foreground")


def test_decision_label_is_saved_only_in_separate_audit_preview(tmp_path):
    project_dir = tmp_path / "project"
    image_path = tmp_path / "scan.png"
    seg_path = project_dir / "segmentations" / "scan.png"
    annot_path = project_dir / "annotations" / "train" / "scan.png"
    seg_path.parent.mkdir(parents=True)
    annot_path.parent.mkdir(parents=True)

    Image.fromarray(
        np.full((80, 120, 3), 90, dtype=np.uint8), mode="RGB").save(
            image_path)
    segmentation = np.zeros((80, 120, 4), dtype=np.uint8)
    segmentation[35:45, 50:70] = [0, 255, 255, 180]
    Image.fromarray(segmentation, mode="RGBA").save(seg_path)
    annotation = np.zeros((80, 120, 4), dtype=np.uint8)
    annotation[40:45, 65:75] = [255, 0, 0, 180]
    Image.fromarray(annotation, mode="RGBA").save(annot_path)
    original_annotation = annot_path.read_bytes()

    record = save_review(
        project_dir,
        "scan.png",
        "present",
        True,
        segmentation_path=seg_path,
        annotation_path=annot_path,
        use_prediction=False,
        image_path=image_path)

    preview_path = project_dir / record["decision_preview"]
    annotation_snapshot = project_dir / record["annotation_snapshot"]
    assert annot_path.read_bytes() == original_annotation
    assert annotation_snapshot.read_bytes() == original_annotation
    assert preview_path.is_file()
    assert Image.open(preview_path).size == (120, 80)
    assert hashlib.sha256(preview_path.read_bytes()).hexdigest() == (
        record["decision_preview_sha256"])

    output_dir = tmp_path / "export"
    export_completed_reviews(project_dir, output_dir=output_dir)
    with (output_dir / "reviews.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    exported_preview = output_dir / row["audit_preview"]
    assert exported_preview.is_file()
    assert exported_preview.read_bytes() == preview_path.read_bytes()


def test_existing_completed_review_can_receive_audit_preview(tmp_path):
    project_dir = tmp_path / "project"
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    image_path = dataset_dir / "scan.png"
    Image.fromarray(
        np.full((5, 6, 3), 90, dtype=np.uint8), mode="RGB").save(image_path)
    seg_path = project_dir / "segmentations" / "scan.png"
    seg_path.parent.mkdir(parents=True)
    _save_segmentation(seg_path, [])
    record = save_review(
        project_dir,
        "scan.png",
        "absent",
        True,
        segmentation_path=seg_path,
        use_prediction=False)
    completed_at = record["completed_at_utc"]
    assert record["decision_preview"] is None

    assert backfill_decision_audit_previews(project_dir, dataset_dir) == 1
    assert backfill_decision_audit_previews(project_dir, dataset_dir) == 0
    updated = load_review(project_dir, "scan.png")
    assert updated["completed_at_utc"] == completed_at
    assert (project_dir / updated["decision_preview"]).is_file()


def test_auto_start_threshold_is_ten_completed_reviews():
    assert not should_auto_start_training(9, 10)
    assert should_auto_start_training(10, 10)
    assert should_auto_start_training(11, 10)
    assert not should_auto_start_training(10, 10, has_started=True)
    assert not should_auto_start_training(10, None)


def test_warmup_requires_the_first_ten_reviews(tmp_path):
    project_dir = tmp_path / "project"
    image_order = [f"scan_{index}.png" for index in range(12)]
    for index, filename in enumerate(image_order):
        if index == 4:
            save_review(project_dir, filename, None, False)
        else:
            save_review(project_dir, filename, "absent", False)

    document = load_review_document(project_dir)
    for index in range(10):
        if index != 4:
            document["reviews"][image_order[index]]["review_complete"] = True
    records_path = project_dir / "reviews" / "review_records.json"
    records_path.write_text(json.dumps(document), encoding="utf-8")

    assert incomplete_warmup_reviews(
        project_dir, image_order, 10) == ["scan_4.png"]
    document["reviews"]["scan_4.png"]["review_complete"] = True
    records_path.write_text(json.dumps(document), encoding="utf-8")
    assert incomplete_warmup_reviews(project_dir, image_order, 10) == []


def test_training_started_marker_is_persistent(tmp_path):
    project_dir = tmp_path / "project"
    assert not training_has_started(project_dir)
    first = mark_training_started(project_dir)
    second = mark_training_started(project_dir)
    assert training_has_started(project_dir)
    assert first == second


def test_existing_training_log_marks_legacy_project_as_started(tmp_path):
    project_dir = tmp_path / "project"
    log_dir = project_dir / "logs"
    log_dir.mkdir(parents=True)
    train_log = log_dir / "2026-09-17_train.csv"
    train_log.write_text(
        "date_time,loss\n2026-09-17-09:44:49,1.0\n",
        encoding="utf-8")

    assert training_has_started(project_dir)


def test_ten_reviews_survive_reopen_and_export_from_snapshots(tmp_path):
    project_dir = tmp_path / "project"
    live_seg_dir = project_dir / "segmentations"
    annot_dir = project_dir / "annotations" / "train"
    live_seg_dir.mkdir(parents=True)
    annot_dir.mkdir(parents=True)

    image_order = []
    expected_pixels = {}
    for index in range(10):
        filename = f"scan_{index:02d}.png"
        image_order.append(filename)
        seg_path = live_seg_dir / filename
        annot_path = annot_dir / filename
        _save_segmentation(seg_path, [(1, 1)])

        if index % 3 == 0:
            decision = "absent"
            _save_annotation(annot_path, green_pixels=[(1, 1)])
            expected_pixels[filename] = 0
        elif index % 3 == 1:
            decision = "present"
            # Correct model suggestion: no correction is necessary.
            annot_path = None
            expected_pixels[filename] = 1
        else:
            decision = "uncertain"
            _save_annotation(annot_path, red_pixels=[(3, 3)])
            expected_pixels[filename] = 2

        save_review(
            project_dir, filename, decision, True,
            segmentation_path=seg_path, annotation_path=annot_path,
            use_prediction=True)

    assert completed_review_count(project_dir) == 10
    reopened = load_review_document(project_dir)
    assert len(reopened["reviews"]) == 10
    assert reopened["reviews"][image_order[0]]["decision"] == "absent"
    assert reopened["reviews"][image_order[0]]["review_complete"] is True

    # A later live prediction must not alter the completed review export.
    _save_segmentation(live_seg_dir / image_order[1], [(4, 5), (3, 4)])

    export_dir = tmp_path / "export"
    result = export_completed_reviews(
        project_dir, image_order=image_order, output_dir=export_dir)

    assert result["completed_review_count"] == 10
    assert result["decision_mask_inconsistency_count"] == 0
    with (export_dir / "reviews.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["filename"] for row in rows] == image_order
    for row in rows:
        mask = np.asarray(Image.open(export_dir / row["binary_mask"]))
        assert int(np.count_nonzero(mask)) == expected_pixels[row["filename"]]
        expected_mode = (
            DECISION_OVERRIDE_EMPTY
            if row["decision"] == "absent" else PREDICTION_CORRECTED)
        assert row["reconstruction_mode"] == expected_mode

    metadata = json.loads(
        (export_dir / "export_metadata.json").read_text(encoding="utf-8"))
    assert metadata["reconstruction"][PREDICTION_CORRECTED] == (
        "displayed prediction - green corrections + red corrections")


def test_export_records_annotation_only_pretraining_mode(tmp_path):
    project_dir = tmp_path / "project"
    seg_path = project_dir / "segmentations" / "scan.png"
    annot_path = project_dir / "annotations" / "train" / "scan.png"
    seg_path.parent.mkdir(parents=True)
    annot_path.parent.mkdir(parents=True)
    _save_segmentation(seg_path, [(1, 1), (2, 2)])
    _save_annotation(annot_path, red_pixels=[(3, 3)])
    save_review(
        project_dir,
        "scan.png",
        "present",
        True,
        segmentation_path=seg_path,
        annotation_path=annot_path,
        use_prediction=False)

    output_dir = tmp_path / "export"
    export_completed_reviews(project_dir, output_dir=output_dir)
    with (output_dir / "reviews.csv").open(
            "r", encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    mask = np.asarray(Image.open(output_dir / row["binary_mask"]))
    assert row["reconstruction_mode"] == ANNOTATION_FOREGROUND_ONLY
    assert row["prediction_used"] == "False"
    assert int(np.count_nonzero(mask)) == 1


def test_incomplete_review_is_distinct_and_not_exported(tmp_path):
    project_dir = tmp_path / "project"
    save_review(project_dir, "unreviewed.png", None, False)
    save_review(project_dir, "started.png", "uncertain", False)

    assert load_review(project_dir, "unreviewed.png") == {
        "decision": None,
        "review_complete": False,
        "updated_at_utc": load_review(
            project_dir, "unreviewed.png")["updated_at_utc"],
    }
    assert completed_review_count(project_dir) == 0

    try:
        export_completed_reviews(project_dir, output_dir=tmp_path / "export")
    except ValueError as error:
        assert "No completed reviews" in str(error)
    else:
        raise AssertionError("Incomplete reviews must not be exported")
