"""Tests for the time and painted-pixel record of an annotation session."""

import csv
import json
import os
import sys

import numpy as np
import pytest
from PIL import Image


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), "src", "main", "python")
sys.path.insert(0, src_dir)

import annotation_effort as effort  # noqa: E402
from interaction_time import (  # noqa: E402
    events_from_client_log,
    get_annot_duration_s,
    screen_time_per_fname_s,
)
from review_utils import (  # noqa: E402
    save_review,
    write_whole_scan_background,
)

HEIGHT, WIDTH = 40, 60
PIXELS = HEIGHT * WIDTH
NAMES = ["P01_a.png", "P01_b.png", "P02_a.png", "P03_a.png"]


def _strokes(path, red=0, green=0):
    """An annotation with ``red`` lesion and ``green`` background pixels."""
    pixels = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
    flat = pixels.reshape(-1, 4)
    flat[:red] = (255, 0, 0, 180)
    flat[red:red + green] = (0, 255, 0, 180)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, "RGBA").save(path)
    return path


def _project(tmp_path, absent_as_background=True):
    project = tmp_path / "project"
    for folder in ("annotations/train", "annotations/val", "segmentations",
                   "models", "logs", "reviews"):
        (project / folder).mkdir(parents=True)
    (project / "project.seg_proj").write_text(json.dumps({
        "name": "project", "file_names": NAMES,
        "absent_scans_as_background": absent_as_background}),
        encoding="utf-8")
    for name in NAMES:
        segmentation = np.zeros((HEIGHT, WIDTH, 4), np.uint8)
        segmentation[:5, :5] = (0, 255, 255, 180)
        Image.fromarray(segmentation, "RGBA").save(
            project / "segmentations" / name)
    return project


def _log(project, lines):
    """Write a client log from ``(seconds, message)`` pairs."""
    text = "".join(f"2026-01-01 00:00:00.000000,{float(seconds)},{message}\n"
                   for seconds, message in lines)
    (project / "logs" / "client.csv").write_text(text)
    return project / "logs" / "client.csv"


def _review(project, name, decision, annotation=None):
    return save_review(project, name, decision, True,
                       segmentation_path=project / "segmentations" / name,
                       annotation_path=annotation)


def _absent_scan(project, name, side="train", green=0):
    """Review a scan as absent the way the painter does with the option on."""
    annotation = project / "annotations" / side / name
    if green:
        _strokes(annotation, green=green)
    _review(project, name, "absent", annotation if green else None)
    write_whole_scan_background(annotation, WIDTH, HEIGHT)
    return annotation


def test_log_lines_that_name_no_file_are_skipped(tmp_path):
    # "automatic_start_training" names no file. It used to raise IndexError,
    # which stopped the metrics plot and export once training had started.
    project = _project(tmp_path)
    log = _log(project, [
        (0, "update_file_end,fname:P01_a.png"),
        (5, "automatic_start_training,completed_reviews:10"),
        (6, "mouse_press,fname:P01_a.png,x:1.0,y:2.0,brush_size:25"),
        (7, "mouse_release,fname:P01_a.png,x:1.0,y:2.0,brush_size:25"),
    ])
    with open(log, "a") as stream:
        stream.write("\nnot an event line\n2026-01-01,oops,mouse_press,"
                     "fname:P01_a.png\n")

    events = events_from_client_log(log)

    assert [event.name for event in events] == [
        "update_file_end", "mouse_press", "mouse_release"]
    assert {event.fname for event in events} == {"P01_a"}


def test_screen_time_includes_looking_without_painting():
    project_events = events_from_lines([
        (0, "update_file_end,fname:P01_a.png"),
        (30, "update_file_end,fname:P01_b.png"),
        (40, "mouse_press,fname:P01_b.png"),
        (42, "mouse_release,fname:P01_b.png"),
        (50, "save_annotation,fname:P01_b.png"),
        (50.5, "update_file_end,fname:P02_a.png"),
    ])

    screen = screen_time_per_fname_s(project_events)

    assert screen["P01_a"] == (30.0, 1, 0.0)
    assert screen["P01_b"] == (20.5, 1, 0.0)
    # Painting time alone would report nothing for the scan left unpainted.
    assert get_annot_duration_s(project_events, "P01_a.png") == (0, 0)
    assert get_annot_duration_s(project_events, "P01_b.png") == (20.0, 1)


def test_a_long_gap_is_time_away_not_session_time():
    events = events_from_lines([
        (0, "update_file_end,fname:P01_a.png"),
        (10, "mouse_press,fname:P01_a.png"),
        (12, "mouse_release,fname:P01_a.png"),
        (1012, "update_file_end,fname:P01_b.png"),
        (1020, "update_file_end,fname:P02_a.png"),
    ])

    screen = screen_time_per_fname_s(events, pause_s=120)

    assert screen["P01_a"] == (12.0, 1, 1000.0)
    assert screen["P01_b"] == (8.0, 1, 0.0)
    # With no limit the break would count as time on the scan.
    assert screen_time_per_fname_s(events, pause_s=1e9)["P01_a"][0] == 1012.0


def test_returning_to_a_scan_adds_to_its_time():
    events = events_from_lines([
        (0, "update_file_end,fname:P01_a.png"),
        (20, "update_file_end,fname:P01_b.png"),
        (30, "update_file_end,fname:P01_a.png"),
        (45, "update_file_end,fname:P01_b.png"),
        (50, "save_annotation,fname:P01_b.png"),
    ])

    screen = screen_time_per_fname_s(events)

    assert screen["P01_a"] == (35.0, 2, 0.0)
    assert screen["P01_b"] == (15.0, 2, 0.0)


def events_from_lines(lines):
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "client.csv")
        with open(path, "w") as stream:
            for seconds, message in lines:
                stream.write(
                    f"2026-01-01 00:00:00.000000,{float(seconds)},{message}\n")
        return events_from_client_log(path)


def test_stroke_pixels_are_counted_as_the_trainer_reads_them(tmp_path):
    path = _strokes(tmp_path / "strokes.png", red=12, green=7)

    assert effort.stroke_pixel_counts(path) == (12, 7)
    assert effort.stroke_pixel_counts(None) == (0, 0)
    assert effort.stroke_pixel_counts(tmp_path / "missing.png") == (0, 0)


def test_an_absent_scan_reports_the_strokes_painted_not_the_whole_scan(
        tmp_path):
    project = _project(tmp_path, absent_as_background=True)
    painted = _absent_scan(project, NAMES[0], green=30)
    unpainted = _absent_scan(project, NAMES[1], side="val")
    present = _strokes(project / "annotations/train" / NAMES[2],
                       red=12, green=7)
    _review(project, NAMES[2], "present", present)
    # The files the trainer reads are background everywhere.
    assert effort.stroke_pixel_counts(painted) == (0, PIXELS)
    assert effort.stroke_pixel_counts(unpainted) == (0, PIXELS)

    rows = {row["file_name"]: row for row in effort.effort_rows(project)}

    assert (rows[NAMES[0]]["painted_fg"], rows[NAMES[0]]["painted_bg"]) == (
        0, 30)
    assert rows[NAMES[0]]["strokes_source"] == "review_snapshot"
    assert (rows[NAMES[1]]["painted_fg"], rows[NAMES[1]]["painted_bg"]) == (
        0, 0)
    assert rows[NAMES[1]]["strokes_source"] == "review_snapshot"
    assert rows[NAMES[1]]["side"] == "val"
    assert (rows[NAMES[2]]["painted_fg"], rows[NAMES[2]]["painted_bg"]) == (
        12, 7)
    assert rows[NAMES[2]]["strokes_source"] == "annotation"
    assert effort.summarise(list(rows.values()))["scans_counted_as_clean"] == 2


def test_projects_without_the_option_count_the_annotation_file(tmp_path):
    project = _project(tmp_path, absent_as_background=False)
    annotation = _strokes(project / "annotations/train" / NAMES[0], green=30)
    _review(project, NAMES[0], "absent", annotation)

    row, = effort.effort_rows(project)

    assert (row["painted_fg"], row["painted_bg"]) == (0, 30)
    assert row["strokes_source"] == "annotation"


def test_rows_follow_the_order_viewed_and_accumulate(tmp_path):
    project = _project(tmp_path)
    # Viewed out of project order: the third scan first, then the first.
    _strokes(project / "annotations/train" / NAMES[2], red=10, green=5)
    _review(project, NAMES[2], "present",
            project / "annotations/train" / NAMES[2])
    _absent_scan(project, NAMES[0], green=20)
    _log(project, [
        (100, "update_file_end,fname:" + NAMES[2]),
        (104, "mouse_press,fname:" + NAMES[2]),
        (110, "mouse_release,fname:" + NAMES[2]),
        (118, "save_annotation,fname:" + NAMES[2]),
        (120, "update_file_end,fname:" + NAMES[0]),
        (125, "mouse_press,fname:" + NAMES[0]),
        (127, "mouse_release,fname:" + NAMES[0]),
        (140, "save_annotation,fname:" + NAMES[0]),
        (141, "automatic_start_training,completed_reviews:10"),
        (141, "update_file_end,fname:" + NAMES[1]),
        (171, "update_file_end,fname:" + NAMES[1]),
    ])

    rows = effort.effort_rows(project)

    # The fourth scan was never opened, reviewed or painted.
    assert [row["file_name"] for row in rows] == [
        NAMES[2], NAMES[0], NAMES[1]]
    assert [row["viewing_order"] for row in rows] == [1, 2, 3]
    assert [row["session_s"] for row in rows] == [0.0, 20.0, 41.0]
    assert [row["screen_time_s"] for row in rows] == [20.0, 21.0, 30.0]
    assert [row["annot_duration_s"] for row in rows] == [18.0, 20.0, 0]
    assert [row["clicks"] for row in rows] == [1, 1, 0]
    assert [row["cumulative_painted_pixels"] for row in rows] == [15, 35, 35]
    assert [row["cumulative_screen_time_s"] for row in rows] == [
        20.0, 41.0, 71.0]
    assert [row["cumulative_annot_duration_s"] for row in rows] == [
        18.0, 38.0, 38.0]

    summary = effort.summarise(rows)
    assert summary["scans_viewed"] == 3
    assert summary["scans_painted"] == 2
    assert summary["scans_with_lesion_strokes"] == 1
    assert summary["completed_reviews"] == {"present": 1, "absent": 1}
    assert summary["session_minutes"] == pytest.approx(71 / 60)
    assert summary["seconds_per_scan_median"] == 21.0
    assert summary["painted_pixels_foreground"] == 10
    assert summary["painted_pixels_background"] == 25

    csv_path = effort.write_effort_csv(
        rows, project / "metrics" / "annotation_effort.csv")
    with open(csv_path, newline="", encoding="utf-8") as stream:
        written = list(csv.DictReader(stream))
    assert [row["file_name"] for row in written] == [
        NAMES[2], NAMES[0], NAMES[1]]
    assert written[1]["cumulative_painted_pixels"] == "35"
    assert set(written[0]) == set(effort.CSV_COLUMNS)


def test_a_project_without_a_log_still_reports_painted_pixels(tmp_path):
    project = _project(tmp_path)
    _strokes(project / "annotations/val" / NAMES[1], red=4)
    _strokes(project / "annotations/train" / NAMES[0], green=9)

    rows = effort.effort_rows(project)

    assert [row["file_name"] for row in rows] == [NAMES[0], NAMES[1]]
    assert [row["painted_fg"] + row["painted_bg"] for row in rows] == [9, 4]
    assert all(row["screen_time_s"] == 0 and row["session_s"] is None
               for row in rows)


def test_the_figure_and_totals_are_written_without_file_names(
        tmp_path, capsys):
    pytest.importorskip("matplotlib")
    project = _project(tmp_path)
    _strokes(project / "annotations/train" / NAMES[0], red=10, green=5)
    _log(project, [
        (0, "update_file_end,fname:" + NAMES[0]),
        (4, "mouse_press,fname:" + NAMES[0]),
        (9, "mouse_release,fname:" + NAMES[0]),
        (20, "update_file_end,fname:" + NAMES[1]),
        (35, "save_annotation,fname:" + NAMES[1]),
    ])

    effort.main(["--project", str(project), "--label", "session"])

    assert (project / "metrics" / "annotation_effort.png").stat().st_size > 0
    assert (project / "metrics" / "annotation_effort.csv").is_file()
    printed = capsys.readouterr().out
    assert "scans_viewed: 2" in printed
    assert "painted_pixels_foreground: 10" in printed
    assert not any(name in printed for name in NAMES)
