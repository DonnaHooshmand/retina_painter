import argparse
import csv
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import prepare_nested_annotation_datasets as nested_datasets  # noqa: E402


def make_args(source, output, exclude_dir, *, seed="2026"):
    return argparse.Namespace(
        source=source,
        output=output,
        seed=seed,
        sizes=[2, 4, 6],
        exclude_dir=[exclude_dir],
        plan_only=False,
    )


def test_seeded_subsets_are_reproducible_and_nested(tmp_path):
    source = tmp_path / "source"
    excluded = tmp_path / "excluded"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    source.mkdir()
    excluded.mkdir()

    for index in range(10):
        (source / f"patient{index:02d}_OD_0001.png").write_bytes(
            f"image-{index}".encode())
    (excluded / "patient03_OD_0001.png").write_bytes(b"old annotation")

    selected, metadata = nested_datasets.build_plan(
        make_args(source, first_output, excluded))
    selected_again, _ = nested_datasets.build_plan(
        make_args(source, second_output, excluded))
    assert [path.name for path in selected] == [
        path.name for path in selected_again]
    assert "patient03_OD_0001.png" not in {path.name for path in selected}
    assert metadata["excluded_matching_source_images"] == 1

    nested_datasets.write_dataset(selected, metadata)
    names_by_size = {
        size: {path.name for path in nested_datasets.image_files(
            first_output / str(size))}
        for size in (2, 4, 6)
    }
    assert len(names_by_size[2]) == 2
    assert len(names_by_size[4]) == 4
    assert len(names_by_size[6]) == 6
    assert names_by_size[2] < names_by_size[4] < names_by_size[6]

    with (first_output / "manifest.csv").open(
            encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["rank"]) for row in rows] == list(range(1, 7))
    assert json.loads(
        (first_output / "dataset_metadata.json").read_text(
            encoding="utf-8"))["seed"] == "2026"


def test_existing_unexpected_image_is_never_deleted(tmp_path):
    source = tmp_path / "source"
    excluded = tmp_path / "excluded"
    output = tmp_path / "output"
    source.mkdir()
    excluded.mkdir()
    for index in range(6):
        (source / f"patient{index:02d}_OS_0001.png").write_bytes(
            f"image-{index}".encode())

    selected, metadata = nested_datasets.build_plan(
        make_args(source, output, excluded))
    (output / "2").mkdir(parents=True)
    unexpected = output / "2" / "unexpected.png"
    unexpected.write_bytes(b"preserve me")

    with pytest.raises(RuntimeError, match="unexpected image"):
        nested_datasets.write_dataset(selected, metadata)
    assert unexpected.read_bytes() == b"preserve me"
