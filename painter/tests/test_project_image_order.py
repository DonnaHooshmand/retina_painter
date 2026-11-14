"""Tests for reproducible image ordering across matched projects."""

import os
import random
import sys
import csv


test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src', 'main', 'python')
sys.path.insert(0, src_dir)

from project_order import (fixed_train_val_split, manifest_file_order,
                           seeded_file_order)


def test_same_seed_and_files_produce_same_order():
    fnames = [f'scan_{i:02d}.png' for i in range(20)]

    first = seeded_file_order(fnames, seed=42)
    second = seeded_file_order(reversed(fnames), seed=42)

    assert first == second
    assert sorted(first) == sorted(fnames)


def test_different_seed_changes_order():
    fnames = [f'scan_{i:02d}.png' for i in range(20)]

    assert seeded_file_order(fnames, seed=1) != seeded_file_order(
        fnames, seed=2)


def test_seeded_order_does_not_mutate_input_or_global_rng():
    fnames = ['c.png', 'a.png', 'b.png']
    original = list(fnames)
    random.seed(123)
    state_before = random.getstate()

    seeded_file_order(fnames, seed=0)

    assert fnames == original
    assert random.getstate() == state_before


def test_fixed_split_matches_five_train_to_one_validation():
    ordered = [f'scan_{i:02d}.png' for i in range(18)]

    train, val = fixed_train_val_split(ordered)

    assert val == [ordered[0], ordered[6], ordered[12]]
    assert len(train) == 15
    assert set(train).isdisjoint(val)
    assert set(train + val) == set(ordered)


def test_seed_fixes_both_navigation_order_and_split():
    fnames = [f'scan_{i:02d}.png' for i in range(30)]

    first_order = seeded_file_order(fnames, seed=42)
    second_order = seeded_file_order(reversed(fnames), seed=42)

    assert first_order == second_order
    assert fixed_train_val_split(first_order) == fixed_train_val_split(
        second_order)


def test_nested_manifest_makes_smaller_dataset_exact_prefix(tmp_path):
    root = tmp_path / 'nested'
    smaller = root / '150'
    larger = root / '250'
    smaller.mkdir(parents=True)
    larger.mkdir()
    ranked = [f'scan_{index:03d}.png' for index in range(1, 251)]
    with (root / 'manifest.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'filename'])
        writer.writeheader()
        for rank, filename in enumerate(ranked, start=1):
            writer.writerow({'rank': rank, 'filename': filename})

    small_order, small_manifest = manifest_file_order(
        smaller, reversed(ranked[:150]))
    large_order, large_manifest = manifest_file_order(
        larger, reversed(ranked[:250]))

    assert small_order == ranked[:150]
    assert large_order[:150] == small_order
    assert small_manifest == large_manifest == root / 'manifest.csv'


def test_manifest_refuses_dataset_image_without_rank(tmp_path):
    dataset = tmp_path / 'nested' / '150'
    dataset.mkdir(parents=True)
    with (dataset.parent / 'manifest.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'filename'])
        writer.writeheader()
        writer.writerow({'rank': 1, 'filename': 'known.png'})

    try:
        manifest_file_order(dataset, ['known.png', 'missing.png'])
    except ValueError as error:
        assert 'missing from manifest' in str(error)
    else:
        raise AssertionError('An incomplete manifest must not silently fall back')


# -- patient-aware split ----------------------------------------------------

from project_order import (filename_parts, grouping_summary,  # noqa: E402
                           patient_group_numbers,
                           patient_grouped_train_val_split,
                           patient_ids_from_csv,
                           patient_ids_from_filename_part,
                           suggest_patient_part)


def _patient_scans(patients=30, seed=3):
    """Synthetic names: 1 to 9 scans per patient, patient first."""
    rng = random.Random(seed)
    names = [f'P{patient:03d}_O{"DS"[scan % 2]}_2024-01-{scan + 1:02d}_{scan}.png'
             for patient in range(patients)
             for scan in range(rng.randint(1, 9))]
    return names


def test_filename_parts_split_at_the_chosen_separators():
    name = 'A17_OD_2024-03-01_012.png'

    assert filename_parts(name) == ['A17', 'OD', '2024', '03', '01', '012']
    assert filename_parts(name, '_') == ['A17', 'OD', '2024-03-01', '012']
    assert filename_parts(name, '') == ['A17_OD_2024-03-01_012']
    assert filename_parts('__A17__OD.png') == ['A17', 'OD']


def test_suggested_part_is_first_that_groups_images():
    assert suggest_patient_part(_patient_scans()) == 0
    # A constant prefix and a per-image counter are both skipped.
    assert suggest_patient_part(
        ['OCT_P1_a.png', 'OCT_P1_b.png', 'OCT_P2_a.png']) == 1
    assert suggest_patient_part(
        ['001_P1.png', '002_P1.png', '003_P2.png']) == 1
    # Nothing groups: every name unique in every part, or nothing to read.
    assert suggest_patient_part(['a.png', 'b.png', 'c.png']) is None
    assert suggest_patient_part([]) is None


def test_patient_ids_from_filename_part_reports_names_that_do_not_fit():
    patient_of, unmatched = patient_ids_from_filename_part(
        ['p1_od.png', 'P1_os.png', 'lonely.png'], part=1)

    assert unmatched == ['lonely.png']
    assert patient_of == {'p1_od.png': 'od', 'P1_os.png': 'os'}
    # Patients compare without regard to case.
    by_first, _ = patient_ids_from_filename_part(
        ['p1_od.png', 'P1_os.png'], part=0)
    assert by_first['p1_od.png'] == by_first['P1_os.png']


def test_patient_split_never_puts_a_patient_on_both_sides():
    names = _patient_scans()
    patient_of, unmatched = patient_ids_from_filename_part(names, 0)
    assert not unmatched
    ordered = seeded_file_order(names, seed=7)

    train, val = patient_grouped_train_val_split(ordered, patient_of)

    assert sorted(train + val) == sorted(names)
    assert {patient_of[n] for n in train}.isdisjoint(
        {patient_of[n] for n in val})
    # Both keep navigation order, and the first image is validation as before.
    assert train == [n for n in ordered if n in set(train)]
    assert val[0] == ordered[0]
    # About one image in six is validation, overall and early in a session.
    assert 0.10 <= len(val) / len(names) <= 0.25
    early = ordered[:36]
    assert 3 <= sum(n in set(val) for n in early) <= 12


def test_patient_split_equals_per_image_split_when_images_are_independent():
    ordered = [f'scan_{i:02d}.png' for i in range(18)]
    patient_of = {name: name for name in ordered}

    assert patient_grouped_train_val_split(
        ordered, patient_of) == fixed_train_val_split(ordered)


def test_patient_split_is_stable_across_nested_prefixes():
    names = _patient_scans(patients=60)
    patient_of, _ = patient_ids_from_filename_part(names, 0)
    ordered = seeded_file_order(names, seed=11)

    small_train, small_val = patient_grouped_train_val_split(
        ordered[:150], patient_of)
    large_train, large_val = patient_grouped_train_val_split(
        ordered, patient_of)

    assert set(small_train) <= set(large_train)
    assert set(small_val) <= set(large_val)


def test_patient_split_refuses_an_image_without_a_patient():
    try:
        patient_grouped_train_val_split(['a.png', 'b.png'], {'a.png': 'p'})
    except ValueError as error:
        assert 'no patient' in str(error)
    else:
        raise AssertionError('An ungrouped image must not be split silently')


def test_patient_list_csv_must_cover_every_image(tmp_path):
    path = tmp_path / 'patients.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(
            stream, fieldnames=['filename', 'patient_id', 'note'])
        writer.writeheader()
        writer.writerow({'filename': 'A.PNG', 'patient_id': 'p1', 'note': ''})
        writer.writerow({'filename': 'b.png', 'patient_id': 'P1', 'note': ''})
        writer.writerow({'filename': 'c.png', 'patient_id': '', 'note': ''})

    patient_of = patient_ids_from_csv(path, ['a.png', 'b.png'])
    assert patient_of['a.png'] == patient_of['b.png'] == 'p1'

    for names, expected in ((['a.png', 'c.png'], '1 of 2 images'),
                            (['a.png', 'zz.png'], '1 of 2 images')):
        try:
            patient_ids_from_csv(path, names)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError('Every image needs a patient')

    path.write_text('name,patient\na.png,p1\n', encoding='utf-8')
    try:
        patient_ids_from_csv(path, ['a.png'])
    except ValueError as error:
        assert 'filename and patient_id' in str(error)
    else:
        raise AssertionError('Unrecognised columns must be refused')


def test_summary_and_group_numbers_carry_counts_not_identifiers():
    names = ['p1_a.png', 'p1_b.png', 'p1_c.png', 'p2_a.png', 'p3_a.png']
    patient_of, _ = patient_ids_from_filename_part(names, 0)

    assert grouping_summary(names, patient_of) == {
        'images': 5, 'ungrouped_images': 0, 'patients': 3,
        'median_images_per_patient': 1, 'largest_patient_images': 3}
    order = ['p2_a.png', 'p1_a.png', 'p3_a.png', 'p1_b.png', 'p1_c.png']
    assert patient_group_numbers(order, patient_of) == [1, 2, 3, 2, 2]
