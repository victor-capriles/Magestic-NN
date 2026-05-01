"""Build reproducible train, validation, and test splits for modeling.

This script creates a stratified split from the RDKit descriptor dataset so later
feature filtering, scaling, and model fitting can be learned from the training
subset only. The full row contents are preserved in each output file so the
notebook can load the split datasets directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from common import resolve_output_dir, resolve_path, write_csv
from preprocessing_common import (
    METADATA_COLUMNS,
    SPLIT_NAMES,
    TARGET_COLUMN,
    get_descriptor_columns,
    read_dataset_rows,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DATASET_PATH = (
    SCRIPT_DIR
    / "rdkit_descriptor_dataset"
    / "classification_rdkit_descriptor_dataset.csv"
)
DEFAULT_OUTPUT_DIR_NAME = "stratified_split_dataset"
DEFAULT_TRAIN_FRACTION = 0.8
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1
DEFAULT_RANDOM_SEED = 42


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> None:
    """Validate the requested split fractions.

    Args:
        train_fraction: Fraction assigned to the training split.
        validation_fraction: Fraction assigned to the validation split.
        test_fraction: Fraction assigned to the test split.

    Raises:
        ValueError: If the fractions are invalid.
    """

    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    if any(value <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be greater than zero.")

    total_fraction = sum(fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "Train, validation, and test fractions must sum to 1.0. "
            f"Received {total_fraction}."
        )


def _compute_split_counts(
    group_size: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, int]:
    """Convert split fractions into deterministic integer counts.

    Args:
        group_size: Number of rows within one activity class.
        train_fraction: Fraction assigned to the training split.
        validation_fraction: Fraction assigned to the validation split.
        test_fraction: Fraction assigned to the test split.

    Returns:
        Per-split row counts that sum exactly to group_size.
    """

    raw_counts = {
        "train": group_size * train_fraction,
        "validation": group_size * validation_fraction,
        "test": group_size * test_fraction,
    }
    split_counts = {
        split_name: math.floor(raw_count)
        for split_name, raw_count in raw_counts.items()
    }

    remaining_rows = group_size - sum(split_counts.values())
    split_priority = sorted(
        SPLIT_NAMES,
        key=lambda split_name: (
            raw_counts[split_name] - split_counts[split_name],
            split_name != "train",
            split_name,
        ),
        reverse=True,
    )
    for split_name in split_priority[:remaining_rows]:
        split_counts[split_name] += 1

    return split_counts


def _stratified_split_rows(
    rows: list[dict[str, str]],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, list[dict[str, str]]]:
    """Split rows by activity while preserving class balance.

    Args:
        rows: Input dataset rows.
        train_fraction: Fraction assigned to the training split.
        validation_fraction: Fraction assigned to the validation split.
        test_fraction: Fraction assigned to the test split.
        random_seed: Seed used for deterministic shuffling.

    Returns:
        Mapping from split name to its rows.
    """

    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row[TARGET_COLUMN]].append(row)

    split_rows = {split_name: [] for split_name in SPLIT_NAMES}
    random_generator = random.Random(random_seed)

    for label, label_rows in sorted(rows_by_label.items()):
        shuffled_rows = list(label_rows)
        random_generator.shuffle(shuffled_rows)

        split_counts = _compute_split_counts(
            len(shuffled_rows),
            train_fraction,
            validation_fraction,
            test_fraction,
        )
        train_end = split_counts["train"]
        validation_end = train_end + split_counts["validation"]

        split_rows["train"].extend(shuffled_rows[:train_end])
        split_rows["validation"].extend(shuffled_rows[train_end:validation_end])
        split_rows["test"].extend(shuffled_rows[validation_end:])

    for split_name in SPLIT_NAMES:
        random_generator.shuffle(split_rows[split_name])

    return split_rows


def _summarize_split_rows(
    split_rows: dict[str, list[dict[str, str]]],
    fieldnames: list[str],
    input_dataset_path: Path,
    output_dir: Path,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, object]:
    """Create a machine-readable summary for the generated splits."""

    descriptor_columns = get_descriptor_columns(fieldnames)
    split_counts = {
        split_name: len(split_rows[split_name]) for split_name in SPLIT_NAMES
    }
    class_distribution = {
        split_name: {
            "active": sum(
                1 for row in split_rows[split_name] if row[TARGET_COLUMN] == "1"
            ),
            "inactive": sum(
                1 for row in split_rows[split_name] if row[TARGET_COLUMN] == "0"
            ),
        }
        for split_name in SPLIT_NAMES
    }

    return {
        "input_dataset": str(input_dataset_path),
        "output_dir": str(output_dir),
        "random_seed": random_seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "total_rows": sum(split_counts.values()),
        "descriptor_column_count": len(descriptor_columns),
        "split_counts": split_counts,
        "class_distribution": class_distribution,
    }


def build_stratified_split_dataset(
    input_dataset_path: Path | None = None,
    output_dir: Path | None = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, object]:
    """Build reproducible train, validation, and test split files.

    Args:
        input_dataset_path: Optional input RDKit descriptor dataset path.
        output_dir: Optional split output directory.
        train_fraction: Training split fraction.
        validation_fraction: Validation split fraction.
        test_fraction: Test split fraction.
        random_seed: Seed used for deterministic shuffling.

    Returns:
        Summary of the generated split outputs.
    """

    _validate_split_fractions(train_fraction, validation_fraction, test_fraction)

    resolved_input_dataset_path = input_dataset_path or DEFAULT_INPUT_DATASET_PATH
    resolved_output_dir = output_dir or resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
    )

    fieldnames, rows = read_dataset_rows(resolved_input_dataset_path)
    split_rows = _stratified_split_rows(
        rows,
        train_fraction,
        validation_fraction,
        test_fraction,
        random_seed,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(resolved_output_dir / "train_dataset.csv", fieldnames, split_rows["train"])
    write_csv(
        resolved_output_dir / "validation_dataset.csv",
        fieldnames,
        split_rows["validation"],
    )
    write_csv(resolved_output_dir / "test_dataset.csv", fieldnames, split_rows["test"])

    summary = _summarize_split_rows(
        split_rows,
        fieldnames,
        resolved_input_dataset_path,
        resolved_output_dir,
        train_fraction,
        validation_fraction,
        test_fraction,
        random_seed,
    )
    with (resolved_output_dir / "split_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for dataset splitting."""

    parser = argparse.ArgumentParser(
        description="Build a stratified train, validation, and test split dataset."
    )
    parser.add_argument(
        "--input-dataset",
        help=(
            "Path to classification_rdkit_descriptor_dataset.csv. Relative paths are "
            "resolved from this script directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory to write the split datasets and summary. Relative paths are "
            "resolved from this script directory."
        ),
    )
    parser.add_argument(
        "--suffix",
        help=(
            "Optional suffix appended to the output directory name so a run can "
            "write alongside existing artifacts."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help="Training fraction. Defaults to 0.8.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Validation fraction. Defaults to 0.1.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help="Test fraction. Defaults to 0.1.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for deterministic shuffling. Defaults to 42.",
    )
    return parser


def main() -> None:
    """Run the dataset splitting workflow and print the summary."""

    args = _build_argument_parser().parse_args()
    input_dataset_path = (
        None if args.input_dataset is None else resolve_path(SCRIPT_DIR, args.input_dataset)
    )
    output_dir = resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    summary = build_stratified_split_dataset(
        input_dataset_path=input_dataset_path,
        output_dir=output_dir,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        random_seed=args.random_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()