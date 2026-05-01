"""Filter zero-variance descriptors using the training split only.

This script learns the descriptor drop list from the training dataset so feature
selection does not inspect validation or test rows. The same kept-column set is
then applied to all three split files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import resolve_output_dir, resolve_path, write_csv
from preprocessing_common import (
    METADATA_COLUMNS,
    SPLIT_FILE_NAMES,
    get_descriptor_columns,
    project_rows,
    read_split_datasets,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "stratified_split_dataset"
DEFAULT_OUTPUT_DIR_NAME = "variance_filtered_dataset"
DEFAULT_VARIANCE_THRESHOLD = 0.0


def _compute_population_variance(values: list[float]) -> float:
    """Compute population variance for one descriptor column."""

    if not values:
        raise ValueError("Variance cannot be computed from an empty value list.")

    mean_value = 0.0
    sum_squared_deltas = 0.0
    for index, value in enumerate(values, start=1):
        delta = value - mean_value
        mean_value += delta / index
        sum_squared_deltas += delta * (value - mean_value)

    return max(0.0, sum_squared_deltas / len(values))


def _compute_train_descriptor_variances(
    train_rows: list[dict[str, str]],
    descriptor_columns: list[str],
) -> dict[str, float]:
    """Compute training-set variances for each descriptor column."""

    descriptor_variances: dict[str, float] = {}
    for column_name in descriptor_columns:
        column_values = [float(row[column_name]) for row in train_rows]
        descriptor_variances[column_name] = _compute_population_variance(column_values)
    return descriptor_variances


def _select_descriptor_columns(
    descriptor_variances: dict[str, float],
    descriptor_columns: list[str],
    variance_threshold: float,
) -> tuple[list[str], list[dict[str, object]]]:
    """Select kept descriptors and build audit rows for dropped ones."""

    kept_descriptor_columns: list[str] = []
    dropped_descriptor_rows: list[dict[str, object]] = []

    for column_name in descriptor_columns:
        variance_value = descriptor_variances[column_name]
        if variance_value <= variance_threshold:
            dropped_descriptor_rows.append(
                {
                    "descriptor_name": column_name,
                    "train_variance": variance_value,
                    "filter_reason": "variance_at_or_below_threshold",
                }
            )
            continue
        kept_descriptor_columns.append(column_name)

    return kept_descriptor_columns, dropped_descriptor_rows


def build_variance_filtered_dataset(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> dict[str, object]:
    """Build variance-filtered train, validation, and test datasets.

    Args:
        input_dir: Optional directory containing train/validation/test split files.
        output_dir: Optional output directory for filtered datasets.
        variance_threshold: Variance threshold applied on the training split only.

    Returns:
        Summary of the generated filtered outputs.
    """

    if variance_threshold < 0.0:
        raise ValueError("The variance threshold must be zero or greater.")

    resolved_input_dir = input_dir or DEFAULT_INPUT_DIR
    resolved_output_dir = output_dir or resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
    )

    fieldnames, split_rows = read_split_datasets(resolved_input_dir)
    descriptor_columns = get_descriptor_columns(fieldnames)
    descriptor_variances = _compute_train_descriptor_variances(
        split_rows["train"],
        descriptor_columns,
    )
    kept_descriptor_columns, dropped_descriptor_rows = _select_descriptor_columns(
        descriptor_variances,
        descriptor_columns,
        variance_threshold,
    )
    kept_fieldnames = [
        field_name
        for field_name in fieldnames
        if field_name in METADATA_COLUMNS or field_name in kept_descriptor_columns
    ]

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, file_name in SPLIT_FILE_NAMES.items():
        write_csv(
            resolved_output_dir / file_name,
            kept_fieldnames,
            project_rows(split_rows[split_name], kept_fieldnames),
        )

    write_csv(
        resolved_output_dir / "dropped_descriptor_columns.csv",
        ["descriptor_name", "train_variance", "filter_reason"],
        dropped_descriptor_rows,
    )

    summary = {
        "input_dir": str(resolved_input_dir),
        "output_dir": str(resolved_output_dir),
        "variance_threshold": variance_threshold,
        "input_descriptor_count": len(descriptor_columns),
        "kept_descriptor_count": len(kept_descriptor_columns),
        "dropped_descriptor_count": len(dropped_descriptor_rows),
        "dropped_descriptors": [
            row["descriptor_name"] for row in dropped_descriptor_rows
        ],
        "split_row_counts": {
            split_name: len(rows) for split_name, rows in split_rows.items()
        },
    }
    with (resolved_output_dir / "variance_filter_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for variance filtering."""

    parser = argparse.ArgumentParser(
        description="Build variance-filtered train, validation, and test datasets."
    )
    parser.add_argument(
        "--input-dir",
        help=(
            "Directory containing train_dataset.csv, validation_dataset.csv, and "
            "test_dataset.csv. Relative paths are resolved from this script directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory to write the filtered datasets and audit files. Relative paths "
            "are resolved from this script directory."
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
        "--variance-threshold",
        type=float,
        default=DEFAULT_VARIANCE_THRESHOLD,
        help="Variance threshold fitted on the training split only. Defaults to 0.0.",
    )
    return parser


def main() -> None:
    """Run the variance filtering workflow and print the summary."""

    args = _build_argument_parser().parse_args()
    input_dir = None if args.input_dir is None else resolve_path(SCRIPT_DIR, args.input_dir)
    output_dir = resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    summary = build_variance_filtered_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        variance_threshold=args.variance_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()