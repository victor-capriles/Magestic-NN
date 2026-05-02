"""Filter highly correlated descriptors using the training split only.

This script learns correlation-based descriptor drops from the training dataset
 after zero-variance filtering. The same kept-column set is then applied to the
validation and test datasets so held-out rows do not influence preprocessing.
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_INPUT_DIR = SCRIPT_DIR / "variance_filtered_dataset"
DEFAULT_OUTPUT_DIR_NAME = "correlation_filtered_dataset"
DEFAULT_CORRELATION_THRESHOLD = 0.8


def _get_train_descriptor_values(
    train_rows: list[dict[str, str]],
    descriptor_columns: list[str],
) -> dict[str, list[float]]:
    """Convert training descriptor columns to float lists."""

    return {
        column_name: [float(row[column_name]) for row in train_rows]
        for column_name in descriptor_columns
    }


def _compute_abs_pearson_correlation(
    left_values: list[float],
    right_values: list[float],
) -> float:
    """Compute absolute Pearson correlation between two descriptor vectors."""

    if len(left_values) != len(right_values):
        raise ValueError("Descriptor vectors must have the same length.")
    if not left_values:
        raise ValueError("Correlation cannot be computed from empty descriptor vectors.")

    left_mean = sum(left_values) / len(left_values)
    right_mean = sum(right_values) / len(right_values)

    numerator = 0.0
    left_sum_squares = 0.0
    right_sum_squares = 0.0
    for left_value, right_value in zip(left_values, right_values, strict=True):
        left_delta = left_value - left_mean
        right_delta = right_value - right_mean
        numerator += left_delta * right_delta
        left_sum_squares += left_delta * left_delta
        right_sum_squares += right_delta * right_delta

    denominator = math.sqrt(left_sum_squares * right_sum_squares)
    if denominator == 0.0:
        raise ValueError(
            "Correlation filtering requires non-constant training descriptors. "
            "Run zero-variance filtering first or use the variance-filtered input."
        )

    return abs(numerator / denominator)


def _select_descriptor_columns(
    descriptor_columns: list[str],
    train_descriptor_values: dict[str, list[float]],
    correlation_threshold: float,
) -> tuple[list[str], list[dict[str, object]]]:
    """Select kept descriptors and record dropped correlated descriptors."""

    kept_descriptor_columns: list[str] = []
    dropped_descriptor_rows: list[dict[str, object]] = []

    for column_name in descriptor_columns:
        drop_record: dict[str, object] | None = None
        for kept_column_name in kept_descriptor_columns:
            absolute_correlation = _compute_abs_pearson_correlation(
                train_descriptor_values[kept_column_name],
                train_descriptor_values[column_name],
            )
            if absolute_correlation > correlation_threshold:
                drop_record = {
                    "descriptor_name": column_name,
                    "retained_descriptor_name": kept_column_name,
                    "absolute_train_correlation": absolute_correlation,
                    "filter_reason": "absolute_correlation_above_threshold",
                }
                break

        if drop_record is None:
            kept_descriptor_columns.append(column_name)
            continue

        dropped_descriptor_rows.append(drop_record)

    return kept_descriptor_columns, dropped_descriptor_rows


def build_correlation_filtered_dataset(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> dict[str, object]:
    """Build correlation-filtered train, validation, and test datasets.

    Args:
        input_dir: Optional directory containing variance-filtered split files.
        output_dir: Optional output directory for filtered datasets.
        correlation_threshold: Absolute Pearson threshold fitted on train only.

    Returns:
        Summary of the generated filtered outputs.
    """

    if correlation_threshold < 0.0 or correlation_threshold > 1.0:
        raise ValueError("The correlation threshold must be between 0.0 and 1.0.")

    resolved_input_dir = input_dir or DEFAULT_INPUT_DIR
    resolved_output_dir = output_dir or resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
    )

    fieldnames, split_rows = read_split_datasets(resolved_input_dir)
    descriptor_columns = get_descriptor_columns(fieldnames)
    train_descriptor_values = _get_train_descriptor_values(
        split_rows["train"],
        descriptor_columns,
    )
    kept_descriptor_columns, dropped_descriptor_rows = _select_descriptor_columns(
        descriptor_columns,
        train_descriptor_values,
        correlation_threshold,
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
        resolved_output_dir / "dropped_correlated_descriptor_columns.csv",
        [
            "descriptor_name",
            "retained_descriptor_name",
            "absolute_train_correlation",
            "filter_reason",
        ],
        dropped_descriptor_rows,
    )

    summary = {
        "input_dir": str(resolved_input_dir),
        "output_dir": str(resolved_output_dir),
        "correlation_threshold": correlation_threshold,
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
    with (resolved_output_dir / "correlation_filter_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for correlation filtering."""

    parser = argparse.ArgumentParser(
        description="Build correlation-filtered train, validation, and test datasets."
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
        "--correlation-threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
        help=(
            "Absolute Pearson threshold fitted on the training split only. "
            "Defaults to 0.8."
        ),
    )
    return parser


def main() -> None:
    """Run the correlation filtering workflow and print the summary."""

    args = _build_argument_parser().parse_args()
    input_dir = None if args.input_dir is None else resolve_path(SCRIPT_DIR, args.input_dir)
    output_dir = resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    summary = build_correlation_filtered_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        correlation_threshold=args.correlation_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()