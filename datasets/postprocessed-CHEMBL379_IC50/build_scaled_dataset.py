"""Scale descriptor columns using training-set statistics only.

This script fits z-score scaling on the training split after correlation
filtering, then applies the same descriptor-wise mean and standard deviation to
validation and test. Metadata columns and the target label are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import resolve_output_dir, resolve_path, write_csv
from preprocessing_common import (
    get_descriptor_columns,
    read_split_datasets,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "correlation_filtered_dataset"
DEFAULT_OUTPUT_DIR_NAME = "scaled_dataset"
SCALING_METHOD = "zscore"


def _compute_population_mean(values: list[float]) -> float:
    """Compute the population mean for a descriptor column."""

    if not values:
        raise ValueError("Mean cannot be computed from an empty value list.")
    return sum(values) / len(values)


def _compute_population_std(values: list[float], mean_value: float) -> float:
    """Compute population standard deviation for a descriptor column."""

    if not values:
        raise ValueError("Standard deviation cannot be computed from an empty value list.")

    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(0.0, variance))


def _compute_train_scaling_parameters(
    train_rows: list[dict[str, str]],
    descriptor_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Compute descriptor-wise scaling parameters from the training split."""

    scaling_parameters: dict[str, dict[str, float]] = {}
    for column_name in descriptor_columns:
        column_values = [float(row[column_name]) for row in train_rows]
        mean_value = _compute_population_mean(column_values)
        std_value = _compute_population_std(column_values, mean_value)
        if std_value == 0.0:
            raise ValueError(
                "Scaling requires non-constant training descriptors. "
                f"Found zero standard deviation for {column_name}."
            )
        scaling_parameters[column_name] = {
            "train_mean": mean_value,
            "train_std": std_value,
        }
    return scaling_parameters


def _scale_rows(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    descriptor_columns: list[str],
    scaling_parameters: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    """Scale descriptor columns while preserving metadata fields."""

    scaled_rows: list[dict[str, object]] = []
    descriptor_column_set = set(descriptor_columns)

    for row in rows:
        scaled_row: dict[str, object] = {}
        for field_name in fieldnames:
            if field_name not in descriptor_column_set:
                scaled_row[field_name] = row[field_name]
                continue

            value = float(row[field_name])
            mean_value = scaling_parameters[field_name]["train_mean"]
            std_value = scaling_parameters[field_name]["train_std"]
            scaled_row[field_name] = (value - mean_value) / std_value
        scaled_rows.append(scaled_row)

    return scaled_rows


def build_scaled_dataset(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, object]:
    """Build scaled train, validation, and test datasets.

    Args:
        input_dir: Optional directory containing correlation-filtered split files.
        output_dir: Optional output directory for scaled datasets.

    Returns:
        Summary of the generated scaled outputs.
    """

    resolved_input_dir = input_dir or DEFAULT_INPUT_DIR
    resolved_output_dir = output_dir or resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
    )

    fieldnames, split_rows = read_split_datasets(resolved_input_dir)
    descriptor_columns = get_descriptor_columns(fieldnames)
    scaling_parameters = _compute_train_scaling_parameters(
        split_rows["train"],
        descriptor_columns,
    )

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        write_csv(
            resolved_output_dir / f"{split_name}_dataset.csv",
            fieldnames,
            _scale_rows(rows, fieldnames, descriptor_columns, scaling_parameters),
        )

    scaling_parameter_rows = [
        {
            "descriptor_name": column_name,
            "train_mean": scaling_parameters[column_name]["train_mean"],
            "train_std": scaling_parameters[column_name]["train_std"],
        }
        for column_name in descriptor_columns
    ]
    write_csv(
        resolved_output_dir / "scaling_parameters.csv",
        ["descriptor_name", "train_mean", "train_std"],
        scaling_parameter_rows,
    )

    summary = {
        "input_dir": str(resolved_input_dir),
        "output_dir": str(resolved_output_dir),
        "scaling_method": SCALING_METHOD,
        "scaled_descriptor_count": len(descriptor_columns),
        "split_row_counts": {
            split_name: len(rows) for split_name, rows in split_rows.items()
        },
    }
    with (resolved_output_dir / "scaling_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for scaling."""

    parser = argparse.ArgumentParser(
        description="Build scaled train, validation, and test datasets."
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
            "Directory to write the scaled datasets and scaling statistics. Relative "
            "paths are resolved from this script directory."
        ),
    )
    parser.add_argument(
        "--suffix",
        help=(
            "Optional suffix appended to the output directory name so a run can "
            "write alongside existing artifacts."
        ),
    )
    return parser


def main() -> None:
    """Run the scaling workflow and print the summary."""

    args = _build_argument_parser().parse_args()
    input_dir = None if args.input_dir is None else resolve_path(SCRIPT_DIR, args.input_dir)
    output_dir = resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    summary = build_scaled_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
