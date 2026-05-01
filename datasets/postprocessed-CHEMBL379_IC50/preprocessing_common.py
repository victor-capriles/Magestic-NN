"""Shared helpers for split-based preprocessing stages."""

from __future__ import annotations

import csv
from pathlib import Path


TARGET_COLUMN = "activity"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FILE_NAMES = {
    "train": "train_dataset.csv",
    "validation": "validation_dataset.csv",
    "test": "test_dataset.csv",
}
METADATA_COLUMNS = {
    "representative_molecule_chembl_id",
    "smiles",
    "canonical_smiles",
    "activity",
    "threshold_nm",
    "source_record_count",
    "source_molecule_chembl_ids",
    "source_relations",
    "source_values_nm",
    "label_sources",
    "target_chembl_id",
}


def read_dataset_rows(dataset_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV dataset into fieldnames and row dictionaries.

    Args:
        dataset_path: Path to a CSV dataset.

    Returns:
        Header columns and dataset rows.
    """

    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"The dataset at {dataset_path} does not contain a header.")
        rows = list(reader)
    return reader.fieldnames, rows


def read_split_datasets(
    input_dir: Path,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Read the train, validation, and test split files.

    Args:
        input_dir: Directory containing split CSV files.

    Returns:
        Shared fieldnames and per-split rows.

    Raises:
        ValueError: If the split files do not share the same header.
    """

    shared_fieldnames: list[str] | None = None
    split_rows: dict[str, list[dict[str, str]]] = {}

    for split_name, file_name in SPLIT_FILE_NAMES.items():
        fieldnames, rows = read_dataset_rows(input_dir / file_name)
        if shared_fieldnames is None:
            shared_fieldnames = fieldnames
        elif fieldnames != shared_fieldnames:
            raise ValueError(
                "All split files must share the same header order. "
                f"Mismatch found in {input_dir / file_name}."
            )
        split_rows[split_name] = rows

    if shared_fieldnames is None:
        raise ValueError(f"No split files were found in {input_dir}.")

    return shared_fieldnames, split_rows


def get_descriptor_columns(fieldnames: list[str]) -> list[str]:
    """Return descriptor columns in stable file order."""

    return [column_name for column_name in fieldnames if column_name not in METADATA_COLUMNS]


def project_rows(
    rows: list[dict[str, str]],
    kept_fieldnames: list[str],
) -> list[dict[str, str]]:
    """Project row dictionaries onto the kept field set."""

    return [
        {field_name: row[field_name] for field_name in kept_fieldnames}
        for row in rows
    ]