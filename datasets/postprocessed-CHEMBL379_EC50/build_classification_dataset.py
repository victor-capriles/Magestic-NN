"""Build a labeled, deduplicated classification dataset for CHEMBL379 EC50 data.

This script applies the thesis curation policy recorded in Memory.md:
- filter to CHEMBL379 EC50 measurements in nM
- derive the activity label using a threshold computed from exact rows only
- exclude ambiguous censored rows
- collapse duplicate SMILES groups only when their resolved labels agree
- exclude duplicate SMILES groups with conflicting labels

The outputs include the final structure-level classification dataset plus audit files
for excluded and collapsed records so the curation remains reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from common import find_repo_root, resolve_output_dir, write_csv


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(SCRIPT_DIR)
RAW_DATASET_PATH = REPO_ROOT / "datasets" / "CHEMBL379_EC50_AllDesc.csv"
DEFAULT_OUTPUT_DIR_NAME = "classification_curation_dataset"
TARGET_CHEMBL_ID = "CHEMBL379"
STANDARD_TYPE = "EC50"
STANDARD_UNITS = "nM"
EXACT_RELATION = "'='"
RIGHT_CENSORED_RELATIONS = {"'>'", "'>='"}
LEFT_CENSORED_RELATIONS = {"'<'", "'<='"}


@dataclass(frozen=True)
class AssayRow:
    """Store the core assay fields required for classification curation.

    Attributes:
        molecule_chembl_id: ChEMBL molecule identifier.
        smiles: SMILES representation used as the structure key.
        standard_relation: Relation qualifier for the assay value.
        standard_value_nm: Assay value in nM.
        standard_units: Unit for the assay value.
        target_chembl_id: ChEMBL target identifier.
    """

    molecule_chembl_id: str
    smiles: str
    standard_relation: str
    standard_value_nm: float
    standard_units: str
    target_chembl_id: str


@dataclass(frozen=True)
class LabeledRow:
    """Represent a curation-ready assay row with a resolved binary label.

    Attributes:
        assay_row: Original assay row.
        activity: Binary activity label where 1 is active and 0 is inactive.
        label_source: Short explanation for how the label was resolved.
    """

    assay_row: AssayRow
    activity: int
    label_source: str


def _read_assay_rows(dataset_path: Path) -> list[AssayRow]:
    """Load the raw CHEMBL379 export and keep only the scope-relevant fields.

    Args:
        dataset_path: Path to the tab-delimited source dataset.

    Returns:
        A list of filtered assay rows in the thesis scope.
    """

    rows: list[AssayRow] = []
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            if raw_row["Standard Type"] != STANDARD_TYPE:
                continue
            if raw_row["Standard Units"] != STANDARD_UNITS:
                continue
            if raw_row["Target ChEMBL ID"] != TARGET_CHEMBL_ID:
                continue
            rows.append(
                AssayRow(
                    molecule_chembl_id=raw_row["Molecule ChEMBL ID"],
                    smiles=raw_row["Smiles"],
                    standard_relation=raw_row["Standard Relation"],
                    standard_value_nm=float(raw_row["Standard Value"]),
                    standard_units=raw_row["Standard Units"],
                    target_chembl_id=raw_row["Target ChEMBL ID"],
                )
            )
    return rows


def _compute_threshold_nm(rows: list[AssayRow]) -> float:
    """Compute the classification threshold from exact EC50 rows only.

    Args:
        rows: Scope-filtered assay rows.

    Returns:
        The median EC50 threshold in nM.

    Raises:
        ValueError: If no exact EC50 rows are available.
    """

    exact_values_nm = [
        row.standard_value_nm for row in rows if row.standard_relation == EXACT_RELATION
    ]
    if not exact_values_nm:
        raise ValueError("No exact EC50 rows were found for threshold computation.")
    return float(median(exact_values_nm))


def _resolve_activity_label(row: AssayRow, threshold_nm: float) -> LabeledRow | None:
    """Resolve a binary activity label using the censor-aware policy.

    Args:
        row: Assay row to label.
        threshold_nm: Activity cutoff in nM.

    Returns:
        A labeled row when the label is logically resolvable, otherwise None.
    """

    relation = row.standard_relation
    value_nm = row.standard_value_nm
    if relation == EXACT_RELATION:
        activity = int(value_nm <= threshold_nm)
        return LabeledRow(row, activity, "exact")
    if relation in RIGHT_CENSORED_RELATIONS and value_nm >= threshold_nm:
        return LabeledRow(row, 0, "right_censored_inactive")
    if relation in LEFT_CENSORED_RELATIONS and value_nm <= threshold_nm:
        return LabeledRow(row, 1, "left_censored_active")
    return None


def _group_by_smiles(rows: list[LabeledRow]) -> dict[str, list[LabeledRow]]:
    """Group labeled rows by their SMILES string.

    Args:
        rows: Resolved labeled rows.

    Returns:
        Mapping from SMILES to the rows that share it.
    """

    grouped_rows: dict[str, list[LabeledRow]] = {}
    for row in rows:
        grouped_rows.setdefault(row.assay_row.smiles, []).append(row)
    return grouped_rows


def _select_representative_row(group: list[LabeledRow]) -> LabeledRow:
    """Pick a deterministic representative for a concordant SMILES group.

    Args:
        group: Labeled rows with the same SMILES and the same resolved label.

    Returns:
        One deterministic representative row.
    """

    return sorted(
        group,
        key=lambda row: (
            row.assay_row.standard_value_nm,
            row.assay_row.molecule_chembl_id,
        ),
    )[0]


def build_classification_dataset(
    output_dir: Path | None = None,
) -> dict[str, int | float | str]:
    """Build the labeled, deduplicated classification outputs.

    Args:
        output_dir: Optional directory for the generated outputs.

    Returns:
        Summary statistics describing the curation result.
    """

    resolved_output_dir = output_dir or resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
    )
    assay_rows = _read_assay_rows(RAW_DATASET_PATH)
    threshold_nm = _compute_threshold_nm(assay_rows)

    labeled_rows: list[LabeledRow] = []
    ambiguous_rows: list[AssayRow] = []
    for row in assay_rows:
        labeled_row = _resolve_activity_label(row, threshold_nm)
        if labeled_row is None:
            ambiguous_rows.append(row)
            continue
        labeled_rows.append(labeled_row)

    grouped_rows = _group_by_smiles(labeled_rows)
    final_rows: list[dict[str, object]] = []
    collapsed_duplicates: list[dict[str, object]] = []
    conflicting_duplicates: list[dict[str, object]] = []

    for smiles, group in sorted(grouped_rows.items()):
        labels = {row.activity for row in group}
        if len(labels) > 1:
            conflicting_duplicates.append(
                {
                    "smiles": smiles,
                    "group_size": len(group),
                    "molecule_chembl_ids": ";".join(
                        row.assay_row.molecule_chembl_id for row in group
                    ),
                    "standard_relations": ";".join(
                        row.assay_row.standard_relation for row in group
                    ),
                    "standard_values_nm": ";".join(
                        str(row.assay_row.standard_value_nm) for row in group
                    ),
                    "resolved_labels": ";".join(str(row.activity) for row in group),
                }
            )
            continue

        representative_row = _select_representative_row(group)
        unique_label = next(iter(labels))
        final_rows.append(
            {
                "smiles": smiles,
                "activity": unique_label,
                "threshold_nm": threshold_nm,
                "source_record_count": len(group),
                "representative_molecule_chembl_id": (
                    representative_row.assay_row.molecule_chembl_id
                ),
                "source_molecule_chembl_ids": ";".join(
                    row.assay_row.molecule_chembl_id for row in group
                ),
                "source_relations": ";".join(
                    row.assay_row.standard_relation for row in group
                ),
                "source_values_nm": ";".join(
                    str(row.assay_row.standard_value_nm) for row in group
                ),
                "label_sources": ";".join(row.label_source for row in group),
                "target_chembl_id": representative_row.assay_row.target_chembl_id,
            }
        )

        if len(group) > 1:
            collapsed_duplicates.append(
                {
                    "smiles": smiles,
                    "group_size": len(group),
                    "activity": unique_label,
                    "representative_molecule_chembl_id": (
                        representative_row.assay_row.molecule_chembl_id
                    ),
                    "source_molecule_chembl_ids": ";".join(
                        row.assay_row.molecule_chembl_id for row in group
                    ),
                    "source_values_nm": ";".join(
                        str(row.assay_row.standard_value_nm) for row in group
                    ),
                }
            )

    ambiguous_rows_output = [
        {
            "molecule_chembl_id": row.molecule_chembl_id,
            "smiles": row.smiles,
            "standard_relation": row.standard_relation,
            "standard_value_nm": row.standard_value_nm,
            "standard_units": row.standard_units,
            "target_chembl_id": row.target_chembl_id,
            "exclusion_reason": "ambiguous_censored_label",
        }
        for row in ambiguous_rows
    ]

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        resolved_output_dir / "classification_base_dataset.csv",
        [
            "smiles",
            "activity",
            "threshold_nm",
            "source_record_count",
            "representative_molecule_chembl_id",
            "source_molecule_chembl_ids",
            "source_relations",
            "source_values_nm",
            "label_sources",
            "target_chembl_id",
        ],
        final_rows,
    )
    write_csv(
        resolved_output_dir / "collapsed_concordant_duplicates.csv",
        [
            "smiles",
            "group_size",
            "activity",
            "representative_molecule_chembl_id",
            "source_molecule_chembl_ids",
            "source_values_nm",
        ],
        collapsed_duplicates,
    )
    write_csv(
        resolved_output_dir / "excluded_conflicting_duplicates.csv",
        [
            "smiles",
            "group_size",
            "molecule_chembl_ids",
            "standard_relations",
            "standard_values_nm",
            "resolved_labels",
        ],
        conflicting_duplicates,
    )
    write_csv(
        resolved_output_dir / "excluded_ambiguous_censored.csv",
        [
            "molecule_chembl_id",
            "smiles",
            "standard_relation",
            "standard_value_nm",
            "standard_units",
            "target_chembl_id",
            "exclusion_reason",
        ],
        ambiguous_rows_output,
    )

    summary = {
        "input_rows": len(assay_rows),
        "threshold_nm": threshold_nm,
        "labeled_rows": len(labeled_rows),
        "ambiguous_censored_rows": len(ambiguous_rows),
        "collapsed_duplicate_groups": len(collapsed_duplicates),
        "conflicting_duplicate_groups": len(conflicting_duplicates),
        "final_structure_rows": len(final_rows),
        "final_active_rows": sum(int(row["activity"] == 1) for row in final_rows),
        "final_inactive_rows": sum(int(row["activity"] == 0) for row in final_rows),
        "output_dir": str(resolved_output_dir),
    }
    with (resolved_output_dir / "classification_curation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    return summary


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for classification curation."""

    parser = argparse.ArgumentParser(
        description="Build the CHEMBL379 classification curation dataset."
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Directory to write the dataset and audit files. Relative paths are "
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
    return parser


def main() -> None:
    """Execute the dataset curation workflow and print a short summary."""

    args = _build_argument_parser().parse_args()
    output_dir = resolve_output_dir(
        SCRIPT_DIR,
        DEFAULT_OUTPUT_DIR_NAME,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    summary = build_classification_dataset(output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()