"""Shared helpers for CHEMBL379 postprocessing scripts."""

from __future__ import annotations

import csv
from pathlib import Path


def find_repo_root(start_path: Path) -> Path:
    """Locate the repository root from a script path.

    Args:
        start_path: Directory from which to begin searching upward.

    Returns:
        The repository root path.

    Raises:
        FileNotFoundError: If no repository root markers are found.
    """

    for candidate in (start_path, *start_path.parents):
        has_repo_markers = (
            (candidate / ".github").exists() or (candidate / ".git").exists()
        )
        if has_repo_markers and (candidate / "datasets").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the repository root from {start_path}."
    )


def resolve_path(base_dir: Path, path_value: str | Path) -> Path:
    """Resolve a path value relative to a base directory when needed.

    Args:
        base_dir: Base directory used for relative paths.
        path_value: Absolute or relative path value.

    Returns:
        Resolved absolute-or-workspace-relative path.
    """

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def resolve_output_dir(
    base_dir: Path,
    default_dir_name: str,
    output_dir: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Resolve an output directory, optionally applying a run suffix.

    Args:
        base_dir: Base directory used for default or relative output paths.
        default_dir_name: Default output directory name.
        output_dir: Optional absolute or relative output directory override.
        suffix: Optional suffix appended to the final directory name.

    Returns:
        Resolved output directory path.

    Raises:
        ValueError: If the suffix contains only whitespace.
    """

    resolved_dir = (
        base_dir / default_dir_name
        if output_dir is None
        else resolve_path(base_dir, output_dir)
    )
    if suffix is None:
        return resolved_dir

    normalized_suffix = suffix.strip()
    if not normalized_suffix:
        raise ValueError(
            "The output suffix must contain at least one non-space character."
        )
    return resolved_dir.with_name(f"{resolved_dir.name}_{normalized_suffix}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    """Write rows to a CSV file with a stable column order.

    Args:
        path: Destination CSV path.
        fieldnames: Ordered column names.
        rows: CSV-ready row dictionaries.
    """

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)