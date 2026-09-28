"""Verify datasets or newly generated results against the frozen references."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("data", "main", "defense"))
    parser.add_argument("--actual", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Verify only the cases present in --actual.",
    )
    return parser.parse_args()


def verify_data() -> None:
    manifest = PROJECT_ROOT / "data" / "MANIFEST.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    failures = []
    for record in records:
        path = PROJECT_ROOT / "data" / record["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != int(record["bytes"]) or digest != record["sha256"]:
            failures.append(record["file"])
    if failures:
        raise SystemExit(f"Dataset verification failed: {failures}")
    print(f"Dataset verification passed: {len(records)} files")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def verify_results(kind: str, actual: Path, tolerance: float, allow_partial: bool) -> None:
    if kind == "main":
        reference = PROJECT_ROOT / "results" / "reference" / "main_tvnia_reference.csv"
        keys = ("dataset", "victim_model")
        metrics = ("clean_accuracy", "poisoned_accuracy")
        exact_fields = (
            "injected_nodes",
            "degree_per_injected_node",
            "total_degree_used",
            "total_degree_budget",
            "budget_valid",
        )
    else:
        reference = PROJECT_ROOT / "results" / "reference" / "defense_tvnia_reference.csv"
        keys = ("dataset", "defense_model")
        metrics = ("clean_defense_accuracy", "poisoned_defense_accuracy")
        exact_fields = (
            "oracle_victim",
            "structure_iterations",
            "injected_nodes",
            "degree_per_injected_node",
            "total_degree_used",
            "total_degree_budget",
            "budget_valid",
        )
    expected = {tuple(row[key] for key in keys): row for row in _read(reference)}
    observed = {
        tuple(row[key] for key in keys): row
        for row in _read(actual)
        if row.get("status", "OK") == "OK"
    }
    errors = []
    rows_to_check = observed if allow_partial else expected
    for key in rows_to_check:
        expected_row = expected.get(key)
        if expected_row is None:
            errors.append(f"unexpected {key}")
            continue
        if key not in observed:
            errors.append(f"missing {key}")
            continue
        for metric in metrics:
            delta = abs(float(observed[key][metric]) - float(expected_row[metric]))
            if delta > tolerance:
                errors.append(f"{key} {metric}: delta={delta:.9g}")
        for field in exact_fields:
            if observed[key][field] != expected_row[field]:
                errors.append(
                    f"{key} {field}: observed={observed[key][field]!r}, "
                    f"expected={expected_row[field]!r}"
                )
    if errors:
        print("Verification differences:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"{kind.capitalize()} result verification passed: {len(rows_to_check)} cases")


def main() -> None:
    args = parse_args()
    if args.kind == "data":
        verify_data()
        return
    default = PROJECT_ROOT / "results" / args.kind / f"tvnia_{args.kind}_results.csv"
    verify_results(args.kind, args.actual or default, args.tolerance, args.allow_partial)


if __name__ == "__main__":
    main()
