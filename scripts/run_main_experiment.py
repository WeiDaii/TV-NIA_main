"""Run the six-dataset, three-victim TV-NIA main experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tvnia.data import DATASET_IDS, DATASET_DISPLAY_NAMES, canonical_dataset_id
from tvnia.experiment import construct_poisoned_dataset, resolve_device, set_seed
from tvnia.experiment_config import load_protocol
from tvnia.models import VICTIM_NAMES, build_victim, canonical_victim_name, train_victim


FIELDS = (
    "dataset",
    "victim_model",
    "seed",
    "injection_ratio",
    "clean_accuracy",
    "poisoned_accuracy",
    "injected_nodes",
    "degree_per_injected_node",
    "total_degree_used",
    "total_degree_budget",
    "budget_valid",
    "tvnia_config",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "final_no_seek_exper_3.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "main" / "tvnia_main_results.csv")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_IDS))
    parser.add_argument("--victims", nargs="+", default=list(VICTIM_NAMES))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def completed_cases(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["dataset"], row["victim_model"])
            for row in csv.DictReader(stream)
            if row.get("status") == "OK"
        }


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    device = resolve_device(args.device or protocol["device"])
    finished = completed_cases(args.output)
    training = protocol["victim_training"]
    for requested_dataset in args.datasets:
        dataset_id = canonical_dataset_id(requested_dataset)
        for requested_victim in args.victims:
            victim = canonical_victim_name(requested_victim)
            key = (dataset_id, victim)
            if key in finished:
                print(f"[skip] {dataset_id}/{victim}", flush=True)
                continue
            row = {
                "dataset": dataset_id,
                "victim_model": victim,
                "seed": protocol["seed"],
                "injection_ratio": protocol["tvnia"]["shared"]["injection_ratio"],
                "status": "ERROR",
            }
            try:
                print(f"[run] {DATASET_DISPLAY_NAMES[dataset_id]}/{victim}", flush=True)
                (
                    dataset,
                    clean_result,
                    poisoned_graph,
                    poisoned_features,
                    poisoned_labels,
                    attack_metadata,
                    tvnia_config,
                ) = construct_poisoned_dataset(
                    dataset_id,
                    victim,
                    protocol,
                    args.data_root,
                    device,
                    training["clean_epochs"],
                )
                set_seed(protocol["seed"])
                poisoned_victim = build_victim(
                    victim,
                    poisoned_features.size(1),
                    training["hidden_dim"],
                    int(poisoned_labels.max()) + 1,
                    training["dropout"],
                    device,
                )
                poisoned_result = train_victim(
                    poisoned_victim,
                    poisoned_graph,
                    poisoned_features,
                    poisoned_labels,
                    dataset.train_indices,
                    dataset.validation_indices,
                    dataset.test_indices,
                    epochs=training["poison_epochs"],
                    patience=training["patience"],
                    learning_rate=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                )
                row.update(
                    clean_accuracy=clean_result.test_accuracy,
                    poisoned_accuracy=poisoned_result.test_accuracy,
                    injected_nodes=attack_metadata["injected_nodes"],
                    degree_per_injected_node=attack_metadata["degree_per_injected_node"],
                    total_degree_used=attack_metadata["total_degree_used"],
                    total_degree_budget=attack_metadata["total_degree_budget"],
                    budget_valid=attack_metadata["budget_valid"],
                    tvnia_config=json.dumps(tvnia_config, sort_keys=True),
                    status="OK",
                )
                print(
                    f"[result] clean={row['clean_accuracy']:.6f} poisoned={row['poisoned_accuracy']:.6f}",
                    flush=True,
                )
            except Exception as error:
                row["error"] = repr(error)
                print(f"[error] {dataset_id}/{victim}: {error!r}", flush=True)
            append_row(args.output, row)


if __name__ == "__main__":
    main()

