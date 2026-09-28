"""Evaluate TV-NIA against the four defense models reported in the paper."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tvnia.data import DATASET_IDS, canonical_dataset_id
from tvnia.defenses import DEFENSE_NAMES, train_defense
from tvnia.experiment import construct_poisoned_dataset, resolve_device, set_seed
from tvnia.experiment_config import load_protocol


FIELDS = (
    "dataset",
    "oracle_victim",
    "defense_model",
    "seed",
    "injection_ratio",
    "structure_iterations",
    "clean_defense_accuracy",
    "poisoned_defense_accuracy",
    "clean_epochs_ran",
    "poisoned_epochs_ran",
    "injected_nodes",
    "degree_per_injected_node",
    "total_degree_used",
    "total_degree_budget",
    "budget_valid",
    "attack_metadata",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "final_no_seek_exper_3.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "defense" / "tvnia_defense_results.csv")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_IDS))
    parser.add_argument("--defenses", nargs="+", default=list(DEFENSE_NAMES))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def completed_cases(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            (row["dataset"], row["defense_model"])
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
    defense_parameters = protocol["defense_training"]
    oracle_victim = defense_parameters["oracle_victim"]
    finished = completed_cases(args.output)
    for requested_dataset in args.datasets:
        dataset_id = canonical_dataset_id(requested_dataset)
        missing = [name.lower() for name in args.defenses if (dataset_id, name.lower()) not in finished]
        if not missing:
            print(f"[skip] {dataset_id}: all defenses complete", flush=True)
            continue
        print(f"[attack] {dataset_id}/{oracle_victim}", flush=True)
        (
            dataset,
            _,
            poisoned_graph,
            poisoned_features,
            poisoned_labels,
            attack_metadata,
            _,
        ) = construct_poisoned_dataset(
            dataset_id,
            oracle_victim,
            protocol,
            args.data_root,
            device,
            defense_parameters["clean_epochs"],
        )
        for defense in missing:
            row = {
                "dataset": dataset_id,
                "oracle_victim": oracle_victim,
                "defense_model": defense,
                "seed": protocol["seed"],
                "injection_ratio": protocol["tvnia"]["shared"]["injection_ratio"],
                "structure_iterations": protocol["tvnia"]["shared"]["contrastive"]["structure_iterations"],
                "status": "ERROR",
            }
            try:
                set_seed(protocol["seed"])
                _, clean_result = train_defense(
                    defense,
                    dataset.graph,
                    dataset.features,
                    dataset.labels,
                    dataset.train_indices,
                    dataset.validation_indices,
                    dataset.test_indices,
                    graph_state="clean",
                    epochs=defense_parameters["clean_epochs"],
                    patience=defense_parameters["patience"],
                    learning_rate=defense_parameters["learning_rate"],
                    weight_decay=defense_parameters["weight_decay"],
                    hidden_dim=defense_parameters["hidden_dim"],
                    dropout=defense_parameters["dropout"],
                    threshold=defense_parameters["threshold"],
                )
                set_seed(protocol["seed"])
                _, poisoned_result = train_defense(
                    defense,
                    poisoned_graph,
                    poisoned_features,
                    poisoned_labels,
                    dataset.train_indices,
                    dataset.validation_indices,
                    dataset.test_indices,
                    graph_state="poisoned",
                    epochs=defense_parameters["poison_epochs"],
                    patience=defense_parameters["patience"],
                    learning_rate=defense_parameters["learning_rate"],
                    weight_decay=defense_parameters["weight_decay"],
                    hidden_dim=defense_parameters["hidden_dim"],
                    dropout=defense_parameters["dropout"],
                    threshold=defense_parameters["threshold"],
                )
                row.update(
                    clean_defense_accuracy=clean_result.test_accuracy,
                    poisoned_defense_accuracy=poisoned_result.test_accuracy,
                    clean_epochs_ran=clean_result.epochs_ran,
                    poisoned_epochs_ran=poisoned_result.epochs_ran,
                    injected_nodes=attack_metadata["injected_nodes"],
                    degree_per_injected_node=attack_metadata["degree_per_injected_node"],
                    total_degree_used=attack_metadata["total_degree_used"],
                    total_degree_budget=attack_metadata["total_degree_budget"],
                    budget_valid=attack_metadata["budget_valid"],
                    attack_metadata=json.dumps(attack_metadata, sort_keys=True),
                    status="OK",
                )
                print(
                    f"[result] {dataset_id}/{defense}: poisoned={poisoned_result.test_accuracy:.6f}",
                    flush=True,
                )
            except Exception as error:
                row["error"] = repr(error)
                print(f"[error] {dataset_id}/{defense}: {error!r}", flush=True)
            append_row(args.output, row)


if __name__ == "__main__":
    main()

