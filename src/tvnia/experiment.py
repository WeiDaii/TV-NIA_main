"""Reusable execution primitives for main and defense experiments."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from .data import GraphDataset, load_dataset
from .experiment_config import make_tvnia_config
from .framework import run_tvnia
from .models import build_victim, extend_labels, train_victim


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def train_clean_oracle(
    dataset: GraphDataset,
    victim_name: str,
    protocol: Dict[str, Any],
    clean_epochs: int,
):
    parameters = protocol["victim_training"]
    model = build_victim(
        victim_name,
        dataset.features.size(1),
        parameters["hidden_dim"],
        int(dataset.labels.max()) + 1,
        parameters["dropout"],
        dataset.features.device,
    )
    result = train_victim(
        model,
        dataset.graph,
        dataset.features,
        dataset.labels,
        dataset.train_indices,
        dataset.validation_indices,
        dataset.test_indices,
        epochs=clean_epochs,
        patience=parameters["patience"],
        learning_rate=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
    )
    return model, result


def construct_poisoned_dataset(
    dataset_name: str,
    victim_name: str,
    protocol: Dict[str, Any],
    data_root: Path,
    device: torch.device,
    clean_epochs: int,
):
    seed = protocol["seed"]
    set_seed(seed)
    dataset = load_dataset(
        dataset_name,
        victim_name,
        data_root,
        device,
        seed,
        split_profile=protocol["data_splits"][dataset_name][victim_name],
    )
    clean_model, clean_result = train_clean_oracle(
        dataset, victim_name, protocol, clean_epochs
    )

    @torch.no_grad()
    def probability_oracle(graph, features, nodes=None):
        clean_model.eval()
        probabilities = F.softmax(clean_model(graph, features), dim=-1)
        return probabilities[nodes.long()] if nodes is not None else probabilities

    tvnia_config = make_tvnia_config(protocol, dataset_name, dataset.graph)
    poisoned_graph, poisoned_features, attack_metadata = run_tvnia(
        dataset.graph,
        dataset.features,
        probability_oracle,
        tvnia_config,
        split=dataset.split,
        device=device,
    )
    poisoned_labels = extend_labels(
        dataset.labels, int(attack_metadata["injected_nodes"])
    )
    poisoned_graph.ndata["features"] = poisoned_features
    poisoned_graph.ndata["labels"] = poisoned_labels
    return (
        dataset,
        clean_result,
        poisoned_graph,
        poisoned_features,
        poisoned_labels,
        attack_metadata,
        asdict(tvnia_config),
    )
