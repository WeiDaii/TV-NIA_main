"""Unified training interface for the four reported defense models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import dgl
import torch
import torch.nn.functional as F

from .gnnguard import GCNGuard
from .gtrans import GTrans
from .prognn import ProGNN
from .rgcn import RobustGCN


DEFENSE_NAMES = ("gnnguard", "rgcn", "prognn", "gtrans")


@dataclass(frozen=True)
class DefenseTrainingResult:
    defense: str
    graph_state: str
    best_validation_accuracy: float
    test_accuracy: float
    epochs_ran: int


def _accuracy(logits, labels, indices) -> float:
    return float((logits[indices].argmax(-1) == labels[indices]).float().mean())


def _forward(model, graph, features, no_grad=False):
    try:
        return model(graph, features, no_grad=no_grad)
    except TypeError:
        return model(graph, features)


def _needs_gradient_during_evaluation(defense: str) -> bool:
    return defense in {"prognn", "gtrans"}


def _prepare_graph(graph, defense, train, validation, test, labels):
    graph = dgl.remove_self_loop(graph)
    if defense in {"rgcn", "prognn", "gtrans"}:
        graph = dgl.add_self_loop(graph)
    for name, indices in (
        ("train_mask", train),
        ("val_mask", validation),
        ("test_mask", test),
    ):
        mask = torch.zeros(graph.num_nodes(), dtype=torch.bool, device=indices.device)
        mask[indices.long()] = True
        graph.ndata[name] = mask
    graph.ndata["label"] = labels
    return graph


def build_defense(
    name: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    threshold: float,
    device: torch.device,
):
    name = name.lower()
    if name == "gnnguard":
        model = GCNGuard(
            input_dim, num_classes, hids=[hidden_dim], dropout=dropout, threshold=threshold
        )
    elif name == "rgcn":
        model = RobustGCN(input_dim, num_classes, n_hids=[hidden_dim], dropout=dropout)
    elif name == "prognn":
        model = ProGNN(input_dim, hidden_dim, num_classes, device=str(device))
    elif name == "gtrans":
        model = GTrans(input_dim, hidden_dim, num_classes, epochs=2, loop_feat=1, loop_adj=1)
    else:
        raise ValueError(f"Unsupported defense model: {name}")
    return model.to(device)


def train_defense(
    name: str,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    *,
    graph_state: str,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    threshold: float,
):
    name = name.lower()
    device = features.device
    graph = _prepare_graph(
        graph.to(device),
        name,
        train_indices,
        validation_indices,
        test_indices,
        labels,
    )
    model = build_defense(
        name, features.size(1), int(labels.max()) + 1, hidden_dim, dropout, threshold, device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: Dict[str, torch.Tensor] | None = None
    best_validation = -1.0
    stale_epochs = 0
    epochs_ran = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = _forward(model, graph, features, no_grad=False)
        F.cross_entropy(logits[train_indices], labels[train_indices]).backward()
        optimizer.step()

        model.eval()
        if _needs_gradient_during_evaluation(name):
            validation_logits = _forward(model, graph, features, no_grad=False).detach()
        else:
            with torch.no_grad():
                validation_logits = _forward(model, graph, features, no_grad=True)
        validation_accuracy = _accuracy(validation_logits, labels, validation_indices)
        epochs_ran = epoch
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    model.eval()
    if _needs_gradient_during_evaluation(name):
        test_logits = _forward(model, graph, features, no_grad=False).detach()
    else:
        with torch.no_grad():
            test_logits = _forward(model, graph, features, no_grad=True)
    result = DefenseTrainingResult(
        defense=name,
        graph_state=graph_state,
        best_validation_accuracy=best_validation,
        test_accuracy=_accuracy(test_logits, labels, test_indices),
        epochs_ran=epochs_ran,
    )
    return model, result

