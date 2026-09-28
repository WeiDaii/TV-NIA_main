"""Victim GNNs and the training protocol used by the final experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GINConv, GraphConv, SAGEConv


VICTIM_NAMES = ("gcn", "graphsage", "gin")


def canonical_victim_name(name: str) -> str:
    normalized = name.lower()
    if normalized not in VICTIM_NAMES:
        raise ValueError(f"Unsupported victim model: {name}")
    return normalized


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.first = GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        self.second = GraphConv(hidden_dim, num_classes, allow_zero_in_degree=True)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        return self.second(graph, self.first(graph, features))


class GraphSAGE(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float
    ):
        super().__init__()
        self.first = SAGEConv(input_dim, hidden_dim, aggregator_type="mean")
        self.second = SAGEConv(hidden_dim, num_classes, aggregator_type="mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        # SAGEConv already has a separate self-feature branch.
        graph = dgl.remove_self_loop(graph)
        hidden = F.relu(self.first(graph, features))
        return self.second(graph, self.dropout(hidden))


class GIN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.first = GINConv(self.first_linear, aggregator_type="sum")
        self.second = GINConv(self.second_linear, aggregator_type="sum")
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.first_linear.weight, gain=gain)
        nn.init.xavier_normal_(self.second_linear.weight, gain=gain)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.first(graph, features))
        return F.elu(self.second(graph, hidden))


def build_victim(
    name: str,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    name = canonical_victim_name(name)
    if name == "gcn":
        model = GCN(input_dim, hidden_dim, num_classes)
    elif name == "graphsage":
        model = GraphSAGE(input_dim, hidden_dim, num_classes, dropout)
    else:
        model = GIN(input_dim, hidden_dim, num_classes)
    return model.to(device)


@dataclass(frozen=True)
class TrainingResult:
    best_validation_accuracy: float
    test_accuracy: float
    epochs_ran: int


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> float:
    prediction = logits[indices].argmax(dim=-1)
    return float((prediction == labels[indices]).float().mean())


def train_victim(
    model: nn.Module,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
) -> TrainingResult:
    """Train with the exact early-stopping behavior of the source experiment."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_validation = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[train_indices], labels[train_indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Kept in training mode for strict compatibility with final_no_seek_exper_3.
        with torch.no_grad():
            validation_logits = model(graph, features)
        validation_accuracy = _accuracy(
            validation_logits, labels, validation_indices
        )
        epochs_ran = epoch
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_accuracy = _accuracy(model(graph, features), labels, test_indices)
    return TrainingResult(best_validation, test_accuracy, epochs_ran)


def extend_labels(labels: torch.Tensor, injected_nodes: int) -> torch.Tensor:
    filler = torch.zeros(injected_nodes, dtype=labels.dtype, device=labels.device)
    return torch.cat([labels, filler])
