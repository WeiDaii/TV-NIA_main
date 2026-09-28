"""Score-only zeroth-order refinement of injected node features."""

from __future__ import annotations

from typing import Dict, Sequence

import dgl
import torch

from .config import InjectionBudget, TVNIAConfig
from .graph_ops import build_poisoned_graph
from .vulnerable_topology import ProbabilityOracle, feature_bounds, project_features


@torch.no_grad()
def _objective(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    oracle: ProbabilityOracle,
    evaluation_nodes: torch.Tensor,
    pseudo_labels: torch.Tensor,
) -> torch.Tensor:
    if evaluation_nodes.numel() == 0:
        return torch.tensor(0.0, device=features.device)
    probabilities = oracle(graph, features, evaluation_nodes)
    row_ids = torch.arange(evaluation_nodes.numel(), device=features.device)
    return probabilities[row_ids, pseudo_labels[evaluation_nodes].long()].mean()


@torch.no_grad()
def refine_injected_features(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    injected_features: torch.Tensor,
    cross_targets: Sequence[torch.Tensor],
    anchor_info: Dict[str, torch.Tensor],
    budget: InjectionBudget,
    oracle: ProbabilityOracle,
    config: TVNIAConfig,
) -> torch.Tensor:
    """Minimize clean pseudo-class confidence with two score queries per step."""
    if config.feature_refinement_steps <= 0 or injected_features.numel() == 0:
        return injected_features
    affected = [anchor_info["anchors"].long()]
    affected.extend(target.long() for target in cross_targets if target.numel())
    evaluation_nodes = torch.unique(torch.cat(affected))
    limit = config.max_feature_evaluation_nodes
    if limit > 0 and evaluation_nodes.numel() > limit:
        score = anchor_info["uncertainty"][evaluation_nodes] + anchor_info["bridge"][evaluation_nodes]
        evaluation_nodes = evaluation_nodes[torch.topk(score, k=limit).indices]
    pseudo_labels = anchor_info["pseudo_labels"].long()
    sigma = config.feature_perturbation_scale
    learning_rate = config.feature_learning_rate
    low, high = feature_bounds(features, config.feature_domain)
    current = injected_features.detach().clone().float().clamp(low, high)
    for _ in range(config.feature_refinement_steps):
        direction = torch.empty_like(current).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        positive = (current + sigma * direction).clamp(low, high)
        negative = (current - sigma * direction).clamp(low, high)
        positive_graph, positive_features, _ = build_poisoned_graph(
            graph, features, positive, cross_targets, budget
        )
        negative_graph, negative_features, _ = build_poisoned_graph(
            graph, features, negative, cross_targets, budget
        )
        positive_loss = _objective(
            positive_graph, positive_features, oracle, evaluation_nodes, pseudo_labels
        )
        negative_loss = _objective(
            negative_graph, negative_features, oracle, evaluation_nodes, pseudo_labels
        )
        gradient = ((positive_loss - negative_loss) / max(1e-6, 2.0 * sigma)) * direction
        current = (current - learning_rate * gradient).clamp(low, high)
    return project_features(current, features, config.feature_topk, config.feature_domain)

