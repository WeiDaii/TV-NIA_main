"""Vulnerable-anchor localization, candidates, and feature initialization.

This file contains only the final protocol used in the reported main and
defense experiments. It never receives ground-truth labels.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import dgl
import torch

from .config import InjectionBudget, TVNIAConfig


ProbabilityOracle = Callable[
    [dgl.DGLGraph, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_binary_features(features: torch.Tensor, tolerance: float = 1e-6) -> bool:
    if features.numel() == 0:
        return False
    if float(features.min()) < -tolerance or float(features.max()) > 1.0 + tolerance:
        return False
    near_zero = features.abs() < tolerance
    near_one = (features - 1.0).abs() < tolerance
    return bool((near_zero | near_one).float().mean() > 0.99)


def project_features(
    values: torch.Tensor,
    reference: torch.Tensor,
    topk: int,
    domain: str,
) -> torch.Tensor:
    if domain == "unit_box":
        return values.clamp(0.0, 1.0)
    if _is_binary_features(reference):
        values = values.clamp(0.0, 1.0)
        k = min(max(1, int(topk)), int(values.size(1)))
        indices = torch.topk(values, k=k, dim=1).indices
        projected = torch.zeros_like(values)
        projected.scatter_(1, indices, 1.0)
        return projected
    return values.clamp(float(reference.min()), float(reference.max()))


def feature_bounds(reference: torch.Tensor, domain: str) -> tuple[float, float]:
    if domain == "unit_box" or _is_binary_features(reference):
        return 0.0, 1.0
    return float(reference.min()), float(reference.max())


def _column_bounds(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if _is_binary_features(reference):
        return torch.zeros_like(reference[0]), torch.ones_like(reference[0])
    return reference.min(dim=0).values, reference.max(dim=0).values


@torch.no_grad()
def _clean_predictions(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    oracle: ProbabilityOracle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = oracle(graph, features, None).detach()
    pseudo_labels = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    top_two = torch.topk(probabilities, k=min(2, probabilities.size(1)), dim=1).values
    margin = top_two[:, 0] if top_two.size(1) == 1 else top_two[:, 0] - top_two[:, 1]
    uncertainty = 1.0 - margin
    return probabilities, pseudo_labels, confidence, uncertainty


@torch.no_grad()
def _boundary_scores(
    graph: dgl.DGLGraph,
    pseudo_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    graph = dgl.remove_self_loop(graph)
    src, dst = graph.edges()
    src = src.to(pseudo_labels.device).long()
    dst = dst.to(pseudo_labels.device).long()
    degree = torch.zeros(graph.num_nodes(), device=pseudo_labels.device)
    disagreement = torch.zeros_like(degree)
    degree.index_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    disagreement.index_add_(0, src, (pseudo_labels[src] != pseudo_labels[dst]).float())
    boundary = disagreement / degree.clamp(min=1.0)
    bridge = boundary / torch.log1p(degree).clamp(min=1.0)
    return boundary, bridge


def _scope_indices(
    num_nodes: int,
    split: Optional[Dict[str, torch.Tensor]],
    scope: str,
    device: torch.device,
) -> torch.Tensor:
    if split is not None and scope in split:
        return split[scope].to(device).long()
    return torch.arange(num_nodes, device=device)


@torch.no_grad()
def _class_prototypes(
    features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    prototypes = torch.zeros(
        (num_classes, features.size(1)), dtype=features.dtype, device=features.device
    )
    global_mean = features.mean(dim=0)
    for class_id in range(num_classes):
        indices = torch.where(pseudo_labels == class_id)[0]
        prototypes[class_id] = features[indices].mean(dim=0) if indices.numel() else global_mean
    return prototypes


@torch.no_grad()
def locate_vulnerable_anchors(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    oracle: ProbabilityOracle,
    config: TVNIAConfig,
    budget: InjectionBudget,
    split: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Rank eligible nodes by the final TV-NIA vulnerability score."""
    scope = _scope_indices(
        graph.num_nodes(), split, config.anchor_scope, features.device
    )
    if scope.numel() == 0:
        scope = torch.arange(graph.num_nodes(), device=features.device)

    probabilities, pseudo_labels, confidence, uncertainty = _clean_predictions(
        graph, features, oracle
    )
    boundary, bridge = _boundary_scores(graph, pseudo_labels)
    score = (
        config.uncertainty_weight * uncertainty
        + config.bridge_weight * bridge
        + config.boundary_weight * boundary
    )

    if config.anchor_query_budget != 0:
        raise ValueError("The final TV-NIA protocol requires anchor_query_budget=0")
    preselect_count = min(config.anchor_preselection_limit, int(scope.numel()))
    preselected = scope[torch.topk(score[scope], k=preselect_count).indices]
    required = int(budget.injected_nodes)
    if preselected.numel() < required:
        selected_ids = set(int(node) for node in preselected.detach().cpu().tolist())
        extras: List[int] = []
        for node in torch.argsort(score, descending=True).detach().cpu().tolist():
            if int(node) not in selected_ids:
                extras.append(int(node))
                selected_ids.add(int(node))
            if preselected.numel() + len(extras) >= required:
                break
        if extras:
            preselected = torch.cat(
                [preselected, torch.tensor(extras, device=features.device)]
            )
    selected_count = min(required, int(preselected.numel()))
    # Preserve the original two-stage top-k ordering. This matters when many
    # nodes have tied vulnerability scores because anchor order determines the
    # per-anchor candidate RNG stream.
    refined_scores = score[preselected]
    anchors = preselected[
        torch.topk(refined_scores, k=selected_count, largest=True).indices
    ].long()
    prototypes = _class_prototypes(features, pseudo_labels, probabilities.size(1))
    return {
        "probabilities": probabilities,
        "pseudo_labels": pseudo_labels,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "boundary": boundary,
        "bridge": bridge,
        "vulnerability_score": score,
        "anchors": anchors,
        "prototypes": prototypes,
        "scope_nodes": scope.long(),
    }


@torch.no_grad()
def construct_candidate_regions(
    features: torch.Tensor,
    anchor_info: Dict[str, torch.Tensor],
    config: TVNIAConfig,
) -> List[torch.Tensor]:
    """Randomly sample the final protocol's local candidate set per anchor."""
    anchors = anchor_info["anchors"]
    scope = anchor_info["scope_nodes"].long()
    width = min(max(config.candidate_nodes_per_anchor, 1), int(scope.numel()))
    regions: List[torch.Tensor] = []
    for index, anchor in enumerate(anchors.detach().cpu().tolist()):
        candidates = scope[scope != int(anchor)]
        if candidates.numel() == 0:
            candidates = torch.tensor([anchor], device=features.device)
        generator = torch.Generator(device=features.device)
        generator.manual_seed(config.seed + 1009 * index + 17)
        permutation = torch.randperm(
            candidates.numel(), device=features.device, generator=generator
        )
        regions.append(candidates[permutation[: min(width, candidates.numel())]].long())
    return regions


@torch.no_grad()
def initialize_injected_features(
    features: torch.Tensor,
    anchor_info: Dict[str, torch.Tensor],
    config: TVNIAConfig,
) -> torch.Tensor:
    """Initialize injected features using the dataset-group protocol."""
    rows: List[torch.Tensor] = []
    probabilities = anchor_info["probabilities"]
    pseudo_labels = anchor_info["pseudo_labels"]
    prototypes = anchor_info["prototypes"]
    low, high = _column_bounds(features)

    for anchor in anchor_info["anchors"].long():
        scores = probabilities[anchor]
        own_class = int(pseudo_labels[anchor])
        wrong_scores = scores.clone()
        wrong_scores[own_class] = -1.0
        wrong_class = int(wrong_scores.argmax())
        least_scores = scores.clone()
        least_scores[own_class] = 2.0
        least_class = int(least_scores.argmin())

        if config.feature_initialization == "wrong_class_prototype":
            row = (
                config.prototype_mix * prototypes[wrong_class]
                + (1.0 - config.prototype_mix) * features[anchor]
            )
        elif config.feature_initialization == "anti_class_boundary":
            direction = prototypes[least_class] - prototypes[own_class]
            row = torch.where(direction >= 0, high, low)
        else:
            raise ValueError(
                f"Unknown feature initialization: {config.feature_initialization}"
            )
        rows.append(row)

    injected = torch.stack(rows) if rows else features.new_zeros((0, features.size(1)))
    return project_features(
        injected, features, config.feature_topk, config.feature_domain
    )


@torch.no_grad()
def initialize_cross_edges(
    anchor_info: Dict[str, torch.Tensor],
    candidate_regions: List[torch.Tensor],
    budget: InjectionBudget,
) -> List[torch.Tensor]:
    """Keep each anchor edge and fill the remaining per-node degree budget."""
    remaining = int(budget.cross_edges)
    selected: List[torch.Tensor] = []
    for anchor, candidates in zip(anchor_info["anchors"], candidate_regions):
        if remaining <= 0:
            selected.append(candidates[:0])
            continue
        cap = min(budget.degree_per_injected_node, remaining)
        candidates = candidates[candidates != anchor]
        chosen = torch.unique(
            torch.cat([anchor.view(1), candidates[: max(0, cap - 1)]])
        )[:cap]
        selected.append(chosen.long())
        remaining -= int(chosen.numel())
    return selected
