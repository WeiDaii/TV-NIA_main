"""End-to-end TV-NIA framework orchestration."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import dgl
import torch

from .config import InjectionBudget, TVNIAConfig
from .contrastive_reconstruction import (
    ContrastiveReconstructionConfig,
    reconstruct_injected_topology,
)
from .graph_ops import build_poisoned_graph
from .vulnerable_topology import (
    ProbabilityOracle,
    construct_candidate_regions,
    initialize_cross_edges,
    initialize_injected_features,
    locate_vulnerable_anchors,
    set_random_seed,
)
from .zeroth_order_features import refine_injected_features


def compute_injection_budget(
    graph: dgl.DGLGraph, config: TVNIAConfig
) -> InjectionBudget:
    """Compute the exact node and degree budget used by the final experiment."""
    graph_without_loops = dgl.remove_self_loop(graph)
    num_nodes = int(graph_without_loops.num_nodes())
    injected_nodes = (
        int(config.injected_nodes)
        if config.injected_nodes > 0
        else max(1, math.ceil(config.injection_ratio * num_nodes))
    )
    degree = (
        int(config.degree_per_injected_node)
        if config.degree_per_injected_node > 0
        else max(1, round(graph_without_loops.num_edges() / max(1, num_nodes)))
    )
    total = injected_nodes * degree
    return InjectionBudget(
        injected_nodes=injected_nodes,
        degree_per_injected_node=degree,
        total_degree_budget=total,
        cross_edges=total,
        internal_edges=0,
    )


def _contrastive_config(config: TVNIAConfig) -> ContrastiveReconstructionConfig:
    return ContrastiveReconstructionConfig(
        hidden_dim=config.contrastive_hidden_dim,
        projection_dim=config.contrastive_projection_dim,
        temperature=config.contrastive_temperature,
        learning_rate=config.contrastive_learning_rate,
        weight_decay=config.contrastive_weight_decay,
        train_epochs=config.contrastive_train_epochs,
        structure_iterations=config.structure_iterations,
        actions_per_iteration=config.structure_actions_per_iteration,
        feature_updates_per_iteration=config.contrastive_feature_updates_per_iteration,
        feature_step_size=config.contrastive_feature_step_size,
        max_local_original_nodes=config.max_local_original_nodes,
        seed=config.seed,
    )


def run_tvnia(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    probability_oracle: ProbabilityOracle,
    config: TVNIAConfig,
    split: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[dgl.DGLGraph, torch.Tensor, Dict[str, object]]:
    """Construct and return a TV-NIA poisoned graph.

    The framework accepts prediction probabilities but no labels, parameters,
    or gradients from the clean victim model.
    """
    set_random_seed(config.seed)
    device = device or features.device
    graph = graph.to(device)
    features = features.to(device).float()
    budget = compute_injection_budget(graph, config)

    anchor_info = locate_vulnerable_anchors(
        graph, features, probability_oracle, config, budget, split
    )
    candidate_regions = construct_candidate_regions(features, anchor_info, config)
    injected_features = initialize_injected_features(features, anchor_info, config)
    cross_targets = initialize_cross_edges(anchor_info, candidate_regions, budget)

    cross_targets, injected_features, reconstruction_info = reconstruct_injected_topology(
        g=graph,
        x=features,
        x_inj=injected_features,
        cross_targets=cross_targets,
        anchor_info={
            "anchors": anchor_info["anchors"],
            "pseudo": anchor_info["pseudo_labels"],
        },
        candidate_regions=candidate_regions,
        budget=budget,
        cfg=_contrastive_config(config),
    )
    injected_features = refine_injected_features(
        graph,
        features,
        injected_features,
        cross_targets,
        anchor_info,
        budget,
        probability_oracle,
        config,
    )
    poisoned_graph, poisoned_features, graph_info = build_poisoned_graph(
        graph, features, injected_features, cross_targets, budget
    )

    metadata: Dict[str, object] = {
        **graph_info,
        "framework": "TV-NIA",
        "anchor_scope": config.anchor_scope,
        "anchor_count": int(anchor_info["anchors"].numel()),
        "candidate_nodes_per_anchor": config.candidate_nodes_per_anchor,
        "feature_initialization": config.feature_initialization,
        "feature_refinement_steps": config.feature_refinement_steps,
        "degree_per_injected_node": budget.degree_per_injected_node,
        **{f"reconstruction_{key}": value for key, value in reconstruction_info.items()},
    }
    return poisoned_graph, poisoned_features, metadata
