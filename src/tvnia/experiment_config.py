"""Single source of truth for the final main/defense experiment parameters."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import dgl

from .config import TVNIAConfig
from .data import CITATION_DATASETS, canonical_dataset_id


def load_protocol(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def make_tvnia_config(
    protocol: Dict[str, Any],
    dataset: str,
    graph: dgl.DGLGraph,
) -> TVNIAConfig:
    dataset_id = canonical_dataset_id(dataset)
    group = "citation" if dataset_id in CITATION_DATASETS else "sampled_network"
    shared = protocol["tvnia"]["shared"]
    profile = protocol["tvnia"][group]
    graph_without_loops = dgl.remove_self_loop(graph)
    num_nodes = int(graph_without_loops.num_nodes())
    injected_nodes = max(1, math.ceil(shared["injection_ratio"] * num_nodes))
    degree = max(1, round(graph_without_loops.num_edges() / max(1, num_nodes)))
    return TVNIAConfig(
        injection_ratio=shared["injection_ratio"],
        injected_nodes=injected_nodes,
        degree_per_injected_node=degree,
        anchor_scope=profile["anchor_scope"],
        anchor_preselection_limit=shared["anchor_preselection_limit"],
        anchor_query_budget=shared["anchor_query_budget"],
        candidate_nodes_per_anchor=profile["candidate_nodes_per_anchor"],
        feature_initialization=profile["feature_initialization"],
        feature_domain=profile["feature_domain"],
        feature_topk=profile["feature_topk"],
        prototype_mix=shared["prototype_mix"],
        uncertainty_weight=shared["vulnerability_weights"]["uncertainty"],
        bridge_weight=shared["vulnerability_weights"]["bridge"],
        boundary_weight=shared["vulnerability_weights"]["boundary"],
        contrastive_hidden_dim=shared["contrastive"]["hidden_dim"],
        contrastive_projection_dim=shared["contrastive"]["projection_dim"],
        contrastive_temperature=shared["contrastive"]["temperature"],
        contrastive_learning_rate=shared["contrastive"]["learning_rate"],
        contrastive_weight_decay=shared["contrastive"]["weight_decay"],
        contrastive_train_epochs=shared["contrastive"]["train_epochs"],
        structure_iterations=shared["contrastive"]["structure_iterations"],
        structure_actions_per_iteration=shared["contrastive"]["actions_per_iteration"],
        contrastive_feature_updates_per_iteration=shared["contrastive"]["feature_updates_per_iteration"],
        contrastive_feature_step_size=shared["contrastive"]["feature_step_size"],
        max_local_original_nodes=profile["max_local_original_nodes"],
        feature_refinement_steps=profile["feature_refinement"]["steps"],
        feature_perturbation_scale=profile["feature_refinement"]["perturbation_scale"],
        feature_learning_rate=profile["feature_refinement"]["learning_rate"],
        max_feature_evaluation_nodes=profile["feature_refinement"]["max_evaluation_nodes"],
        seed=protocol["seed"],
    )
