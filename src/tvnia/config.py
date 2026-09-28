"""Typed configuration for the final TV-NIA experiment protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionBudget:
    """Node-injection budget measured on the undirected attack graph."""

    injected_nodes: int
    degree_per_injected_node: int
    total_degree_budget: int
    cross_edges: int
    internal_edges: int = 0

    @property
    def total_degree_used(self) -> int:
        return self.cross_edges + 2 * self.internal_edges

    @property
    def is_valid(self) -> bool:
        return self.total_degree_used <= self.total_degree_budget


@dataclass(frozen=True)
class TVNIAConfig:
    """Configuration used to construct one TV-NIA poisoned graph.

    The defaults match ``final_no_seek_exper_3``. Dataset-dependent values are
    supplied by :func:`tvnia.experiment_config.make_tvnia_config`.
    """

    injection_ratio: float = 0.05
    injected_nodes: int = -1
    degree_per_injected_node: int = -1
    anchor_scope: str = "train"
    anchor_preselection_limit: int = 512
    anchor_query_budget: int = 0
    candidate_nodes_per_anchor: int = 16
    feature_initialization: str = "wrong_class_prototype"
    feature_domain: str = "unit_box"
    feature_topk: int = 50
    prototype_mix: float = 0.65

    # Vulnerability score: uncertainty + weak bridge + boundary disagreement.
    uncertainty_weight: float = 0.45
    bridge_weight: float = 0.35
    boundary_weight: float = 0.20

    # Contrastive structural reconstruction.
    contrastive_hidden_dim: int = 256
    contrastive_projection_dim: int = 32
    contrastive_temperature: float = 0.4
    contrastive_learning_rate: float = 0.05
    contrastive_weight_decay: float = 1e-5
    contrastive_train_epochs: int = 20
    structure_iterations: int = 3
    structure_actions_per_iteration: int = 16
    contrastive_feature_updates_per_iteration: int = 0
    contrastive_feature_step_size: float = 1.0
    max_local_original_nodes: int = 1024

    # Zeroth-order injected-feature refinement.
    feature_refinement_steps: int = 60
    feature_perturbation_scale: float = 0.1
    feature_learning_rate: float = 0.8
    max_feature_evaluation_nodes: int = -1

    seed: int = 42
