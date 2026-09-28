"""Graph construction utilities shared by main and defense experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import dgl
import torch

from .config import InjectionBudget


def build_poisoned_graph(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    injected_features: torch.Tensor,
    cross_targets: Sequence[torch.Tensor],
    budget: InjectionBudget,
) -> Tuple[dgl.DGLGraph, torch.Tensor, Dict[str, object]]:
    """Append injected nodes and bidirectional cross edges.

    The final protocol sets the internal-edge budget to zero, so this function
    intentionally exposes no injected-to-injected edge path.
    """
    device = features.device
    graph_without_loops = dgl.remove_self_loop(graph)
    original_nodes = int(graph_without_loops.num_nodes())
    injected_nodes = int(injected_features.size(0))
    poisoned = dgl.add_nodes(graph_without_loops, injected_nodes)
    injected_ids = torch.arange(
        original_nodes,
        original_nodes + injected_nodes,
        dtype=torch.long,
        device=device,
    )
    sources: List[torch.Tensor] = []
    destinations: List[torch.Tensor] = []
    remaining = int(budget.cross_edges)
    used = 0
    for injected_index, targets in enumerate(cross_targets):
        if remaining <= 0:
            break
        targets = torch.unique(targets.to(device).long())[:remaining]
        if targets.numel() == 0:
            continue
        injected = injected_ids[injected_index].repeat(targets.numel())
        sources.extend([injected, targets])
        destinations.extend([targets, injected])
        used += int(targets.numel())
        remaining -= int(targets.numel())
    if sources:
        poisoned = dgl.add_edges(
            poisoned, torch.cat(sources).long(), torch.cat(destinations).long()
        )
    poisoned = dgl.add_self_loop(poisoned).to(device)
    poisoned_features = torch.cat([features, injected_features.to(device)], dim=0)
    metadata: Dict[str, object] = {
        "injected_nodes": injected_nodes,
        "cross_edges": used,
        "internal_edges": 0,
        "total_degree_used": used,
        "total_degree_budget": int(budget.total_degree_budget),
        "budget_valid": used <= budget.total_degree_budget,
    }
    return poisoned, poisoned_features, metadata

