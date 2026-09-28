# -*- coding: utf-8 -*-
"""Contrastive reconstruction of the injected topology.

The clean extended graph and current poisoned graph form two contrastive views.
Only injected-to-original edges can be changed; original graph edges and
features remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import dgl
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ContrastiveReconstructionConfig:
    hidden_dim: int = 256
    projection_dim: int = 32
    temperature: float = 0.4
    learning_rate: float = 0.05
    weight_decay: float = 1e-5
    train_epochs: int = 20
    structure_iterations: int = 3
    actions_per_iteration: int = 16
    feature_updates_per_iteration: int = 0
    feature_step_size: float = 1.0
    max_local_original_nodes: int = 1024
    seed: int = 42


class DifferentiableGCNEncoder(nn.Module):
    """Dense differentiable GCN encoder used by topology reconstruction."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 2 * hidden_dim, bias=False)
        self.fc2 = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        self.act = nn.PReLU()

    @staticmethod
    def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        adj = adj.clamp(0.0, 1.0)
        adj = torch.maximum(adj, adj.t())
        eye = torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
        adj = torch.maximum(adj, eye)
        deg = adj.sum(dim=1).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = self.normalize_adj(adj)
        h = self.act(adj_norm @ self.fc1(x))
        return self.act(adj_norm @ self.fc2(h))


class ContrastiveModel(nn.Module):
    """Two-view node-level contrastive model."""

    def __init__(self, encoder: DifferentiableGCNEncoder, hidden_dim: int, proj_dim: int, tau: float):
        super().__init__()
        self.encoder = encoder
        self.tau = float(tau)
        self.fc1 = nn.Linear(hidden_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, adj)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))

    @staticmethod
    def sim(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, p=2, dim=1, eps=1e-12)
        z2 = F.normalize(z2, p=2, dim=1, eps=1e-12)
        return z1 @ z2.t()

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        f = lambda s: torch.exp(s / max(self.tau, 1e-6))
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        denom = refl_sim.sum(dim=1) + between_sim.sum(dim=1) - refl_sim.diag()
        return -torch.log(between_sim.diag() / denom.clamp(min=1e-12))

    def loss(self, z_clean: torch.Tensor, z_adv: torch.Tensor, mean: bool = True) -> torch.Tensor:
        h_clean = self.projection(z_clean)
        h_adv = self.projection(z_adv)
        loss_clean_to_adv = self.semi_loss(h_clean, h_adv)
        loss_adv_to_clean = self.semi_loss(h_adv, h_clean)
        loss = 0.5 * (loss_clean_to_adv + loss_adv_to_clean)
        return loss.mean() if mean else loss.sum()


def _unique_preserve_order(values: Iterable[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    return out


@torch.no_grad()
def _collect_local_original_nodes(
    anchors: torch.Tensor,
    candidate_regions: Sequence[torch.Tensor],
    cross_targets: Sequence[torch.Tensor],
    cfg: ContrastiveReconstructionConfig,
) -> torch.Tensor:
    """Collect neighbor nodes in g_i while keeping the local block bounded."""
    device = anchors.device
    vals: List[int] = []
    vals.extend(int(v) for v in anchors.detach().cpu().tolist())
    for seq in list(candidate_regions) + list(cross_targets):
        if seq.numel() > 0:
            vals.extend(int(v) for v in seq.detach().cpu().tolist())
    vals = _unique_preserve_order(vals)
    vals = vals[: max(1, int(cfg.max_local_original_nodes))]
    return torch.tensor(vals, dtype=torch.long, device=device)


@torch.no_grad()
def _make_local_clean_view(
    g: dgl.DGLGraph,
    x: torch.Tensor,
    x_inj: torch.Tensor,
    original_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
    """Eq. (5): extended clean graph G^(0), with isolated injected nodes."""
    device = x.device
    n_orig = int(original_nodes.numel())
    n_inj = int(x_inj.size(0))
    node_to_local = {int(v): i for i, v in enumerate(original_nodes.detach().cpu().tolist())}
    adj = torch.zeros((n_orig + n_inj, n_orig + n_inj), dtype=x.dtype, device=device)

    src, dst = g.edges()
    src = src.detach().cpu().tolist()
    dst = dst.detach().cpu().tolist()
    for u, v in zip(src, dst):
        if int(u) == int(v):
            continue
        if int(u) in node_to_local and int(v) in node_to_local:
            adj[node_to_local[int(u)], node_to_local[int(v)]] = 1.0

    features = torch.cat([x[original_nodes].float(), x_inj.float()], dim=0)
    return adj, features, node_to_local


def _make_local_adv_view(
    clean_adj: torch.Tensor,
    clean_x: torch.Tensor,
    x_inj: torch.Tensor,
    cross_targets: Sequence[torch.Tensor],
    node_to_local: Dict[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Current adversarial graph G_adv^(t-1) restricted to g_i nodes."""
    n_orig = len(node_to_local)
    adj = clean_adj.clone()
    for inj_idx, targets in enumerate(cross_targets):
        inj_local = n_orig + int(inj_idx)
        for v in targets.detach().cpu().tolist():
            if int(v) not in node_to_local:
                continue
            orig_local = node_to_local[int(v)]
            adj[inj_local, orig_local] = 1.0
            adj[orig_local, inj_local] = 1.0
    x_adv = clean_x.clone()
    x_adv[n_orig:] = x_inj.float()
    return adj, x_adv


def _train_contrastive_encoder(
    clean_adj: torch.Tensor,
    clean_x: torch.Tensor,
    adv_adj: torch.Tensor,
    adv_x: torch.Tensor,
    cfg: ContrastiveReconstructionConfig,
) -> ContrastiveModel:
    """Fit the encoder to the clean-extended and poisoned local views."""
    torch.manual_seed(int(cfg.seed))
    encoder = DifferentiableGCNEncoder(clean_x.size(1), int(cfg.hidden_dim)).to(clean_x.device)
    model = ContrastiveModel(
        encoder, int(cfg.hidden_dim), int(cfg.projection_dim), float(cfg.temperature)
    ).to(clean_x.device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay)
    )
    for _ in range(max(1, int(cfg.train_epochs))):
        model.train()
        opt.zero_grad()
        z_clean = model(clean_x, clean_adj)
        z_adv = model(adv_x, adv_adj)
        loss = model.loss(z_clean, z_adv)
        loss.backward()
        opt.step()
    return model


def _extract_gradients(
    model: ContrastiveModel,
    clean_adj: torch.Tensor,
    clean_x: torch.Tensor,
    adv_adj: torch.Tensor,
    adv_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eq. (14)-(15): gradients of L_cl w.r.t. A_gi and X_gi."""
    model.eval()
    adj_var = adv_adj.detach().clone().requires_grad_(True)
    x_var = adv_x.detach().clone().requires_grad_(True)
    z_clean = model(clean_x.detach(), clean_adj.detach())
    z_adv = model(x_var, adj_var)
    loss = model.loss(z_clean.detach(), z_adv)
    loss.backward()
    return adj_var.grad.detach(), x_var.grad.detach(), loss.detach()


@torch.no_grad()
def _edge_action_candidates(
    grad_adj: torch.Tensor,
    cross_targets: Sequence[torch.Tensor],
    anchors: torch.Tensor,
    candidate_regions: Sequence[torch.Tensor],
    node_to_local: Dict[int, int],
    n_orig_local: int,
    budget,
) -> List[Tuple[float, int, int, str]]:
    """Collect Eq. (16) edge flips that do not touch original-original edges."""
    actions: List[Tuple[float, int, int, str]] = []
    used_total = sum(int(t.numel()) for t in cross_targets)
    cross_budget = int(getattr(budget, "cross_edges", used_total))
    deg_cap = int(getattr(budget, "degree_per_injected_node", cross_budget))

    current = [set(int(v) for v in t.detach().cpu().tolist()) for t in cross_targets]
    anchor_ids = [int(v) for v in anchors.detach().cpu().tolist()]
    for inj_idx, cand in enumerate(candidate_regions):
        inj_local = n_orig_local + int(inj_idx)
        if inj_local >= grad_adj.size(0):
            continue
        for v in cand.detach().cpu().tolist():
            iv = int(v)
            if iv not in node_to_local:
                continue
            orig_local = node_to_local[iv]
            grad = 0.5 * (grad_adj[inj_local, orig_local] + grad_adj[orig_local, inj_local])
            gval = float(grad.item())
            exists = iv in current[inj_idx]
            if exists and iv != anchor_ids[inj_idx] and gval < 0.0:
                actions.append((abs(gval), inj_idx, iv, "remove"))
            elif (not exists) and gval > 0.0:
                if len(current[inj_idx]) < deg_cap:
                    actions.append((abs(gval), inj_idx, iv, "add"))
    actions.sort(key=lambda z: z[0], reverse=True)
    return actions


@torch.no_grad()
def _apply_edge_updates(
    cross_targets: Sequence[torch.Tensor],
    actions: Sequence[Tuple[float, int, int, str]],
    cfg: ContrastiveReconstructionConfig,
    budget,
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, int]]:
    """Eq. (16) and Eq. (18): flip selected injected edges and mix back."""
    sets = [set(int(v) for v in t.detach().cpu().tolist()) for t in cross_targets]
    selected_actions = list(actions)[: max(0, int(cfg.actions_per_iteration))]
    removed = 0
    added = 0
    for _, inj_idx, node_id, kind in selected_actions:
        if kind == "remove":
            before = len(sets[inj_idx])
            sets[inj_idx].discard(int(node_id))
            removed += before - len(sets[inj_idx])
    cross_budget = int(getattr(budget, "cross_edges", sum(len(s) for s in sets)))
    deg_cap = int(getattr(budget, "degree_per_injected_node", cross_budget))
    used_total = sum(len(s) for s in sets)
    for _, inj_idx, node_id, kind in selected_actions:
        if kind != "add":
            continue
        if used_total >= cross_budget or len(sets[inj_idx]) >= deg_cap:
            continue
        before = len(sets[inj_idx])
        sets[inj_idx].add(int(node_id))
        delta = len(sets[inj_idx]) - before
        used_total += delta
        added += delta
    updated = [
        torch.tensor(sorted(s), dtype=torch.long, device=device) if s else torch.empty(0, dtype=torch.long, device=device)
        for s in sets
    ]
    return updated, {"added_edges": int(added), "removed_edges": int(removed), "selected_actions": int(len(selected_actions))}


@torch.no_grad()
def _apply_feature_updates(
    x_inj: torch.Tensor,
    grad_x: torch.Tensor,
    n_orig_local: int,
    ref_x: torch.Tensor,
    cfg: ContrastiveReconstructionConfig,
) -> torch.Tensor:
    """Eq. (17): update injected features by largest gradient dimensions."""
    if int(cfg.feature_updates_per_iteration) <= 0:
        return x_inj
    grad_inj = grad_x[n_orig_local : n_orig_local + x_inj.size(0)]
    if grad_inj.numel() == 0:
        return x_inj

    z = x_inj.detach().clone()
    flat = grad_inj.abs().flatten()
    k = min(int(cfg.feature_updates_per_iteration), int(flat.numel()))
    top = torch.topk(flat, k=k, largest=True).indices
    rows = top // grad_inj.size(1)
    cols = top % grad_inj.size(1)
    alpha = float(cfg.feature_step_size)

    lo = 0.0 if bool(((ref_x == 0.0) | (ref_x == 1.0)).float().mean().item() > 0.99) else float(ref_x.min().item())
    hi = 1.0 if bool(((ref_x == 0.0) | (ref_x == 1.0)).float().mean().item() > 0.99) else float(ref_x.max().item())
    z[rows, cols] = (z[rows, cols] + alpha * torch.sign(grad_inj[rows, cols])).clamp(lo, hi)
    return z


def reconstruct_injected_topology(
    g: dgl.DGLGraph,
    x: torch.Tensor,
    x_inj: torch.Tensor,
    cross_targets: Sequence[torch.Tensor],
    anchor_info: Dict[str, torch.Tensor],
    candidate_regions: Sequence[torch.Tensor],
    budget,
    cfg: ContrastiveReconstructionConfig,
) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, object]]:
    """Run contrastive topology reconstruction on the injected structure.

    Returns:
        updated cross_targets, updated x_inj, metadata.
    """
    device = x.device
    g = dgl.remove_self_loop(g).to(device)
    x = x.float().to(device)
    z = x_inj.float().to(device)
    targets = [t.long().to(device) for t in cross_targets]

    original_nodes = _collect_local_original_nodes(anchor_info["anchors"].long(), candidate_regions, targets, cfg)
    clean_adj, clean_x, node_to_local = _make_local_clean_view(g, x, z, original_nodes)

    last_loss = None
    total_added = 0
    total_removed = 0
    total_selected_actions = 0
    for _ in range(max(1, int(cfg.structure_iterations))):
        adv_adj, adv_x = _make_local_adv_view(clean_adj, clean_x, z, targets, node_to_local)
        model = _train_contrastive_encoder(clean_adj, clean_x, adv_adj, adv_x, cfg)
        grad_adj, grad_x, loss = _extract_gradients(model, clean_adj, clean_x, adv_adj, adv_x)
        last_loss = float(loss.item())

        actions = _edge_action_candidates(
            grad_adj=grad_adj,
            cross_targets=targets,
            anchors=anchor_info["anchors"].long(),
            candidate_regions=candidate_regions,
            node_to_local=node_to_local,
            n_orig_local=len(node_to_local),
            budget=budget,
        )
        targets, edge_stats = _apply_edge_updates(targets, actions, cfg, budget, device)
        total_added += int(edge_stats["added_edges"])
        total_removed += int(edge_stats["removed_edges"])
        total_selected_actions += int(edge_stats["selected_actions"])
        z = _apply_feature_updates(z, grad_x, len(node_to_local), x, cfg)

    info = {
        "module": "contrastive_topology_reconstruction",
        "local_original_nodes": int(original_nodes.numel()),
        "num_injected": int(z.size(0)),
        "structure_iterations": int(cfg.structure_iterations),
        "train_epochs": int(cfg.train_epochs),
        "actions_per_iteration": int(cfg.actions_per_iteration),
        "feature_updates_per_iteration": int(cfg.feature_updates_per_iteration),
        "added_edges": int(total_added),
        "removed_edges": int(total_removed),
        "selected_edge_actions": int(total_selected_actions),
        "last_contrastive_loss": last_loss,
    }
    return targets, z, info
