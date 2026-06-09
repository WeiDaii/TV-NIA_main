# -*- coding: utf-8 -*-
"""
SCVNI / SC-VNIA: Sparse Cooperative Vicious Node Injection.

This module contains a clean PyTorch implementation of a white-box sparse
cooperative node injection attack for node classification. It does not contain
any reinforcement-learning component.

Core setting:
    - original nodes, original edges, and original features are fixed;
    - only injected nodes, injected-node features, cross edges, and internal
      injected-node edges are optimized;
    - optimization is white-box and differentiable through a dense GCN victim.

The runner script `main_new.py` loads datasets with the GraphDC-aligned loader
in utils.py, converts the DGL graph to a dense adjacency matrix, trains a clean
victim, runs SCVNI, reports evasion accuracy, and then retrains a fresh victim
on the poisoned graph for poisoning evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover - scipy may be absent in minimal envs
    sp = None

Tensor = torch.Tensor


@dataclass
class SCVNIAConfig:
    """Configuration for SCVNI attack optimization and decoding."""

    # Candidate malicious nodes before pruning.
    n_inj_max: int = 30

    # Final number of kept malicious nodes. If None, use gate_threshold.
    n_inj_final: Optional[int] = 15

    # Candidate original nodes for cross edges are selected from target k-hop region.
    khop: int = 2
    max_candidates: int = 1500

    # White-box optimization.
    steps: int = 300
    lr: float = 0.03
    weight_decay: float = 0.0
    margin_weight: float = 0.5

    # Sparse malicious-resource penalties.
    lambda_active: float = 0.01
    lambda_cross: float = 0.005
    lambda_intra: float = 0.005
    lambda_feat: float = 0.0005

    # Soft budget penalties during optimization.
    edge_budget_cross: int = 120
    edge_budget_intra: int = 30
    lambda_budget_cross: float = 0.05
    lambda_budget_intra: float = 0.05

    # Discretization.
    gate_threshold: float = 0.5
    cross_per_node_budget: Optional[int] = None
    feature_topk: Optional[int] = 50
    feature_threshold: float = 0.5

    # Initialization probabilities for soft masks.
    init_cross_prob: float = 0.03
    init_intra_prob: float = 0.02
    init_gate_prob: float = 0.7

    # Injected features.
    binary_features: bool = True
    feature_init: str = "target_mean"  # target_mean | random

    # Logging.
    log_every: int = 50


def _to_dense_tensor(
    x: Union[np.ndarray, Tensor, Any],
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Convert numpy/scipy/torch matrix to a dense torch tensor."""
    if torch.is_tensor(x):
        if x.is_sparse:
            return x.coalesce().to_dense().to(device=device, dtype=dtype)
        return x.to(device=device, dtype=dtype)
    if sp is not None and sp.issparse(x):
        return torch.tensor(x.toarray(), device=device, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, device=device, dtype=dtype)
    raise TypeError(f"Unsupported matrix type: {type(x)}")


def _to_long_tensor(
    x: Union[np.ndarray, Tensor, Iterable[int]],
    device: Union[str, torch.device],
) -> Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long).view(-1)
    return torch.tensor(list(x), device=device, dtype=torch.long).view(-1)


def normalize_adj_dense(adj: Tensor, add_self_loop: bool = True, eps: float = 1e-12) -> Tensor:
    """
    Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}.

    The attacker optimizes soft edge masks, so this function is intentionally
    written for dense adjacency matrices.
    """
    adj = adj.clamp(min=0.0)
    if add_self_loop:
        eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        adj_hat = adj + eye
    else:
        adj_hat = adj
    deg = adj_hat.sum(dim=1).clamp(min=eps)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt[:, None] * adj_hat * deg_inv_sqrt[None, :]


def default_victim_forward(model: nn.Module, x: Tensor, adj_norm: Tensor) -> Tensor:
    """Default victim interface: logits = model(x, adj_norm)."""
    return model(x, adj_norm)


class SimpleGCN(nn.Module):
    """Small dense GCN used by the SCVNI runner."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: Tensor, adj_norm: Tensor) -> Tensor:
        h = adj_norm @ x
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = adj_norm @ h
        h = self.lin2(h)
        return h


def train_simple_gcn(
    model: nn.Module,
    adj: Tensor,
    features: Tensor,
    labels: Tensor,
    idx_train: Tensor,
    idx_val: Optional[Tensor] = None,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 50,
    verbose: bool = False,
    forward_fn: Callable[[nn.Module, Tensor, Tensor], Tensor] = default_victim_forward,
) -> nn.Module:
    """Train a dense GCN and restore the best validation checkpoint."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    adj_norm = normalize_adj_dense(adj)
    best_val = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad()
        logits = forward_fn(model, features, adj_norm)
        loss = F.cross_entropy(logits[idx_train], labels[idx_train])
        loss.backward()
        opt.step()

        if idx_val is not None and idx_val.numel() > 0:
            model.eval()
            with torch.no_grad():
                logits_eval = forward_fn(model, features, adj_norm)
                pred = logits_eval[idx_val].argmax(dim=1)
                val_acc = (pred == labels[idx_val]).float().mean().item()
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= int(patience):
                    break
            if verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"  [train] epoch={epoch:04d} loss={loss.item():.4f} val={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_node_classification(
    model: nn.Module,
    adj: Tensor,
    features: Tensor,
    labels: Tensor,
    idx_eval: Tensor,
    forward_fn: Callable[[nn.Module, Tensor, Tensor], Tensor] = default_victim_forward,
) -> float:
    model.eval()
    adj_norm = normalize_adj_dense(adj)
    logits = forward_fn(model, features, adj_norm)
    pred = logits[idx_eval].argmax(dim=1)
    return float((pred == labels[idx_eval]).float().mean().item())


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    adj: Tensor,
    features: Tensor,
    forward_fn: Callable[[nn.Module, Tensor, Tensor], Tensor] = default_victim_forward,
) -> Tensor:
    model.eval()
    return forward_fn(model, features, normalize_adj_dense(adj))


@torch.no_grad()
def low_margin_target_selection(
    victim_model: nn.Module,
    features: Tensor,
    adj: Tensor,
    labels_or_pseudo: Tensor,
    idx_pool: Tensor,
    k: int,
    forward_fn: Callable[[nn.Module, Tensor, Tensor], Tensor] = default_victim_forward,
) -> Tensor:
    """
    Select nodes with the smallest classification margin.

    If labels_or_pseudo are clean predictions, this target selection does not
    use test ground-truth labels. If true labels are passed, it becomes a
    label-aware upper-bound target selection.
    """
    logits = predict_logits(victim_model, adj, features, forward_fn)
    pool_logits = logits[idx_pool]
    y = labels_or_pseudo[idx_pool]
    true_logits = pool_logits.gather(1, y.view(-1, 1)).squeeze(1)
    wrong_logits = pool_logits.clone()
    wrong_logits.scatter_(1, y.view(-1, 1), -1e9)
    max_wrong = wrong_logits.max(dim=1).values
    margin = true_logits - max_wrong
    order = torch.argsort(margin, descending=False)
    k = min(int(k), idx_pool.numel())
    return idx_pool[order[:k]]


def build_khop_candidate_pool(adj: Tensor, idx_attack: Tensor, khop: int = 2, max_candidates: int = 1500) -> Tensor:
    """Build a candidate original-node pool from the k-hop region of attack targets."""
    device = adj.device
    n = adj.size(0)
    adj_bin = adj > 0
    idx_attack = idx_attack.unique()

    visited = torch.zeros(n, dtype=torch.bool, device=device)
    frontier = torch.zeros(n, dtype=torch.bool, device=device)
    frontier[idx_attack] = True
    visited |= frontier

    for _ in range(int(khop)):
        if frontier.sum() == 0:
            break
        neigh = adj_bin[frontier].any(dim=0)
        frontier = neigh & (~visited)
        visited |= neigh

    candidates = torch.where(visited)[0]
    if candidates.numel() > int(max_candidates):
        deg = adj.sum(dim=1)
        target_set = idx_attack.unique()
        is_target = torch.zeros(n, dtype=torch.bool, device=device)
        is_target[target_set] = True
        non_target = candidates[~is_target[candidates]]
        rest_budget = max(int(max_candidates) - target_set.numel(), 0)
        if rest_budget > 0 and non_target.numel() > 0:
            order = torch.argsort(deg[non_target], descending=True)
            candidates = torch.cat([target_set, non_target[order[:rest_budget]]]).unique()
        else:
            candidates = target_set
    return candidates.unique()


class SCVNIAAttacker:
    """White-box sparse cooperative node injection attacker."""

    def __init__(
        self,
        victim_model: nn.Module,
        num_classes: int,
        config: Optional[SCVNIAConfig] = None,
        forward_fn: Callable[[nn.Module, Tensor, Tensor], Tensor] = default_victim_forward,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.victim_model = victim_model.to(device)
        self.num_classes = int(num_classes)
        self.config = config or SCVNIAConfig()
        self.forward_fn = forward_fn
        self.device = torch.device(device)

        self.victim_model.eval()
        for p in self.victim_model.parameters():
            p.requires_grad_(False)

        self.params_initialized = False

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    def _make_candidate_selector(self, candidate_nodes: Tensor, n_orig: int) -> Tensor:
        c = candidate_nodes.numel()
        selector = torch.zeros(c, n_orig, device=self.device)
        selector[torch.arange(c, device=self.device), candidate_nodes] = 1.0
        return selector

    def _init_attack_parameters(self, features: Tensor, idx_attack: Tensor, candidate_nodes: Tensor) -> None:
        cfg = self.config
        n_inj = int(cfg.n_inj_max)
        feat_dim = int(features.size(1))
        num_candidates = int(candidate_nodes.numel())

        cross_init = self._logit(cfg.init_cross_prob)
        intra_init = self._logit(cfg.init_intra_prob)
        gate_init = self._logit(cfg.init_gate_prob)

        self.cross_logits = nn.Parameter(
            torch.full((n_inj, num_candidates), cross_init, device=self.device)
            + 0.01 * torch.randn(n_inj, num_candidates, device=self.device)
        )
        self.intra_logits = nn.Parameter(
            torch.full((n_inj, n_inj), intra_init, device=self.device)
            + 0.01 * torch.randn(n_inj, n_inj, device=self.device)
        )
        self.gate_logits = nn.Parameter(
            torch.full((n_inj,), gate_init, device=self.device)
            + 0.01 * torch.randn(n_inj, device=self.device)
        )

        if cfg.feature_init == "target_mean":
            init_feat = features[idx_attack].mean(dim=0).clamp(1e-4, 1.0 - 1e-4)
            init_feat = torch.logit(init_feat).repeat(n_inj, 1)
            init_feat = init_feat + 0.01 * torch.randn_like(init_feat)
        elif cfg.feature_init == "random":
            raw = torch.rand(n_inj, feat_dim, device=self.device).clamp(1e-4, 1.0 - 1e-4)
            init_feat = torch.logit(raw)
        else:
            raise ValueError(f"Unknown feature_init: {cfg.feature_init}")
        self.feature_logits = nn.Parameter(init_feat)
        self.params_initialized = True

    def _soft_variables(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        gate = torch.sigmoid(self.gate_logits)  # [m]

        p_cross = torch.sigmoid(self.cross_logits) * gate[:, None]

        p_intra_raw = torch.sigmoid(self.intra_logits)
        p_intra_upper = torch.triu(p_intra_raw, diagonal=1)
        p_intra = p_intra_upper + p_intra_upper.t()
        p_intra = p_intra * (gate[:, None] * gate[None, :])

        x_inj = torch.sigmoid(self.feature_logits)
        return gate, p_cross, p_intra, x_inj

    def _build_soft_augmented_graph(
        self,
        adj_orig: Tensor,
        features_orig: Tensor,
        candidate_selector: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        gate, p_cross, p_intra, x_inj = self._soft_variables()
        b_out = p_cross @ candidate_selector  # [m, N]
        top = torch.cat([adj_orig, b_out.t()], dim=1)
        bottom = torch.cat([b_out, p_intra], dim=1)
        adj_aug = torch.cat([top, bottom], dim=0)
        features_aug = torch.cat([features_orig, x_inj], dim=0)
        aux = {"gate": gate, "p_cross": p_cross, "p_intra": p_intra, "x_inj": x_inj, "b_out": b_out}
        return adj_aug, features_aug, aux

    def _attack_objective(self, logits: Tensor, labels: Tensor, idx_attack: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_logits = logits[idx_attack]
        y = labels[idx_attack]
        ce = F.cross_entropy(target_logits, y)
        true_logits = target_logits.gather(1, y.view(-1, 1)).squeeze(1)
        wrong_logits = target_logits.clone()
        wrong_logits.scatter_(1, y.view(-1, 1), -1e9)
        max_wrong = wrong_logits.max(dim=1).values
        adv_margin = (max_wrong - true_logits).mean()
        atk_obj = ce + self.config.margin_weight * adv_margin
        return atk_obj, {"ce": ce.detach(), "adv_margin": adv_margin.detach(), "atk_obj": atk_obj.detach()}

    def _regularization(self, aux: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        cfg = self.config
        gate = aux["gate"]
        p_cross = aux["p_cross"]
        p_intra = aux["p_intra"]
        x_inj = aux["x_inj"]

        active_pen = gate.sum()
        cross_pen = p_cross.sum()
        intra_pen = torch.triu(p_intra, diagonal=1).sum()
        feat_pen = x_inj.sum() if cfg.binary_features else x_inj.abs().sum()
        budget_cross_pen = F.relu(cross_pen - cfg.edge_budget_cross).pow(2)
        budget_intra_pen = F.relu(intra_pen - cfg.edge_budget_intra).pow(2)

        reg = (
            cfg.lambda_active * active_pen
            + cfg.lambda_cross * cross_pen
            + cfg.lambda_intra * intra_pen
            + cfg.lambda_feat * feat_pen
            + cfg.lambda_budget_cross * budget_cross_pen
            + cfg.lambda_budget_intra * budget_intra_pen
        )
        return reg, {
            "active_pen": active_pen.detach(),
            "cross_pen": cross_pen.detach(),
            "intra_pen": intra_pen.detach(),
            "feat_pen": feat_pen.detach(),
            "budget_cross_pen": budget_cross_pen.detach(),
            "budget_intra_pen": budget_intra_pen.detach(),
            "reg": reg.detach(),
        }

    def attack(
        self,
        adj: Union[np.ndarray, Tensor, Any],
        features: Union[np.ndarray, Tensor],
        labels: Union[np.ndarray, Tensor, List[int]],
        idx_attack: Union[np.ndarray, Tensor, List[int]],
        candidate_nodes: Optional[Union[np.ndarray, Tensor, List[int]]] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Run white-box SCVNI optimization and return a discrete poisoned graph."""
        cfg = self.config
        adj_orig = _to_dense_tensor(adj, self.device, torch.float32)
        features_orig = _to_dense_tensor(features, self.device, torch.float32)
        labels = _to_long_tensor(labels, self.device)
        idx_attack = _to_long_tensor(idx_attack, self.device).unique()

        adj_orig = ((adj_orig + adj_orig.t()) > 0).float()
        adj_orig.fill_diagonal_(0.0)
        n_orig = int(adj_orig.size(0))

        if candidate_nodes is None:
            candidate_nodes = build_khop_candidate_pool(adj_orig, idx_attack, khop=cfg.khop, max_candidates=cfg.max_candidates)
        else:
            candidate_nodes = _to_long_tensor(candidate_nodes, self.device).unique()
        if candidate_nodes.numel() == 0:
            candidate_nodes = idx_attack.unique()

        candidate_selector = self._make_candidate_selector(candidate_nodes, n_orig)
        self._init_attack_parameters(features_orig, idx_attack, candidate_nodes)

        optimizer = torch.optim.Adam(
            [self.cross_logits, self.intra_logits, self.gate_logits, self.feature_logits],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        history: List[Dict[str, float]] = []
        for step in range(int(cfg.steps)):
            optimizer.zero_grad()
            adj_aug, features_aug, aux = self._build_soft_augmented_graph(adj_orig, features_orig, candidate_selector)
            logits_aug = self.forward_fn(self.victim_model, features_aug, normalize_adj_dense(adj_aug))
            atk_obj, atk_logs = self._attack_objective(logits_aug[:n_orig], labels, idx_attack)
            reg, reg_logs = self._regularization(aux)
            loss = -atk_obj + reg
            loss.backward()
            optimizer.step()

            if cfg.log_every > 0 and (step % int(cfg.log_every) == 0 or step == int(cfg.steps) - 1):
                history.append(
                    {
                        "step": int(step),
                        "loss": float(loss.detach().cpu()),
                        "atk_obj": float(atk_logs["atk_obj"].cpu()),
                        "ce": float(atk_logs["ce"].cpu()),
                        "adv_margin": float(atk_logs["adv_margin"].cpu()),
                        "reg": float(reg_logs["reg"].cpu()),
                        "active_soft": float(reg_logs["active_pen"].cpu()),
                        "cross_soft": float(reg_logs["cross_pen"].cpu()),
                        "intra_soft": float(reg_logs["intra_pen"].cpu()),
                    }
                )

        adj_hard, features_hard, decode_info = self.decode(adj_orig, features_orig, candidate_nodes)
        info = {"history": history, "candidate_nodes": candidate_nodes.detach().cpu(), **decode_info}
        return adj_hard.detach(), features_hard.detach(), info

    def decode(self, adj_orig: Tensor, features_orig: Tensor, candidate_nodes: Tensor) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Decode soft variables into a hard injected graph."""
        cfg = self.config
        n_orig = int(adj_orig.size(0))
        with torch.no_grad():
            gate, p_cross, p_intra, x_inj = self._soft_variables()

            if cfg.n_inj_final is not None:
                k_active = min(int(cfg.n_inj_final), int(cfg.n_inj_max))
                active_ids = torch.argsort(gate, descending=True)[:k_active]
            else:
                active_ids = torch.where(gate >= cfg.gate_threshold)[0]
                if active_ids.numel() == 0:
                    active_ids = torch.argsort(gate, descending=True)[:1]
            active_ids = active_ids.sort().values
            m_keep = int(active_ids.numel())

            p_cross_keep = p_cross[active_ids]
            x_keep = x_inj[active_ids]

            b_out_hard = torch.zeros(m_keep, n_orig, device=self.device)
            if cfg.cross_per_node_budget is not None:
                per_k = int(cfg.cross_per_node_budget)
                for local_i in range(m_keep):
                    row = p_cross_keep[local_i]
                    k = min(per_k, int(row.numel()))
                    if k > 0:
                        top_c = torch.argsort(row, descending=True)[:k]
                        b_out_hard[local_i, candidate_nodes[top_c]] = 1.0
            else:
                total_k = min(int(cfg.edge_budget_cross), int(p_cross_keep.numel()))
                if total_k > 0:
                    flat = p_cross_keep.reshape(-1)
                    top_flat = torch.argsort(flat, descending=True)[:total_k]
                    row_ids = top_flat // p_cross_keep.size(1)
                    col_ids = top_flat % p_cross_keep.size(1)
                    b_out_hard[row_ids, candidate_nodes[col_ids]] = 1.0
                for local_i in range(m_keep):
                    if b_out_hard[local_i].sum() == 0:
                        best_c = torch.argmax(p_cross_keep[local_i])
                        b_out_hard[local_i, candidate_nodes[best_c]] = 1.0

            a_inj_hard = torch.zeros(m_keep, m_keep, device=self.device)
            if int(cfg.edge_budget_intra) > 0 and m_keep > 1:
                p_intra_keep = p_intra[active_ids][:, active_ids]
                upper = torch.triu_indices(m_keep, m_keep, offset=1, device=self.device)
                scores = p_intra_keep[upper[0], upper[1]]
                total_intra = min(int(cfg.edge_budget_intra), int(scores.numel()))
                if total_intra > 0:
                    top_e = torch.argsort(scores, descending=True)[:total_intra]
                    src = upper[0][top_e]
                    dst = upper[1][top_e]
                    a_inj_hard[src, dst] = 1.0
                    a_inj_hard[dst, src] = 1.0

            if cfg.binary_features:
                if cfg.feature_topk is not None:
                    x_hard = torch.zeros_like(x_keep)
                    kf = min(int(cfg.feature_topk), int(x_keep.size(1)))
                    if kf > 0:
                        top_f = torch.argsort(x_keep, dim=1, descending=True)[:, :kf]
                        rows = torch.arange(m_keep, device=self.device).view(-1, 1).repeat(1, kf)
                        x_hard[rows, top_f] = 1.0
                else:
                    x_hard = (x_keep >= cfg.feature_threshold).float()
            else:
                x_hard = x_keep.clamp(0.0, 1.0)

            top = torch.cat([adj_orig, b_out_hard.t()], dim=1)
            bottom = torch.cat([b_out_hard, a_inj_hard], dim=1)
            adj_hard = torch.cat([top, bottom], dim=0)
            adj_hard.fill_diagonal_(0.0)
            features_hard = torch.cat([features_orig, x_hard], dim=0)

            cross_edges = []
            for local_i, orig_j in torch.nonzero(b_out_hard > 0, as_tuple=False).tolist():
                cross_edges.append((n_orig + int(local_i), int(orig_j)))
            intra_edges = []
            for local_i, local_j in torch.nonzero(torch.triu(a_inj_hard, diagonal=1) > 0, as_tuple=False).tolist():
                intra_edges.append((n_orig + int(local_i), n_orig + int(local_j)))

            info = {
                "active_original_param_ids": active_ids.detach().cpu(),
                "num_active_injected_nodes": m_keep,
                "num_cross_edges": int(b_out_hard.sum().item()),
                "num_intra_edges": int(torch.triu(a_inj_hard, diagonal=1).sum().item()),
                "cross_edges": cross_edges,
                "intra_edges": intra_edges,
                "gate_values": gate.detach().cpu(),
            }
        return adj_hard, features_hard, info
