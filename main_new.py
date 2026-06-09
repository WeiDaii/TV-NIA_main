# -*- coding: utf-8 -*-
"""
Clean SCVNI runner.

This file intentionally removes the original reinforcement-learning attack
pipeline. Dataset loading is delegated to utils.load_original_graph(...), whose
`graphdc_aligned` split option follows the GraphDC-style dataset/split handling
already implemented in utils.py.

Pipeline:
    1. Load graph/features/labels/splits with GraphDC-aligned loader.
    2. Convert the DGL graph to dense adjacency for differentiable SCVNI.
    3. Train a clean victim from victim_models.py.
    4. Run SCVNI to generate injected nodes/edges/features.
    5. Report evasion accuracy on the fixed clean victim.
    6. Retrain a fresh victim on the poisoned graph and report poisoning accuracy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import dgl
import torch
import torch.nn.functional as F

from utils import load_original_graph, set_seed
from sc_vnia import (
    SCVNIAConfig,
    SCVNIAAttacker,
    low_margin_target_selection,
)
from victim_models import Victim

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR
VICTIM_CHOICES = ["gcn", "gat", "sage", "graphsage", "appnp", "gin", "sgc", "simpgcn"]


def canonical_victim_name(name: str) -> str:
    name = str(name).lower()
    if name in {"graphsage", "graph_sage"}:
        return "sage"
    return name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SCVNI: clean white-box sparse cooperative node injection")

    # Data / split. `graphdc_aligned` is implemented in utils.py.
    p.add_argument("--dataset", default="cora")
    p.add_argument("--data_root", default=str(PROJECT_ROOT / "data"))
    p.add_argument("--split_name", default="graphdc_aligned")
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)

    # Device / reproducibility.
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # Victim parameters are aligned with g2a2c_poison.py / victim_models.py.
    p.add_argument("--victim", default="gcn", choices=VICTIM_CHOICES)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--clean_epochs", type=int, default=200)
    p.add_argument("--poison_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--num_heads1", type=int, default=8)
    p.add_argument("--num_heads2", type=int, default=1)
    p.add_argument("--feat_dropout", type=float, default=0.5)
    p.add_argument("--attn_dropout", type=float, default=0.5)
    p.add_argument("--sage_agg_type", default="mean")
    p.add_argument("--sage_dropout", type=float, default=0.5)
    p.add_argument("--appnp_hidden", type=int, default=64)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gin_hidden", type=int, default=64)
    p.add_argument("--gin_agg_type", default="sum")

    # Attack target selection.
    p.add_argument("--target_limit", type=int, default=200)
    p.add_argument("--attack_label_mode", default="pseudo", choices=["pseudo", "true"])
    p.add_argument("--only_correct", action="store_true")

    # SCVNI budgets / optimization.
    p.add_argument("--n_inj_max", type=int, default=30)
    p.add_argument("--n_inj_final", type=int, default=15)
    p.add_argument("--khop", type=int, default=2)
    p.add_argument("--max_candidates", type=int, default=1500)
    p.add_argument("--attack_steps", type=int, default=300)
    p.add_argument("--attack_lr", type=float, default=0.03)
    p.add_argument("--edge_budget_cross", type=int, default=120)
    p.add_argument("--edge_budget_intra", type=int, default=30)
    p.add_argument("--cross_per_node_budget", type=int, default=-1)
    p.add_argument("--feature_topk", type=int, default=50)
    p.add_argument("--margin_weight", type=float, default=0.5)
    p.add_argument("--lambda_active", type=float, default=0.01)
    p.add_argument("--lambda_cross", type=float, default=0.005)
    p.add_argument("--lambda_intra", type=float, default=0.005)
    p.add_argument("--lambda_feat", type=float, default=0.0005)
    p.add_argument("--log_every", type=int, default=50)

    args = p.parse_args()
    args.victim = canonical_victim_name(args.victim)
    return args


def build_loader_args(args: argparse.Namespace) -> SimpleNamespace:
    """Arguments required by utils.load_original_graph and GraphDC-aligned splits."""
    return SimpleNamespace(
        dataset=args.dataset,
        data_root=args.data_root,
        split_name=args.split_name,
        model_name=args.victim,
        victim=args.victim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_id=args.split_id,
        seed=args.seed,
        # The following fields are used by victim_models.Victim and older loader code.
        hidden=args.hidden,
        dropout=args.dropout,
        num_heads1=args.num_heads1,
        num_heads2=args.num_heads2,
        feat_dropout=args.feat_dropout,
        attn_dropout=args.attn_dropout,
        sage_agg_type=args.sage_agg_type,
        sage_dropout=args.sage_dropout,
        appnp_hidden=args.appnp_hidden,
        K=args.K,
        alpha=args.alpha,
        gin_hidden=args.gin_hidden,
        gin_agg_type=args.gin_agg_type,
    )


def resolve_device(device_str: str) -> torch.device:
    requested = torch.device(device_str)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"[Warning] requested {device_str}, but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    if requested.type == "cuda":
        torch.cuda.set_device(requested)
    return requested


def dgl_graph_to_dense_adj_no_loop(graph, device: torch.device) -> torch.Tensor:
    """Convert DGL graph to a dense 0/1 adjacency matrix and remove self-loops."""
    n = graph.num_nodes()
    src, dst = graph.edges()
    adj = torch.zeros((n, n), dtype=torch.float32, device=device)
    adj[src.to(device), dst.to(device)] = 1.0
    adj = ((adj + adj.t()) > 0).float()
    adj.fill_diagonal_(0.0)
    return adj


def extend_labels_for_injected_nodes(labels: torch.Tensor, n_injected: int) -> torch.Tensor:
    """Injected labels are placeholders; injected nodes are not used for training/evaluation."""
    if n_injected <= 0:
        return labels
    filler = torch.zeros(n_injected, dtype=labels.dtype, device=labels.device)
    return torch.cat([labels, filler], dim=0)


def make_victim(args: argparse.Namespace, common_args: SimpleNamespace, feats: torch.Tensor, labels: torch.Tensor, device):
    return Victim(
        model_name=args.victim,
        in_dim=feats.shape[1],
        n_classes=int(labels.max().item() + 1),
        lr=args.lr,
        wd=args.wd,
        args=common_args,
        device=device,
    )


def dense_adj_to_dgl_graph(
    adj: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    add_self_loops: bool = True,
):
    adj_bin = (adj > 0).detach().clone()
    if add_self_loops:
        adj_bin.fill_diagonal_(True)
    src, dst = torch.nonzero(adj_bin, as_tuple=True)
    src = src.long()
    dst = dst.long()
    device = features.device
    try:
        graph = dgl.graph((src, dst), num_nodes=int(adj_bin.size(0)), device=device)
    except TypeError:
        graph = dgl.graph((src.cpu(), dst.cpu()), num_nodes=int(adj_bin.size(0))).to(device)
    graph.ndata["feat"] = features
    graph.ndata["label"] = labels
    return graph


@torch.no_grad()
def accuracy_from_victim(victim: Victim, graph, feats: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor) -> float:
    logits = victim.logits(graph, feats)
    return float((logits[idx].argmax(-1) == labels[idx]).float().mean().item())


def _layer_param(layer, *names):
    for name in names:
        if hasattr(layer, name):
            value = getattr(layer, name)
            if value is not None:
                return value
    return None


def _apply_weight_bias(layer, x: torch.Tensor) -> torch.Tensor:
    weight = _layer_param(layer, "weight", "_weight")
    bias = _layer_param(layer, "bias", "_bias")
    if weight is not None:
        x = x @ weight
    if bias is not None:
        x = x + bias
    return x


def _row_normalize_dense(adj: torch.Tensor) -> torch.Tensor:
    return adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-12)


def _drop_diagonal(adj: torch.Tensor) -> torch.Tensor:
    out = adj.clone()
    out.fill_diagonal_(0.0)
    return out


def _dense_graphconv_layer(layer, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    return _apply_weight_bias(layer, adj_norm @ x)


def _dense_gcn_forward(model, features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    h = _dense_graphconv_layer(model.layer1, features, adj_norm)
    h = _dense_graphconv_layer(model.layer2, h, adj_norm)
    return h


def _dense_sage_layer(layer, x: torch.Tensor, adj_norm: torch.Tensor, aggregator_type: str) -> torch.Tensor:
    aggr = str(aggregator_type).lower()
    adj_no_loop = _drop_diagonal(adj_norm)

    if aggr == "gcn":
        neigh = (adj_no_loop @ x + x) / (adj_no_loop.sum(dim=1, keepdim=True) + 1.0).clamp(min=1e-12)
        out = layer.fc_neigh(neigh) if hasattr(layer, "fc_neigh") else _apply_weight_bias(layer, neigh)
        bias = _layer_param(layer, "bias", "_bias")
        if bias is not None:
            out = out + bias
        return out

    if aggr == "pool" and hasattr(layer, "fc_pool"):
        pooled = F.relu(layer.fc_pool(x))
        mask = adj_no_loop > 0
        expanded = pooled.unsqueeze(0).expand(adj_no_loop.size(0), -1, -1)
        neg_inf = torch.full_like(expanded, -1e9)
        neigh = torch.where(mask.unsqueeze(-1), expanded, neg_inf).max(dim=1).values
        neigh = torch.where(mask.any(dim=1, keepdim=True), neigh, torch.zeros_like(neigh))
    else:
        neigh = _row_normalize_dense(adj_no_loop) @ x

    if hasattr(layer, "fc_neigh"):
        out = layer.fc_neigh(neigh)
    else:
        out = _apply_weight_bias(layer, neigh)
    if hasattr(layer, "fc_self"):
        out = out + layer.fc_self(x)
    return out


def _dense_sage_forward(model, features: torch.Tensor, adj_norm: torch.Tensor, aggregator_type: str) -> torch.Tensor:
    h = _dense_sage_layer(model.layer1, features, adj_norm, aggregator_type)
    h = F.relu(h)
    dropout = getattr(model, "dropout", None)
    if dropout is not None:
        h = dropout(h)
    h = _dense_sage_layer(model.layer2, h, adj_norm, aggregator_type)
    return h


def _dense_gin_aggregate(adj: torch.Tensor, x: torch.Tensor, aggregator_type: str) -> torch.Tensor:
    aggr = str(aggregator_type).lower()
    if aggr == "mean":
        return _row_normalize_dense(adj) @ x
    if aggr == "max":
        mask = adj > 0
        expanded = x.unsqueeze(0).expand(adj.size(0), -1, -1)
        neg_inf = torch.full_like(expanded, -1e9)
        return torch.where(mask.unsqueeze(-1), expanded, neg_inf).max(dim=1).values
    return adj @ x


def _dense_gin_layer(layer, apply_func, x: torch.Tensor, adj_norm: torch.Tensor, aggregator_type: str) -> torch.Tensor:
    eps = _layer_param(layer, "eps", "_eps")
    if eps is None:
        eps = 0.0
    neigh = _dense_gin_aggregate(adj_norm, x, aggregator_type)
    out = (1.0 + eps) * x + neigh
    if apply_func is not None:
        out = apply_func(out)
    return out


def _dense_gin_forward(model, features: torch.Tensor, adj_norm: torch.Tensor, aggregator_type: str) -> torch.Tensor:
    h = _dense_gin_layer(model.layer1, getattr(model, "apply_func1", None), features, adj_norm, aggregator_type)
    h = F.elu(h)
    h = _dense_gin_layer(model.layer2, getattr(model, "apply_func2", None), h, adj_norm, aggregator_type)
    h = F.elu(h)
    return h


def _dense_sgconv_layer(layer, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    k = int(getattr(layer, "_k", getattr(layer, "k", 1)))
    h = x
    for _ in range(max(k, 1)):
        h = adj_norm @ h
    if hasattr(layer, "fc"):
        return layer.fc(h)
    if hasattr(layer, "lin"):
        return layer.lin(h)
    return _apply_weight_bias(layer, h)


def _dense_sgc_forward(model, features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    h = _dense_sgconv_layer(model.layer1, features, adj_norm)
    h = F.elu(h)
    h = _dense_sgconv_layer(model.layer2, h, adj_norm)
    return h


def _dense_appnp_prop(layer, x: torch.Tensor, adj_norm: torch.Tensor, default_k: int, default_alpha: float) -> torch.Tensor:
    k = int(getattr(layer, "_k", getattr(layer, "k", default_k)))
    alpha = float(getattr(layer, "_alpha", getattr(layer, "alpha", default_alpha)))
    h0 = x
    h = x
    for _ in range(max(k, 0)):
        h = (1.0 - alpha) * (adj_norm @ h) + alpha * h0
    return h


def _dense_appnp_forward(
    model,
    features: torch.Tensor,
    adj_norm: torch.Tensor,
    default_k: int,
    default_alpha: float,
) -> torch.Tensor:
    h = model.fc1(features)
    h = _dense_appnp_prop(model.layer1, h, adj_norm, default_k, default_alpha)
    h = F.elu(model.fc2(h))
    h = _dense_appnp_prop(model.layer2, h, adj_norm, default_k, default_alpha)
    h = F.elu(h)
    return h


def _dense_gat_layer(layer, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    if hasattr(layer, "fc"):
        feat_src = layer.fc(x)
        feat_dst = feat_src
    else:
        feat_src = layer.fc_src(x)
        feat_dst = layer.fc_dst(x)

    attn_l = _layer_param(layer, "attn_l", "attn_src")
    attn_r = _layer_param(layer, "attn_r", "attn_dst")
    if attn_l is None or attn_r is None:
        raise RuntimeError("Unsupported GATConv layout: missing attention parameters.")

    num_heads = int(attn_l.shape[1])
    out_feats = int(attn_l.shape[2])
    feat_src = feat_src.view(x.size(0), num_heads, out_feats)
    feat_dst = feat_dst.view(x.size(0), num_heads, out_feats)

    el = (feat_src * attn_l).sum(dim=-1)
    er = (feat_dst * attn_r).sum(dim=-1)
    scores = el[:, None, :] + er[None, :, :]
    leaky_relu = getattr(layer, "leaky_relu", None)
    if leaky_relu is not None:
        scores = leaky_relu(scores)
    else:
        slope = float(getattr(layer, "_negative_slope", 0.2))
        scores = F.leaky_relu(scores, negative_slope=slope)

    weights = adj_norm.clamp(min=1e-12)
    masked_scores = scores + weights.log().unsqueeze(-1)
    masked_scores = torch.where((adj_norm > 0).unsqueeze(-1), masked_scores, torch.full_like(masked_scores, -1e9))
    alpha = F.softmax(masked_scores, dim=0)
    out = torch.einsum("ijh,ihd->jhd", alpha, feat_src)

    bias = _layer_param(layer, "bias", "bias_param")
    if bias is not None:
        out = out + bias.view(1, num_heads, out_feats)

    res_fc = getattr(layer, "res_fc", None)
    if res_fc is not None:
        try:
            out = out + res_fc(x).view(x.size(0), num_heads, out_feats)
        except Exception:
            pass

    activation = getattr(layer, "activation", None)
    if activation is not None:
        out = activation(out)
    return out


def _dense_gat_forward(model, features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    h = _dense_gat_layer(model.layer1, features, adj_norm)
    h = h.flatten(1)
    h = F.elu(h)
    h = _dense_gat_layer(model.layer2, h, adj_norm)
    h = h.squeeze(1)
    if int(getattr(model, "num_heads", [1, 1])[1]) > 1:
        h = torch.mean(h, dim=1)
    h = F.elu(h)
    return h


def _dense_feature_knn_norm(x: torch.Tensor, k: int) -> torch.Tensor:
    n = int(x.size(0))
    if n <= 1:
        return torch.zeros((n, n), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        kk = min(int(k), n - 1)
        xn = F.normalize(x.detach(), p=2, dim=1)
        sim = xn @ xn.t()
        sim.fill_diagonal_(-1.0)
        vals, idx = torch.topk(sim, k=kk, dim=1, largest=True)
        rows = torch.arange(n, device=x.device).view(-1, 1).expand_as(idx).reshape(-1)
        cols = idx.reshape(-1)
        vals = vals.reshape(-1).clamp(min=0.0)
        sf = torch.zeros((n, n), dtype=x.dtype, device=x.device)
        sf[rows, cols] = vals
        sf = torch.maximum(sf, sf.t())
        deg = sf.sum(dim=1).clamp(min=1e-12).pow(-0.5)
        sf = deg[:, None] * sf * deg[None, :]
    return sf


def _dense_simpgcn_forward(model, features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
    if not hasattr(model, "W"):
        raise RuntimeError("Unsupported SimPGCN layout: missing W layers.")
    h = model.idrop(features) if hasattr(model, "idrop") else features
    sf = _dense_feature_knn_norm(h, int(getattr(model, "k", 20)))
    gamma = float(getattr(model, "gamma", 0.1))
    first_hidden = None

    for layer_id, linear in enumerate(model.W):
        s = torch.sigmoid(model.s_mlps[layer_id](h)).squeeze(-1)
        k_gate = model.k_mlps[layer_id](h).squeeze(-1)
        ah = adj_norm @ h
        sfh = sf @ h
        ph = s.unsqueeze(1) * ah + (1.0 - s).unsqueeze(1) * sfh + gamma * (k_gate.unsqueeze(1) * h)
        h = linear(ph)
        if layer_id < len(model.W) - 1:
            h = F.relu(h)
            if getattr(model, "bn", None) is not None:
                h = model.bn[layer_id](h)
            if hasattr(model, "drop"):
                h = model.drop(h)
            if first_hidden is None:
                first_hidden = h

    if first_hidden is not None:
        model.last_embedding = first_hidden
    return h


class DenseVictimForward:
    """Dense differentiable forward adapter for victim_models.py victims."""

    def __init__(self, args: argparse.Namespace):
        self.model_name = canonical_victim_name(args.victim)
        self.sage_agg_type = args.sage_agg_type
        self.gin_agg_type = args.gin_agg_type
        self.K = args.K
        self.alpha = args.alpha

    def __call__(self, model, features: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        if self.model_name == "gcn":
            return _dense_gcn_forward(model, features, adj_norm)
        if self.model_name == "sage":
            return _dense_sage_forward(model, features, adj_norm, self.sage_agg_type)
        if self.model_name == "gin":
            return _dense_gin_forward(model, features, adj_norm, self.gin_agg_type)
        if self.model_name == "sgc":
            return _dense_sgc_forward(model, features, adj_norm)
        if self.model_name == "appnp":
            return _dense_appnp_forward(model, features, adj_norm, self.K, self.alpha)
        if self.model_name == "gat":
            return _dense_gat_forward(model, features, adj_norm)
        if self.model_name == "simpgcn":
            return _dense_simpgcn_forward(model, features, adj_norm)
        raise ValueError(f"Unsupported victim model for dense SCVNI forward: {self.model_name}")


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    set_seed(args.seed)
    device = resolve_device(args.device)
    print(f"[Device] {device}")

    loader_args = build_loader_args(args)
    graph, features, labels, idx_train, idx_val, idx_test = load_original_graph(args.dataset, device, loader_args)
    features = features.float().to(device)
    labels = labels.long().to(device)
    idx_train = idx_train.long().to(device)
    idx_val = idx_val.long().to(device)
    idx_test = idx_test.long().to(device)
    adj = dgl_graph_to_dense_adj_no_loop(graph, device)
    victim_graph = dense_adj_to_dgl_graph(adj, features, labels, add_self_loops=True)
    dense_forward = DenseVictimForward(args)

    num_classes = int(labels.max().item() + 1)
    print(
        f"[Data] dataset={args.dataset} | N={adj.size(0)}, E_undir={int(adj.sum().item() // 2)}, "
        f"F={features.size(1)}, C={num_classes}, train={idx_train.numel()}, "
        f"val={idx_val.numel()}, test={idx_test.numel()} | split={args.split_name}"
    )

    # 1. Clean victim training.
    set_seed(args.seed)
    clean_victim = make_victim(args, loader_args, features, labels, device)
    clean_result = clean_victim.fit_eval(
        victim_graph,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        epochs=args.clean_epochs,
        patience=args.patience,
        verbose=False,
    )
    clean_acc = float(clean_result["test_acc"])
    print(f"[Clean] test_acc={clean_acc:.4f}")

    # 2. Attack-label construction. Default uses clean predictions as pseudo labels,
    # avoiding test-label use during attack generation.
    pseudo_labels = clean_victim.predict(victim_graph, features).detach()
    attack_labels = labels if args.attack_label_mode == "true" else pseudo_labels

    target_pool = idx_test
    if args.only_correct:
        target_pool = idx_test[pseudo_labels[idx_test] == labels[idx_test]]
        print(f"[Targets] only_correct enabled: {target_pool.numel()} correctly predicted test nodes")
    if target_pool.numel() == 0:
        raise RuntimeError("No target nodes available for SCVNI.")

    idx_attack = low_margin_target_selection(
        victim_model=clean_victim.model,
        features=features,
        adj=adj,
        labels_or_pseudo=attack_labels,
        idx_pool=target_pool,
        k=args.target_limit,
        forward_fn=dense_forward,
    )
    print(
        f"[Targets] selected={idx_attack.numel()} | pool={target_pool.numel()} | "
        f"attack_label_mode={args.attack_label_mode}"
    )

    # 3. SCVNI attack.
    cfg = SCVNIAConfig(
        n_inj_max=args.n_inj_max,
        n_inj_final=args.n_inj_final,
        khop=args.khop,
        max_candidates=args.max_candidates,
        steps=args.attack_steps,
        lr=args.attack_lr,
        edge_budget_cross=args.edge_budget_cross,
        edge_budget_intra=args.edge_budget_intra,
        cross_per_node_budget=None if args.cross_per_node_budget < 0 else args.cross_per_node_budget,
        feature_topk=None if args.feature_topk < 0 else args.feature_topk,
        margin_weight=args.margin_weight,
        lambda_active=args.lambda_active,
        lambda_cross=args.lambda_cross,
        lambda_intra=args.lambda_intra,
        lambda_feat=args.lambda_feat,
        binary_features=True,
        log_every=args.log_every,
    )
    attacker = SCVNIAAttacker(
        clean_victim.model,
        num_classes=num_classes,
        config=cfg,
        forward_fn=dense_forward,
        device=device,
    )
    mod_adj, mod_features, info = attacker.attack(
        adj=adj,
        features=features,
        labels=attack_labels,
        idx_attack=idx_attack,
    )

    print(
        f"[Poison graph] injected_nodes={info['num_active_injected_nodes']}, "
        f"cross_edges={info['num_cross_edges']}, intra_edges={info['num_intra_edges']}, "
        f"N={mod_adj.size(0)}, E_undir={int(mod_adj.sum().item() // 2)}"
    )
    if info.get("history"):
        print("[Attack history]")
        for h in info["history"]:
            print(h)

    # 4. Evasion evaluation: fixed clean victim, attacked graph.
    mod_labels = extend_labels_for_injected_nodes(labels, info["num_active_injected_nodes"])
    mod_graph = dense_adj_to_dgl_graph(mod_adj, mod_features, mod_labels, add_self_loops=True)
    evasion_acc = accuracy_from_victim(clean_victim, mod_graph, mod_features, mod_labels, idx_test)
    print(f"[Evasion] test_acc={evasion_acc:.4f}, acc_drop={clean_acc - evasion_acc:.4f}")

    # 5. Poisoning evaluation: retrain a fresh victim on the poisoned graph.
    set_seed(args.seed)
    poison_victim = make_victim(args, loader_args, mod_features, mod_labels, device)
    poison_result = poison_victim.fit_eval(
        mod_graph,
        mod_features,
        mod_labels,
        idx_train,  # injected nodes are not added to the supervised training set
        idx_val,
        idx_test,
        epochs=args.poison_epochs,
        patience=args.patience,
        verbose=False,
    )
    poison_acc = float(poison_result["test_acc"])

    print("========== Result ==========")
    print(f"dataset={args.dataset}")
    print(f"victim={args.victim}")
    print(f"split_name={args.split_name}")
    print(f"attack_label_mode={args.attack_label_mode}")
    print(f"target_nodes={idx_attack.numel()}")
    print(f"injected_nodes={info['num_active_injected_nodes']}")
    print(f"cross_edges={info['num_cross_edges']}")
    print(f"intra_edges={info['num_intra_edges']}")
    print(f"clean_test_acc={clean_acc:.4f}")
    print(f"evasion_test_acc={evasion_acc:.4f}")
    print(f"poison_test_acc={poison_acc:.4f}")
    print(f"evasion_acc_drop={clean_acc - evasion_acc:.4f}")
    print(f"poison_acc_drop={clean_acc - poison_acc:.4f}")


if __name__ == "__main__":
    main()
