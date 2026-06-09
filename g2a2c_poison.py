# -*- coding: utf-8 -*-
"""G2A2C-style node injection poisoning on top of the CaVNI loader/victim.

The original G2A2C code is an evasion attack. This script keeps its
Node_Generator, Edge_Sampler, and Value_Predictor modules, generates injected
nodes against a clean-trained victim, then retrains a fresh victim on the
poisoned graph to report poisoning accuracy.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
# PROJECT_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = THIS_DIR
sys.path.insert(0, str(THIS_DIR))

from g2a2c_model import Edge_Sampler, Node_Generator, Value_Predictor
from utils import set_seed, load_original_graph
from victim_models import Victim


# def build_common_args(args: argparse.Namespace) -> SimpleNamespace:
#     return SimpleNamespace(
#         dataset=args.dataset,
#         data_root=str(THIS_DIR / "data"),
#         train_ratio=args.train_ratio,
#         val_ratio=args.val_ratio,
#         split_id=args.split_id,
#         seed=args.seed,
#         hidden=args.hidden,
#         num_heads1=args.num_heads1,
#         num_heads2=args.num_heads2,
#         feat_dropout=args.feat_dropout,
#         attn_dropout=args.attn_dropout,
#         sage_agg_type=args.sage_agg_type,
#         appnp_hidden=args.appnp_hidden,
#         K=args.K,
#         alpha=args.alpha,
#         gin_hidden=args.gin_hidden,
#         gin_agg_type=args.gin_agg_type,
#     )
def build_common_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        data_root=str(PROJECT_ROOT / "datasets"),
        split_name="graphdc_aligned",
        model_name=args.victim,
        victim=args.victim,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_id=args.split_id,
        seed=args.seed,
        hidden=args.hidden,
        num_heads1=args.num_heads1,
        num_heads2=args.num_heads2,
        feat_dropout=args.feat_dropout,
        attn_dropout=args.attn_dropout,
        sage_agg_type=args.sage_agg_type,
        appnp_hidden=args.appnp_hidden,
        K=args.K,
        alpha=args.alpha,
        gin_hidden=args.gin_hidden,
        gin_agg_type=args.gin_agg_type,
    )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("G2A2C-style poisoning with CaVNI loader")
    p.add_argument("--dataset", default="cora")
    p.add_argument("--victim", default="gcn",
                   choices=["gcn", "gat", "sage", "appnp", "gin", "sgc", "simpgcn"])
    p.add_argument("--device", default='cuda:0',
                   help="Device to run on, e.g. cuda:0 or cpu. Defaults to cuda:<gpu> when CUDA is available.")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU id used when --device is not set and CUDA is available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean_epochs", type=int, default=80)
    p.add_argument("--poison_epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--attack_epochs", type=int, default=1)
    p.add_argument("--target_limit", type=int, default=30)
    p.add_argument("--node_budget", type=int, default=1)
    p.add_argument("--edge_budget", type=int, default=1)
    p.add_argument("--khop_feat", type=int, default=1)
    p.add_argument("--hid_dim", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr_attack", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--feature_budget", type=float, default=1.0)
    p.add_argument("--discrete_feat", action="store_true", default=True)
    p.add_argument("--poison_reward_mode", default="clean_loss", choices=["clean_loss", "val_acc_drop"])
    p.add_argument("--poison_train_mode", default="structure_only", choices=["structure_only", "labeled_injected"])
    p.add_argument("--reward_epochs", type=int, default=3)
    p.add_argument("--poison_success_drop", type=float, default=0.0)
    p.add_argument("--fake_label_strategy", default="target", choices=["target", "second_best", "random_wrong"])
    p.add_argument("--train_ratio", type=float, default=0.1)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--split_id", type=int, default=0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num_heads1", type=int, default=8)
    p.add_argument("--num_heads2", type=int, default=1)
    p.add_argument("--feat_dropout", type=float, default=0.5)
    p.add_argument("--attn_dropout", type=float, default=0.5)
    p.add_argument("--sage_agg_type", default="mean")
    p.add_argument("--appnp_hidden", type=int, default=64)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gin_hidden", type=int, default=64)
    p.add_argument("--gin_agg_type", default="sum")
    return p.parse_args()


@torch.no_grad()
def accuracy_from_victim(victim: Victim, graph, feats, labels, idx) -> float:
    logits = victim.logits(graph, feats)
    return float((logits[idx].argmax(-1) == labels[idx]).float().mean().item())


@torch.no_grad()
def model_loss_and_success(victim: Victim, graph, node_id: torch.Tensor) -> tuple[float, bool]:
    logits = victim.logits(graph, graph.ndata["feat"])[node_id].reshape(1, -1)
    label = graph.ndata["label"][node_id].reshape(1)
    loss = F.cross_entropy(logits, label).item()
    success = int(logits.argmax(-1).item()) != int(label.item())
    return loss, bool(success)


def reward(victim: Victim, node_id: torch.Tensor, prev_graph, cur_graph) -> tuple[float, bool]:
    cur_loss, success = model_loss_and_success(victim, cur_graph, node_id)
    prev_loss, _ = model_loss_and_success(victim, prev_graph, node_id)
    return cur_loss - prev_loss + (10.0 if success else 0.0), success


@torch.no_grad()
def choose_fake_label(args, victim: Victim, graph, target: torch.Tensor, n_classes: int) -> torch.Tensor:
    true_label = graph.ndata["label"][target]
    if args.fake_label_strategy == "target":
        return true_label.detach()
    if args.fake_label_strategy == "second_best":
        logits = victim.logits(graph, graph.ndata["feat"])[target]
        ranked = torch.argsort(logits, descending=True)
        for cls in ranked:
            if int(cls.item()) != int(true_label.item()):
                return cls.detach()
        return true_label.detach()
    if args.fake_label_strategy == "random_wrong":
        choices = torch.arange(n_classes, device=graph.device)
        choices = choices[choices != true_label]
        return choices[torch.randint(0, choices.numel(), (1,), device=graph.device)[0]].detach()
    raise ValueError(f"Unknown fake_label_strategy: {args.fake_label_strategy}")


def build_poison_train_idx(train_idx: torch.Tensor, graph, base_num_nodes: int, mode: str) -> torch.Tensor:
    if mode == "structure_only" or graph.num_nodes() <= base_num_nodes:
        return train_idx
    injected_idx = torch.arange(base_num_nodes, graph.num_nodes(), dtype=torch.long, device=train_idx.device)
    return torch.cat([train_idx, injected_idx], dim=0)


def warm_start_val_acc(args, common_args, clean_victim: Victim, graph, train_idx, val_idx, base_num_nodes: int) -> float:
    feats = graph.ndata["feat"].detach()
    labels = graph.ndata["label"].detach()
    tr = build_poison_train_idx(train_idx, graph, base_num_nodes, args.poison_train_mode)
    tmp_victim = make_victim(args, common_args, feats, labels, graph.device)
    tmp_victim.model.load_state_dict({k: v.detach().clone() for k, v in clean_victim.model.state_dict().items()})
    tmp_victim.model.train()
    opt = torch.optim.Adam(tmp_victim.model.parameters(), lr=args.lr, weight_decay=args.wd)
    for _ in range(max(1, args.reward_epochs)):
        logits = tmp_victim.model(graph, feats)
        loss = F.cross_entropy(logits[tr], labels[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return accuracy_from_victim(tmp_victim, graph, feats, labels, val_idx)


def poisoning_reward(args, common_args, clean_victim, prev_graph, cur_graph, train_idx, val_idx, base_num_nodes):
    prev_val = warm_start_val_acc(args, common_args, clean_victim, prev_graph, train_idx, val_idx, base_num_nodes)
    cur_val = warm_start_val_acc(args, common_args, clean_victim, cur_graph, train_idx, val_idx, base_num_nodes)
    drop = prev_val - cur_val
    success = drop > args.poison_success_drop
    return drop, success


def inject_node(graph, feat, label):
    new_id = graph.num_nodes()
    graph = dgl.add_nodes(
        graph,
        1,
        {
            "feat": feat.reshape(1, -1),
            "label": label.reshape(1),
        },
    )
    graph = dgl.add_edges(graph, new_id, new_id)
    return graph


def wire_edge(graph, dst):
    new_id = graph.num_nodes() - 1
    src = torch.tensor([int(dst), int(new_id)], device=graph.device)
    dst_t = torch.tensor([int(new_id), int(dst)], device=graph.device)
    return dgl.add_edges(graph, src, dst_t)


def append_poison_solution(base_graph, solutions: list[dict]):
    graph = base_graph
    for sol in solutions:
        graph = inject_node(graph, sol["feat"], sol["label"])
        for dst in sol["edges"]:
            graph = wire_edge(graph, dst)
    graph.ndata["feat"] = graph.ndata["feat"].float()
    graph.ndata["label"] = graph.ndata["label"].long()
    return graph


def make_victim(args, common_args, feats, labels, device):
    return Victim(
        model_name=args.victim,
        in_dim=feats.shape[1],
        n_classes=int(labels.max().item() + 1),
        lr=args.lr,
        wd=args.wd,
        args=common_args,
        device=device,
    )


def train_attack_policy(args, common_args, clean_victim, graph, train_idx, val_idx, test_idx):
    feat_dim = graph.ndata["feat"].shape[1]
    n_classes = int(graph.ndata["label"].max().item() + 1)
    device = graph.device
    base_num_nodes = graph.num_nodes()

    node_generator = Node_Generator(feat_dim, args.hid_dim * 2, args.discrete_feat).to(device)
    edge_sampler = Edge_Sampler(feat_dim, args.hid_dim * 2).to(device)
    value_predictor = Value_Predictor(feat_dim, args.hid_dim * 2, n_classes).to(device)
    optimizer = torch.optim.Adam(
        list(node_generator.parameters())
        + list(edge_sampler.parameters())
        + list(value_predictor.parameters()),
        lr=args.lr_attack,
    )

    feature_budget = (graph.ndata["feat"] > 0).float().sum(1).mean()
    eps = np.finfo(np.float32).eps.item()
    targets = test_idx[: min(args.target_limit, test_idx.numel())]
    best_solutions: list[dict] = []

    for epoch in range(args.attack_epochs):
        epoch_solutions: list[dict] = []
        successes = 0
        order = targets[torch.randperm(targets.numel(), device=targets.device)]
        pbar = tqdm(order.detach().cpu().tolist(), desc=f"[Attack epoch {epoch + 1}]")

        for target_int in pbar:
            target = torch.tensor(int(target_int), dtype=torch.long, device=device)
            local_graph = copy.deepcopy(graph)
            action_buffer = []
            reward_buffer = []
            feature_loss = torch.tensor(0.0, device=device)
            node_solutions = []

            success = False
            for _ in range(args.node_budget):
                subgraph, local_target = dgl.khop_in_subgraph(local_graph, target, args.khop_feat)
                feat, num_feat, feat_log_prob = node_generator(subgraph, local_target.item())
                if args.discrete_feat:
                    target_density = torch.floor(feature_budget * args.feature_budget) / feat_dim
                    feature_loss = feature_loss + F.mse_loss(num_feat, target_density) / args.node_budget

                edges = []
                edge_set = []
                fake_label = choose_fake_label(args, clean_victim, local_graph, target, n_classes)
                for edge_step in range(args.edge_budget):
                    if edge_step == 0:
                        prev_graph = local_graph.clone()
                        local_graph = inject_node(local_graph, feat, fake_label)
                        local_graph = wire_edge(local_graph, target)
                        edges.append(int(target.item()))
                        edge_set = [target]
                        if args.poison_reward_mode == "clean_loss":
                            r, success = reward(clean_victim, target, prev_graph, local_graph)
                        else:
                            r, success = poisoning_reward(
                                args, common_args, clean_victim, prev_graph, local_graph,
                                train_idx, val_idx, base_num_nodes
                            )
                        value = value_predictor(local_graph, target, local_graph.ndata["label"][target])
                        action_buffer.append((feat_log_prob, value))
                        reward_buffer.append(r)
                    else:
                        edge_dist, edge_log_prob = edge_sampler(local_graph, target, edge_set)
                        value = value_predictor(local_graph, target, local_graph.ndata["label"][target])
                        dst = (edge_dist == 1).nonzero(as_tuple=True)[0][0]
                        prev_graph = local_graph.clone()
                        local_graph = wire_edge(local_graph, dst)
                        if args.poison_reward_mode == "clean_loss":
                            r, success = reward(clean_victim, target, prev_graph, local_graph)
                        else:
                            r, success = poisoning_reward(
                                args, common_args, clean_victim, prev_graph, local_graph,
                                train_idx, val_idx, base_num_nodes
                            )
                        action_buffer.append((edge_log_prob + feat_log_prob, value))
                        reward_buffer.append(r)
                        edge_set.append(dst)
                        edges.append(int(dst.item()))
                    if success:
                        break

                node_solutions.append(
                    {
                        "target": int(target.item()),
                        "feat": feat.detach(),
                        "label": fake_label.detach(),
                        "edges": edges,
                    }
                )
                if success:
                    break

            if action_buffer:
                returns = []
                ret = 0.0
                for r in reward_buffer[::-1]:
                    ret = float(r) + args.gamma * ret
                    returns.insert(0, ret)
                returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
                if returns_t.numel() > 1:
                    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + eps)

                policy_losses = []
                value_losses = []
                for (log_prob, value), ret_t in zip(action_buffer, returns_t):
                    advantage = ret_t - value.detach()
                    policy_losses.append(-log_prob * advantage)
                    value_losses.append(F.smooth_l1_loss(value, ret_t))
                loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum() + feature_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if success:
                successes += 1
            # For poisoning, keep the generated injection even if it does not
            # immediately flip the fixed clean victim. The later evaluation is
            # retraining on the poisoned graph, not evasion-only success.
            epoch_solutions.extend(node_solutions)
            pbar.set_postfix(success=f"{successes}/{targets.numel()}")

        best_solutions = epoch_solutions

    return best_solutions, int(targets.numel())


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    set_seed(args.seed)
    if args.device is None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        requested = torch.device(args.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({args.device}) but CUDA is not available.")
        device = requested
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"[Device] using {device}")
    common_args = build_common_args(args)

    graph, feats, labels, train_idx, val_idx, test_idx = load_original_graph(args.dataset, device, common_args)
    print(
        f"[Data] {args.dataset}: N={graph.num_nodes()}, E={graph.num_edges()}, "
        f"F={feats.shape[1]}, C={int(labels.max().item() + 1)}, "
        f"train={train_idx.numel()}, val={val_idx.numel()}, test={test_idx.numel()}"
    )

    set_seed(args.seed)
    clean_victim = make_victim(args, common_args, feats, labels, device)
    clean_result = clean_victim.fit_eval(
        graph, feats, labels, train_idx, val_idx, test_idx,
        epochs=args.clean_epochs, patience=args.patience, verbose=False
    )
    clean_acc = float(clean_result["test_acc"])
    print(f"[Clean] test_acc={clean_acc:.4f}")

    correctly_pred = test_idx[
        clean_victim.predict(graph, feats, test_idx) == labels[test_idx]
    ]
    if correctly_pred.numel() == 0:
        raise RuntimeError("No correctly predicted test nodes available for G2A2C targets.")

    solutions, target_count = train_attack_policy(
        args, common_args, clean_victim, graph, train_idx, val_idx, correctly_pred
    )
    poisoned_graph = append_poison_solution(graph, solutions).to(device)
    poisoned_feats = poisoned_graph.ndata["feat"]
    poisoned_labels = poisoned_graph.ndata["label"]
    poison_train_idx = build_poison_train_idx(
        train_idx, poisoned_graph, graph.num_nodes(), args.poison_train_mode
    )

    print(
        f"[Poison graph] injected_nodes={len(solutions)}, "
        f"N={poisoned_graph.num_nodes()}, E={poisoned_graph.num_edges()}, "
        f"poison_train={poison_train_idx.numel()}"
    )

    set_seed(args.seed)
    poison_victim = make_victim(args, common_args, poisoned_feats, poisoned_labels, device)
    poison_result = poison_victim.fit_eval(
        poisoned_graph,
        poisoned_feats,
        poisoned_labels,
        poison_train_idx,
        val_idx,
        test_idx,
        epochs=args.poison_epochs,
        patience=args.patience,
        verbose=False,
    )
    poison_acc = float(poison_result["test_acc"])
    print("========== Result ==========")
    print(f"dataset={args.dataset}")
    print(f"victim={args.victim}")
    print(f"targets_used={target_count}")
    print(f"injected_nodes={len(solutions)}")
    print(f"poison_reward_mode={args.poison_reward_mode}")
    print(f"poison_train_mode={args.poison_train_mode}")
    print(f"fake_label_strategy={args.fake_label_strategy}")
    print(f"clean_test_acc={clean_acc:.4f}")
    print(f"poison_test_acc={poison_acc:.4f}")
    print(f"acc_drop={clean_acc - poison_acc:.4f}")


if __name__ == "__main__":
    main()
