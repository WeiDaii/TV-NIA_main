# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch as th
import dgl
import torch
import scipy.sparse as sp
import numpy as np
import copy
from sklearn.model_selection import train_test_split
import os, glob
import sys
import pickle as pkl
import networkx as nx

def batch_subgraphs_from_injid_to_cands(G: dgl.DGLGraph, injid_to_cands: dict):
    """
    对每个 inj_id 构造一个列表 [inj_id] + cands 并抽子图。
    返回 dict: {inj_id: (subg, nodes_t, nid_map)}
    """
    out = {}
    for inj_id, cands in injid_to_cands.items():
        subg, nodes_t, nid_map = subgraph_from_inj_and_cands(G, inj_id, cands)
        out[int(inj_id)] = (subg, nodes_t, nid_map)
    return out

def subgraph_from_inj_and_cands(G: dgl.DGLGraph, inj_id, cands):
    """
    把 inj_id 和 cands 合到一个列表，一次性从全图 G 抽子图（有向/无向均可）。
    返回：subg（紧凑编号）、nodes_t（列表对应的全局ID）、nid_map（subg节点->全局ID）
    """
    device = G.device if hasattr(G, "device") else torch.device("cpu")
    N = G.num_nodes()

    # 1) 合并成一个列表（去重、过滤非法、保持 inj_id 在最前）
    if isinstance(cands, torch.Tensor):
        cands = cands.detach().cpu().tolist()
    cands = [int(v) for v in cands if 0 <= int(v) < N and int(v) != int(inj_id)]

    # 去重但保序
    seen, c_uniq = set(), []
    for v in cands:
        if v not in seen:
            seen.add(v); c_uniq.append(v)

    node_list = [int(inj_id)] + c_uniq
    nodes_t = torch.tensor(node_list, dtype=torch.long, device=device)

    # 2) 用这个【一个列表】从全图索引子图
    subg = dgl.node_subgraph(G, nodes_t)

    # 3) 子图 -> 全局ID 映射（紧凑编号还原到原图编号）
    nid_map = subg.ndata[dgl.NID].to(device).long()
    return subg, nodes_t, nid_map


def merge_anchor_maps(anchor_to_injid: dict, candidate: dict, ensure_anchor_in_cands: bool = True):
    """
    anchor_to_injid: {anchor -> injected_id}
    candidate:       {anchor -> [c1, c2, ...]}
    ensure_anchor_in_cands: 若 True，确保候选列表里包含 anchor 本人（若无则放到最前）

    返回：
      merged_by_anchor: {anchor: {"inj_id": inj_id or None, "cands": [..]}}
      injid_to_cands:   {inj_id: [..]}
      edge_pairs:       [(inj_id, v), ...]  # 仅对有 inj_id 的 anchor 生成
    """
    # 所有出现过的锚点
    all_anchors = set(anchor_to_injid.keys()) | set(candidate.keys())

    merged_by_anchor = {}
    injid_to_cands = {}
    edge_pairs = []

    for a in sorted(all_anchors):
        inj = anchor_to_injid.get(a, None)
        cands_raw = candidate.get(a, [])

        # 去重但保序
        seen = set()
        cands = []
        for v in cands_raw:
            if v not in seen:
                seen.add(v)
                cands.append(v)

        # 可选：确保候选里包含 anchor 自身
        if ensure_anchor_in_cands and a not in cands:
            cands = [a] + cands

        merged_by_anchor[a] = {"inj_id": inj, "cands": cands}

        if inj is not None:
            injid_to_cands[inj] = cands
            edge_pairs.extend((inj, v) for v in cands)

    return merged_by_anchor, injid_to_cands, edge_pairs

@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(-1)==y).float().mean().item()
# c2sni/utils.py

import random
from typing import Any, Optional
import numpy as np
import torch
import dgl

__all__ = [
    "set_seed",
    "device_of",
    "to_cpu",
    "to_device",
    "clone_graph_with_data",
]

import dgl
import torch
import dgl
import torch
import numpy as np

def _to_long_1d(x, device, N=None):
    """把多种节点集合表示统一为 1D LongTensor（全局节点ID）。"""
    if isinstance(x, torch.Tensor):
        t = x.to(device)
        if t.dtype == torch.bool:
            idx = torch.nonzero(t, as_tuple=True)[0].long()
            return idx
        return t.long().view(-1)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(x, dtype=torch.long, device=device).view(-1)
    if isinstance(x, np.ndarray):
        if x.dtype == bool:
            idx = np.nonzero(x)[0].astype(np.int64)
            return torch.from_numpy(idx).to(device)
        return torch.from_numpy(x.astype(np.int64)).to(device).view(-1)
    # 未识别则返回空
    return torch.empty(0, dtype=torch.long, device=device)

def _normalize_candidate(candidate, anchors_1d: torch.Tensor, device, N):
    """
    规范化为 dict[anchor_id] -> 1D LongTensor(节点ID列表)
    支持:
      - dict: {anchor: list/tensor/mask 或 {chosen/pool/...}}
      - list/tuple: 与 anchors 对齐或通用一份
      - torch.Tensor:
           2D: (num_anchors, K)
           1D: (K,)  —— 所有 anchors 共用同一份
           bool mask: 同上规则
    """
    mapping = {}
    anchors_list = anchors_1d.tolist()

    # dict
    if isinstance(candidate, dict):
        for a in anchors_list:
            item = candidate.get(int(a), [])
            if isinstance(item, dict):
                for key in ['chosen', 'nodes', 'pool', 'cands', 'neighbors']:
                    if key in item:
                        item = item[key]; break
            mapping[int(a)] = _to_long_1d(item, device, N)
        return mapping

    # Tensor
    if isinstance(candidate, torch.Tensor):
        cand = candidate.to(device)
        if cand.dim() == 2:
            assert cand.size(0) == len(anchors_list), \
                f"candidate第一维({cand.size(0)})应与anchors长度({len(anchors_list)})一致"
            for i, a in enumerate(anchors_list):
                row = cand[i]
                if row.dtype == torch.bool:
                    idx = torch.nonzero(row, as_tuple=True)[0].long()
                else:
                    idx = row.long().view(-1)
                mapping[int(a)] = idx
            return mapping
        elif cand.dim() == 1:
            if cand.dtype == torch.bool:
                idx = torch.nonzero(cand, as_tuple=True)[0].long()
            else:
                idx = cand.long().view(-1)
            for a in anchors_list:
                mapping[int(a)] = idx
            return mapping

    # list/tuple
    if isinstance(candidate, (list, tuple)):
        if len(candidate) == len(anchors_list) and len(candidate) > 0 and isinstance(candidate[0], (list, tuple, torch.Tensor, np.ndarray)):
            for i, a in enumerate(anchors_list):
                mapping[int(a)] = _to_long_1d(candidate[i], device, N)
            return mapping
        else:
            idx = _to_long_1d(candidate, device, N)
            for a in anchors_list:
                mapping[int(a)] = idx
            return mapping

    # 兜底：全空
    for a in anchors_list:
        mapping[int(a)] = torch.empty(0, dtype=torch.long, device=device)
    return mapping

def extract_anchor_subgraphs_from_G(
    G: dgl.DGLGraph,
    anchors,
    candidate,
    anchor_to_injid: dict = None,
    include_anchor: bool = True,
    include_injected: bool = False,  # 若要把注入节点也放进子图，置 True 并传入 anchor_to_injid
):
    """
    返回：
      subgraphs: dict[anchor] -> DGL 子图（紧凑编号）
      node_maps: dict[anchor] -> subg.ndata[dgl.NID]（子图节点 -> 原图节点ID）
      used_nodes: dict[anchor] -> 该 anchor 的全局节点ID集合（1D LongTensor）
    """
    device = G.device if hasattr(G, "device") else torch.device("cpu")
    N = G.num_nodes()

    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(list(anchors), dtype=torch.long, device=device)
    else:
        anchors = anchors.to(device).long()

    # 统一 candidate 结构
    cand_map = _normalize_candidate(candidate, anchors, device, N)

    subgraphs, node_maps, used_nodes = {}, {}, {}
    for a in anchors.tolist():
        nodes = cand_map.get(int(a), torch.empty(0, dtype=torch.long, device=device))

        # 追加 anchor / injected（可选）
        extra = []
        if include_anchor:
            extra.append(int(a))
        if include_injected and (anchor_to_injid is not None) and (int(a) in anchor_to_injid):
            extra.append(int(anchor_to_injid[int(a)]))

        if len(extra) > 0:
            extra_t = torch.tensor(extra, dtype=torch.long, device=device)
            nodes = torch.cat([nodes, extra_t], dim=0)

        # 清理非法/重复
        if nodes.numel() > 0:
            nodes = nodes[(nodes >= 0) & (nodes < N)]
            nodes = torch.unique(nodes)

        if nodes.numel() == 0:
            # 至少包含 anchor 本人，避免空子图
            nodes = torch.tensor([int(a)], dtype=torch.long, device=device)

        # 诱导子图 + 原ID映射
        subg = dgl.node_subgraph(G, nodes)
        nid_map = subg.ndata[dgl.NID].to(device).long()

        subgraphs[int(a)] = subg
        node_maps[int(a)] = nid_map
        used_nodes[int(a)] = nodes

    return subgraphs, node_maps, used_nodes


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(labels, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,labels.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    train_mask = index_to_mask(train_idx,size=len(labels))
    val_mask = index_to_mask(val_idx,size=len(labels))
    test_mask = index_to_mask(test_idx,size=len(labels))
    return train_mask, val_mask, test_mask


GRAPHDC_CLEAN_TARGETS = {
    "cora": {"gcn": 0.8116, "sage": 0.8256},
    "citeseer": {"gcn": 0.7091, "sage": 0.7185},
    "pubmed": {"gcn": 0.7917, "sage": 0.7921},
    "ogbn_arxiv": {"gcn": 0.7360, "sage": 0.7295},
    "10k_ogbproducts": {"gcn": 0.7583, "sage": 0.7603},
    "12k_reddit": {"gcn": 0.9162, "sage": 0.9187},
}


GRAPHDC_ALIGNED_SPLIT_CONFIG = {
    # This config is intentionally explicit. It is only used when callers pass
    # split_name="graphdc_aligned"; existing loaders keep their current default.
    "cora": {
        "split_name": "public",
        "seed": 42,
        "model_splits": {
            "gcn": {"split_name": "public", "seed": 42},
            "sage": {"split_name": "random_10_10_80", "seed": 42},
        },
    },
    "citeseer": {
        "split_name": "public",
        "seed": 42,
        "model_splits": {
            "gcn": {"split_name": "public", "seed": 42},
            "sage": {"split_name": "random_10_10_80", "seed": 0},
        },
    },
    "pubmed": {
        "split_name": "public",
        "seed": 42,
        "model_splits": {
            "gcn": {"split_name": "public", "seed": 42},
            "sage": {"split_name": "public", "seed": 42},
        },
    },
    "10k_ogbproducts": {
        "split_name": "random_60_20_20",
        "seed": 1,
        "model_splits": {
            "gcn": {"split_name": "random_60_20_20", "seed": 1},
            "sage": {"split_name": "random_60_20_20", "seed": 123},
        },
    },
    "12k_reddit": {"split_name": "builtin", "seed": 42},
}


def _indices_to_masks(idx_train, idx_val, idx_test, size, device=None):
    train_mask = index_to_mask(torch.as_tensor(idx_train, dtype=torch.long), size)
    val_mask = index_to_mask(torch.as_tensor(idx_val, dtype=torch.long), size)
    test_mask = index_to_mask(torch.as_tensor(idx_test, dtype=torch.long), size)
    if device is not None:
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
    return train_mask, val_mask, test_mask


def _stratified_ratio_split(labels, train_ratio, val_ratio, seed):
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    all_idx = np.arange(labels_np.shape[0])
    values, counts = np.unique(labels_np, return_counts=True)
    stratify_first = labels_np if counts.min() >= 2 else None
    idx_train, idx_temp, _, y_temp = train_test_split(
        all_idx,
        labels_np,
        train_size=float(train_ratio),
        stratify=stratify_first,
        random_state=int(seed),
    )
    relative_val_size = float(val_ratio) / (1.0 - float(train_ratio))
    if stratify_first is None:
        stratify_second = None
    else:
        _, temp_counts = np.unique(y_temp, return_counts=True)
        stratify_second = y_temp if temp_counts.min() >= 2 else None
    idx_val, idx_test = train_test_split(
        idx_temp,
        train_size=relative_val_size,
        stratify=stratify_second,
        random_state=int(seed),
    )
    return torch.LongTensor(idx_train), torch.LongTensor(idx_val), torch.LongTensor(idx_test)


def _per_class_fixed_split(labels, per_class_train=20, val_size=500, test_size=1000, seed=42):
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    rng = np.random.RandomState(int(seed))
    train_idx = []
    all_idx = np.arange(labels_np.shape[0])
    for c in sorted(np.unique(labels_np).tolist()):
        cls_idx = np.where(labels_np == c)[0]
        take = min(int(per_class_train), cls_idx.shape[0])
        train_idx.extend(rng.choice(cls_idx, take, replace=False).tolist())
    train_set = set(train_idx)
    rest_idx = np.array([i for i in all_idx.tolist() if i not in train_set], dtype=np.int64)
    rest_labels = labels_np[rest_idx]
    val_take = min(int(val_size), rest_idx.shape[0])
    idx_val_rel, idx_test_rel = train_test_split(
        np.arange(rest_idx.shape[0]),
        train_size=val_take,
        stratify=rest_labels,
        random_state=int(seed),
    )
    idx_val = rest_idx[idx_val_rel]
    idx_test_pool = rest_idx[idx_test_rel]
    if test_size is not None and idx_test_pool.shape[0] > int(test_size):
        test_rel = rng.choice(np.arange(idx_test_pool.shape[0]), int(test_size), replace=False)
        idx_test_pool = idx_test_pool[test_rel]
    return torch.LongTensor(train_idx), torch.LongTensor(idx_val), torch.LongTensor(idx_test_pool)


def _planetoid_public_split(dataset_name, labels, data_root=None):
    name = str(dataset_name).lower()
    if name not in {"cora", "citeseer", "pubmed"}:
        raise ValueError(f"Planetoid public split is only available for citation datasets, got {dataset_name}")
    root = data_root or file_dir_citation
    with open("{}/ind.{}.y".format(root, name), "rb") as f:
        y = pkl.load(f, encoding="latin1") if sys.version_info > (3, 0) else pkl.load(f)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(root, name))
    test_idx_range = np.sort(test_idx_reorder)
    idx_train = torch.LongTensor(range(y.shape[0]))
    idx_val = torch.LongTensor(range(y.shape[0], y.shape[0] + 500))
    idx_test = torch.LongTensor(test_idx_range)
    return idx_train, idx_val, idx_test


def get_graphdc_aligned_split(
    data,
    dataset_name,
    split_name="graphdc_aligned",
    seed=42,
    data_root=None,
    return_mask=False,
    model_name=None,
):
    """
    Return reproducible train/val/test indices for GraphDC clean-accuracy checks.

    The function only creates split indices or masks. It never changes the graph,
    node features, labels, or node labels. Existing loaders are unaffected unless
    callers explicitly pass split_name="graphdc_aligned" or another split name.
    """
    original_splits = None
    if isinstance(data, torch.Tensor):
        labels = data
    elif hasattr(data, "ndata") and "label" in data.ndata:
        labels = data.ndata["label"]
    elif isinstance(data, dict) and "label" in data:
        labels = data["label"]
        if all(k in data for k in ("idx_train", "idx_val", "idx_test")):
            original_splits = (data["idx_train"], data["idx_val"], data["idx_test"])
    else:
        raise TypeError("data must be labels tensor, DGL graph with ndata['label'], or dict with key 'label'")

    name = str(dataset_name).lower().replace("-", "_")
    selected_split = split_name
    selected_seed = int(seed)
    if split_name == "graphdc_aligned":
        cfg = GRAPHDC_ALIGNED_SPLIT_CONFIG.get(name, {})
        if model_name is not None and isinstance(cfg.get("model_splits"), dict):
            model_cfg = cfg["model_splits"].get(str(model_name).lower(), {})
        else:
            model_cfg = {}
        selected_split = model_cfg.get(
            "split_name",
            cfg.get("split_name", "public" if name in {"cora", "citeseer", "pubmed"} else "builtin"),
        )
        selected_seed = int(model_cfg.get("seed", cfg.get("seed", seed)))

    if selected_split == "public":
        idx_train, idx_val, idx_test = _planetoid_public_split(name, labels, data_root=data_root)
    elif selected_split == "builtin":
        if original_splits is None:
            raise ValueError("builtin split requires data dict with idx_train, idx_val, and idx_test")
        idx_train, idx_val, idx_test = original_splits
    elif selected_split in {"per_class_20", "fixed_20_500_1000"}:
        idx_train, idx_val, idx_test = _per_class_fixed_split(labels, 20, 500, 1000, selected_seed)
    elif selected_split in {"random_10_10_80", "ratio_10_10_80"}:
        idx_train, idx_val, idx_test = _stratified_ratio_split(labels, 0.1, 0.1, selected_seed)
    elif selected_split in {"random_20_10_70", "ratio_20_10_70"}:
        idx_train, idx_val, idx_test = _stratified_ratio_split(labels, 0.2, 0.1, selected_seed)
    elif selected_split in {"random_60_20_20", "ratio_60_20_20"}:
        idx_train, idx_val, idx_test = _stratified_ratio_split(labels, 0.6, 0.2, selected_seed)
    else:
        raise ValueError(f"Unsupported explicit split for get_graphdc_aligned_split: {selected_split}")

    device = labels.device if isinstance(labels, torch.Tensor) else None
    idx_train = idx_train.to(device) if device is not None else idx_train
    idx_val = idx_val.to(device) if device is not None else idx_val
    idx_test = idx_test.to(device) if device is not None else idx_test
    if return_mask:
        return _indices_to_masks(idx_train, idx_val, idx_test, len(labels), device=device)
    return idx_train, idx_val, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features



file_dir_citation = os.getcwd() + '/data'
def load_data_citation(dataset_str='cora'):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/ind.{}.test.index".format(file_dir_citation, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # G = nx.from_dict_of_lists(graph, create_using=nx.DiGraph())
    # adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    # #原来写法
    # D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1 = sp.diags(D1[:, 0], format='csr')
    # D2 = sp.diags(D2[0, :], format='csr')
    #
    # A = adj.dot(D1)
    # A = D2.dot(A)

    #新写法

    deg_row = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_col = np.asarray(adj.sum(axis=0)).reshape(-1)
    with np.errstate(divide='ignore'):
        D1_vec = np.power(deg_row, -0.5)
        D2_vec = np.power(deg_col, -0.5)
    D1_vec[~np.isfinite(D1_vec)] = 0.0
    D2_vec[~np.isfinite(D2_vec)] = 0.0
    D1 = sp.diags(D1_vec, format='csr')
    D2 = sp.diags(D2_vec, format='csr')
    A = D2.dot(adj.dot(D1))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    features = torch.FloatTensor(np.array(features.todense()))
    # features_norm = torch.FloatTensor(np.array(features_norm.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)
    label_classes = labels.numpy()

    # 训练/验证/测试集划分比例
    train_size = 0.1
    val_size = 0.1
    test_size = 0.8

    # 全部节点索引
    all_idx = np.arange(len(label_classes))

    # 先划分训练集
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        all_idx, label_classes, train_size=train_size, stratify=label_classes, random_state=42)

    # 再从剩下的中划分验证与测试
    relative_val_size = val_size / (val_size + test_size)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, train_size=relative_val_size, stratify=y_temp, random_state=42)

    # 转换成 torch Tensor
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print(idx_test)
    print(idx_train)
    print(idx_val)
    return A, features, labels, idx_train, idx_val, idx_test, adj
def _pick_geom_split(root_dir: str, dataset: str, train_ratio=0.6, val_ratio=0.2, split_id=0):
    """
    root_dir: 形如 ".../chameleon_03d661d3" 或 ".../squirrel_xxx" 的目录
    返回：匹配到的 .npz 路径
    """
    want = f"{dataset}_split_{train_ratio}_{val_ratio}_{split_id}.npz"
    cand = glob.glob(os.path.join(root_dir, f"{dataset}_split_*_*_{split_id}.npz"))
    if not cand:
        raise FileNotFoundError(f"No split found under {root_dir}")
    for p in cand:
        if os.path.basename(p) == want:
            return p
    return cand[0]  # 兜底：取第一个

def _load_geom_graph_from_txt(root_dir: str, dataset: str):
    """
    从 out1_graph_edges.txt / out1_node_feature_label.txt 构图 + 读特征/标签。
    返回：adj(Scipy CSR), feats(torch.FloatTensor[N,d]), labs(torch.LongTensor[N])
    节点顺序按 id 升序，保证与 .npz 掩码对齐。
    """
    edges_fp = os.path.join(root_dir, "out1_graph_edges.txt")
    nfeat_fp = os.path.join(root_dir, "out1_node_feature_label.txt")

    G = nx.Graph()  # Geom-GCN 版本是无向图
    with open(nfeat_fp) as f:
        _ = f.readline()  # 跳过表头
        for line in f:
            nid_str, feat_str, y_str = line.rstrip().split('\t')
            nid = int(nid_str)
            feat = np.array(feat_str.split(','), dtype=np.uint8)  # 0/1
            y = int(y_str)
            G.add_node(nid, features=feat, label=y)

    with open(edges_fp) as f:
        _ = f.readline()  # 跳过表头
        for line in f:
            u, v = map(int, line.rstrip().split('\t'))
            G.add_edge(u, v)

    nodes_sorted = sorted(G.nodes())
    adj = nx.adjacency_matrix(G, nodelist=nodes_sorted)
    X = np.array([feat for _, feat in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    y = np.array([lbl for _, lbl in sorted(G.nodes(data='label'),    key=lambda x: x[0])])
    import torch as th
    feats = th.as_tensor(X, dtype=th.float32)
    labs  = th.as_tensor(y, dtype=th.long)
    return adj, feats, labs
import os
import numpy as np
import scipy.sparse as sp
import torch
import dgl
from scipy.sparse.csgraph import connected_components

# ---------- helpers (与“别人”的实现等价/兼容) ----------
def _np_load_any(path, **kwargs):
    """兼容别人代码里的 np.aload；没有就用 np.load(allow_pickle=True)"""
    if hasattr(np, 'aload'):
        return np.aload(path)
    return np.load(path, allow_pickle=True, **kwargs)

def load_npz(file_path):
    """加载 12k_reddit.npz（GraphSAINT/NRG 常用打包方式）"""
    if not file_path.endswith('.npz'):
        file_path += '.npz'
    with _np_load_any(file_path) as loader:
        loader = dict(loader)
        adj = sp.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape']
        )
        if 'attr_data' in loader:
            feats = sp.csr_matrix(
                (loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                shape=loader['attr_shape']
            )
        else:
            feats = None
        labels = loader.get('labels')
    return adj, feats, labels

def largest_connected_components(adj, n_components=1):
    _, comp = connected_components(adj)
    sizes = np.bincount(comp)
    keep = np.argsort(sizes)[::-1][:n_components]
    nodes = [i for i, c in enumerate(comp) if c in keep]
    print(f"Selecting {n_components} largest connected components")
    return nodes

def sparse_mx_to_torch_sparse_tensor(mx):
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values  = torch.from_numpy(mx.data)
    shape   = torch.Size(mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_tensor(sp_adj_tensor: torch.Tensor) -> torch.Tensor:
    """归一化 Ā = D^{-1/2} A D^{-1/2}，输入 Torch 稀疏张量"""
    sp_adj_tensor = sp_adj_tensor.coalesce()
    idx = sp_adj_tensor.indices()
    val = sp_adj_tensor.values()
    row, col = idx[0], idx[1]
    deg = torch.sparse.sum(sp_adj_tensor, dim=1).to_dense().flatten()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]
    return torch.sparse.FloatTensor(idx, norm_val, sp_adj_tensor.shape)

def _split_to_indices(split_dict):
    """把 _split.npy 里的 mask 或 index 转成 LongTensor 索引"""
    def _to_index(x):
        x = np.asarray(x)
        if x.dtype == bool or (x.dtype == np.bool_):
            return torch.LongTensor(np.where(x)[0])
        # 假设已是索引
        return torch.LongTensor(x)
    return _to_index(split_dict['train']), _to_index(split_dict['val']), _to_index(split_dict['test'])


# ---------- Reddit-12k 加载 ----------
def load_reddit12k_from_npz(data_root='data', dataset_name='12k_reddit', connect=False):
    """
    读取 12k_reddit.npz & 12k_reddit_split.npy，并返回：
    A_torch(归一化稀疏邻接), features_t, labels_t, idx_train, idx_val, idx_test, adj_sp(scipy,含自环)
    """
    npz_path   = os.path.join(data_root, f'{dataset_name}.npz')
    split_path = os.path.join(data_root, f'{dataset_name}_split.npy')  # 兼容别人写法

    # 1) 读 npz
    adj, features, labels_np = load_npz(npz_path)
    n = adj.shape[0]

    # 2) 按“别人”的处理方式：对称化 + 加自环 + 去多重边
    #    注：这一行相当于把 A 对称化并补 I，自环权重=1
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(n, format='csr')
    # 把边权截断到 {0,1}
    adj.data[:] = 1.0

    # 3) 可选：保留最大连通子图（如果想与别人一致，可把 connect=True）
    if connect:
        keep_nodes = largest_connected_components(adj, n_components=1)
        adj = adj[keep_nodes][:, keep_nodes].tocsr()
        features = features[keep_nodes]
        labels_np = labels_np[keep_nodes]
        n = adj.shape[0]
        print('Nodes num:', n)

    # 4) 转 Torch
    adj_torch = sparse_mx_to_torch_sparse_tensor(adj)
    nor_adj_torch = normalize_tensor(adj_torch)  # D^{-1/2} A D^{-1/2}

    # feats（稀疏->稠密->float32）
    feat = torch.from_numpy(features.todense().astype(np.float32))

    labels_t = torch.LongTensor(labels_np)

    # 5) 读 split（支持 .npy 是 dict{'train','val','test'}）
    split = _np_load_any(split_path).item()
    idx_train, idx_val, idx_test = _split_to_indices(split)

    # 6) 打印统计
    #    注意：|E| 这里统计的是无向边数（去自环）
    adj_wo_loop = adj.copy()
    adj_wo_loop.setdiag(0); adj_wo_loop.eliminate_zeros()
    E_undirected = adj_wo_loop.nnz // 2
    print(f"[reddit12k] N={adj.shape[0]}, |E|={E_undirected}, d_avg={2 * E_undirected / adj.shape[0]:.3f}")

    return nor_adj_torch, feat, labels_t, idx_train, idx_val, idx_test, adj

def load_data_citation_public(dataset_str='pubmed'):
    """
    Planetoid public split loader.

    For PubMed:
        train = 60
        val   = 500
        test  = 1000

    This setting usually gives PubMed clean accuracy around 79%.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(file_dir_citation, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(
        "{}/ind.{}.test.index".format(file_dir_citation, dataset_str)
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)

        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    deg_row = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg_col = np.asarray(adj.sum(axis=0)).reshape(-1)

    with np.errstate(divide='ignore'):
        D1_vec = np.power(deg_row, -0.5)
        D2_vec = np.power(deg_col, -0.5)

    D1_vec[~np.isfinite(D1_vec)] = 0.0
    D2_vec[~np.isfinite(D2_vec)] = 0.0

    D1 = sp.diags(D1_vec, format='csr')
    D2 = sp.diags(D2_vec, format='csr')
    A = D2.dot(adj.dot(D1))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(A)

    # Planetoid public split
    idx_train = torch.LongTensor(range(y.shape[0]))
    idx_val = torch.LongTensor(range(y.shape[0], y.shape[0] + 500))
    idx_test = torch.LongTensor(test_idx_range)

    print(
        f"[Planetoid-public] {dataset_str} | "
        f"N={labels.numel()}, F={features.size(1)}, C={int(labels.max().item() + 1)} | "
        f"train={idx_train.numel()}, val={idx_val.numel()}, test={idx_test.numel()}"
    )

    return A, features, labels, idx_train, idx_val, idx_test, adj


def _split_to_indices_flexible(split_obj):
    """
    兼容 split.npy 的多种保存格式：
    1. dict: {'train': ..., 'val': ..., 'test': ...}
    2. dict: {'idx_train': ..., 'idx_val': ..., 'idx_test': ...}
    3. list/tuple: [train, val, test]
    """
    if isinstance(split_obj, np.ndarray) and split_obj.shape == ():
        split_obj = split_obj.item()

    if isinstance(split_obj, np.ndarray) and split_obj.dtype == object:
        if split_obj.shape == ():
            split_obj = split_obj.item()
        else:
            split_obj = split_obj.tolist()

    def _to_index(x):
        x = np.asarray(x)
        if x.dtype == bool or x.dtype == np.bool_:
            return torch.LongTensor(np.where(x)[0])
        return torch.LongTensor(x.astype(np.int64))

    if isinstance(split_obj, dict):
        train = split_obj.get('train', split_obj.get('idx_train', split_obj.get('train_idx', None)))
        val = split_obj.get('val', split_obj.get('valid', split_obj.get('idx_val', split_obj.get('val_idx', None))))
        test = split_obj.get('test', split_obj.get('idx_test', split_obj.get('test_idx', None)))

        if train is None or val is None or test is None:
            raise KeyError(f"Unknown split keys: {list(split_obj.keys())}")

        return _to_index(train), _to_index(val), _to_index(test)

    if isinstance(split_obj, (list, tuple)) and len(split_obj) >= 3:
        return _to_index(split_obj[0]), _to_index(split_obj[1]), _to_index(split_obj[2])

    if isinstance(split_obj, np.ndarray) and split_obj.ndim >= 1 and len(split_obj) >= 3:
        return _to_index(split_obj[0]), _to_index(split_obj[1]), _to_index(split_obj[2])

    raise ValueError(f"Unsupported split format: type={type(split_obj)}")


def _find_existing_file(candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p
    raise FileNotFoundError("None of these files exists:\n" + "\n".join(str(x) for x in candidates))


def load_gnia_npz_graph(dataset_name: str, device, args):
    """
    读取 G-NIA 风格的 npz 数据集，并返回与 load_original_graph 对齐的六个对象：
        G, feats, labels, train_idx, val_idx, test_idx

    支持:
        10k_ogbproducts.npz + 10k_ogbproducts_split.npy
        12k_reddit.npz + 12k_reddit_split.npy
    """
    name = str(dataset_name).lower().replace('-', '_')

    data_root = getattr(args, 'data_root', None)
    if data_root is None:
        data_root = os.path.join(os.getcwd(), 'datasets')
    data_root = os.path.expanduser(str(data_root))

    if name in {'10k_ogbproducts', 'ogbproducts', 'ogbn_product', 'ogbn_products', 'products'}:
        base = '10k_ogbproducts'
    elif name in {'12k_reddit', 'reddit', 'reddit10k', 'reddit12k'}:
        base = '12k_reddit'
    elif name in {'gnia_citeseer', 'citeseer_npz'}:
        base = 'citeseer'
    else:
        raise ValueError(f"Unknown G-NIA npz dataset: {dataset_name}")

    npz_path = _find_existing_file([
        os.path.join(data_root, f'{base}.npz'),
        os.path.join(os.getcwd(), 'datasets', f'{base}.npz'),
        os.path.join(os.getcwd(), 'data', f'{base}.npz'),
    ])

    split_path = _find_existing_file([
        os.path.join(data_root, f'{base}_split.npy'),
        os.path.join(os.getcwd(), 'datasets', f'{base}_split.npy'),
        os.path.join(os.getcwd(), 'data', f'{base}_split.npy'),
    ])

    # 1. 读取 npz
    adj, features, labels_np = load_npz(npz_path)

    if features is None:
        raise RuntimeError(f"{npz_path} does not contain node features.")

    # # 2. 对称化 + self-loop，保持和你 citation loader 的图处理风格一致
    # n = adj.shape[0]
    # adj = adj.tocsr()
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(n, format='csr')
    # adj.data[:] = 1.0
    # 2. 对称化
    adj = adj.tocsr()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.data[:] = 1.0

    # 2.1 Reddit 必须先取 LCC；12k_reddit_split.npy 是 LCC 重新编号后的 split
    if base == "12k_reddit":
        n_comp, comp = connected_components(adj, directed=False)
        sizes = np.bincount(comp)
        lcc_id = sizes.argmax()
        keep_nodes = np.where(comp == lcc_id)[0]

        adj = adj[keep_nodes][:, keep_nodes].tocsr()
        features = features[keep_nodes]
        labels_np = labels_np[keep_nodes]

    # 2.2 裁剪后再加 self-loop
    n = adj.shape[0]
    adj = adj + sp.eye(n, format="csr")
    adj.data[:] = 1.0
    # 3. 特征
    if sp.issparse(features):
        feats = torch.FloatTensor(np.asarray(features.todense(), dtype=np.float32))
    else:
        feats = torch.FloatTensor(np.asarray(features, dtype=np.float32))

    # 4. 标签
    labels_np = np.asarray(labels_np)
    if labels_np.ndim > 1:
        labels_np = labels_np.argmax(axis=1)
    labels = torch.LongTensor(labels_np.astype(np.int64)).view(-1)

    # 5. split
    # split_obj = np.load(split_path, allow_pickle=True)
    # idx_train, idx_val, idx_test = _split_to_indices_flexible(split_obj)
    model_name = getattr(args, "model_name", getattr(args, "victim", None))
    split_name = getattr(args, "split_name", "graphdc_aligned")
    seed = int(getattr(args, "seed", 42))

    if split_name == "graphdc_aligned" and base == "10k_ogbproducts":
        idx_train, idx_val, idx_test = get_graphdc_aligned_split(
            labels,
            base,
            split_name="graphdc_aligned",
            seed=seed,
            model_name=model_name,
        )
        split_info = f"graphdc_aligned, model={model_name}"
    else:
        split_obj = np.load(split_path, allow_pickle=True)
        idx_train, idx_val, idx_test = _split_to_indices_flexible(split_obj)
        split_info = f"loaded split={split_path}"

    # 6. DGL 图
    g = dgl.from_scipy(adj).to(device)
    feats = feats.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    g.ndata['feat'] = feats
    g.ndata['label'] = labels

    # 打印边数：这里 G.num_edges() 包含 self-loop 和双向边 entry
    adj_wo_loop = adj.copy()
    adj_wo_loop.setdiag(0)
    adj_wo_loop.eliminate_zeros()
    undirected_edges = adj_wo_loop.nnz // 2

    # print(
    #     f"[Dataset] G-NIA-{base} | "
    #     f"N={g.num_nodes()}, E_entries={g.num_edges()}, E_undir={undirected_edges}, "
    #     f"F={feats.size(1)}, C={int(labels.max().item() + 1)} | "
    #     f"train={idx_train.numel()}, val={idx_val.numel()}, test={idx_test.numel()}"
    # )
    print(
        f"[Dataset] G-NIA-{base} | "
        f"N={g.num_nodes()}, E_entries={g.num_edges()}, E_undir={undirected_edges}, "
        f"F={feats.size(1)}, C={int(labels.max().item() + 1)} | "
        f"train={idx_train.numel()}, val={idx_val.numel()}, test={idx_test.numel()} | "
        f"{split_info}"
    )
    return g, feats, labels, idx_train, idx_val, idx_test


def load_cora_ml_npz_graph(device, args):
    """
    Load Cora-ML from graph2gauss / PyG CitationFull npz format.

    Expected file:
        ./datasets/cora_ml.npz
    Optional split file:
        ./datasets/cora_ml_split.npy

    If split file does not exist, use stratified 10%/10%/80% split,
    consistent with your load_data_citation() setting.
    """
    data_root = getattr(args, "data_root", None)
    if data_root is None:
        data_root = os.path.join(os.getcwd(), "datasets")
    data_root = os.path.expanduser(str(data_root))

    npz_path = _find_existing_file([
        os.path.join(data_root, "cora_ml.npz"),
        os.path.join(data_root, "Cora_ML.npz"),
        os.path.join(os.getcwd(), "datasets", "cora_ml.npz"),
        os.path.join(os.getcwd(), "data", "cora_ml.npz"),
    ])

    split_candidates = [
        os.path.join(data_root, "cora_ml_split.npy"),
        os.path.join(os.getcwd(), "datasets", "cora_ml_split.npy"),
        os.path.join(os.getcwd(), "data", "cora_ml_split.npy"),
    ]
    split_path = None
    for p in split_candidates:
        if os.path.exists(p):
            split_path = p
            break

    # 1. 读取 graph2gauss npz
    adj, features, labels_np = load_npz(npz_path)

    if features is None:
        raise RuntimeError(f"{npz_path} does not contain node features.")
    if labels_np is None:
        raise RuntimeError(f"{npz_path} does not contain labels.")

    # 2. 邻接矩阵处理：对称化，去重。
    # 注意：这里默认不加 self-loop，以保持和你普通 cora/citeseer/pubmed loader 更一致。
    n = adj.shape[0]
    adj = adj.tocsr()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()

    # 3. 特征处理：row-normalize，和 load_data_citation() 中 features = normalize(features) 对齐。
    if sp.issparse(features):
        features = normalize(features)
        feats = torch.FloatTensor(np.asarray(features.todense(), dtype=np.float32))
    else:
        features = sp.csr_matrix(np.asarray(features, dtype=np.float32))
        features = normalize(features)
        feats = torch.FloatTensor(np.asarray(features.todense(), dtype=np.float32))

    # 4. 标签
    labels_np = np.asarray(labels_np)
    if labels_np.ndim > 1:
        labels_np = labels_np.argmax(axis=1)
    labels = torch.LongTensor(labels_np.astype(np.int64)).view(-1)

    # 5. split：优先读取 cora_ml_split.npy；没有则按 10/10/80 分层随机划分
    if split_path is not None:
        split_obj = np.load(split_path, allow_pickle=True)
        idx_train, idx_val, idx_test = _split_to_indices_flexible(split_obj)
        split_info = f"loaded split={split_path}"
    else:
        label_classes = labels.cpu().numpy()
        all_idx = np.arange(len(label_classes))

        train_size = float(getattr(args, "train_ratio", 0.1))
        val_size = float(getattr(args, "val_ratio", 0.1))
        seed = int(getattr(args, "seed", 42))

        idx_train_np, idx_temp_np, _, y_temp = train_test_split(
            all_idx,
            label_classes,
            train_size=train_size,
            stratify=label_classes,
            random_state=seed,
        )

        relative_val_size = val_size / (1.0 - train_size)

        idx_val_np, idx_test_np = train_test_split(
            idx_temp_np,
            train_size=relative_val_size,
            stratify=y_temp,
            random_state=seed,
        )

        idx_train = torch.LongTensor(idx_train_np)
        idx_val = torch.LongTensor(idx_val_np)
        idx_test = torch.LongTensor(idx_test_np)

        split_info = (
            f"generated stratified split | "
            f"train={train_size:.2f}, val={val_size:.2f}, "
            f"test={1.0 - train_size - val_size:.2f}, seed={seed}"
        )

    # 6. DGL graph
    g = dgl.from_scipy(adj).to(device)

    feats = feats.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    g.ndata["feat"] = feats
    g.ndata["label"] = labels

    # 7. 打印统计
    undirected_edges = adj.nnz // 2
    print(
        f"[Dataset] Cora-ML | "
        f"N={g.num_nodes()}, E_entries={g.num_edges()}, E_undir={undirected_edges}, "
        f"F={feats.size(1)}, C={int(labels.max().item() + 1)} | "
        f"train={idx_train.numel()}, val={idx_val.numel()}, test={idx_test.numel()} | "
        f"{split_info}"
    )

    return g, feats, labels, idx_train, idx_val, idx_test

# ---------- 修改你的入口 ----------
def load_original_graph(dataset: str, device, args):
    name = dataset.lower()
    model_name = getattr(args, "model_name", getattr(args, "victim", None))
    split_name = getattr(args, "split_name", "graphdc_aligned")
    seed = int(getattr(args, "seed", 42))

    if name in {'cora', 'citeseer', 'pubmed'}:
        _, feats, labs, _, _, _, adj = load_data_citation(name)

        if split_name == "graphdc_aligned":
            tr, va, te = get_graphdc_aligned_split(
                labs,
                name,
                split_name="graphdc_aligned",
                seed=seed,
                data_root=file_dir_citation,
                model_name=model_name,
            )
        else:
            tr, va, te = _stratified_ratio_split(labs, 0.1, 0.1, seed)

        g = dgl.from_scipy(adj).to(device)
        g.ndata['feat'] = feats.to(device)
        g.ndata['label'] = labs.to(device)

        print(
            f"[load_original_graph] dataset={name}, model={model_name}, "
            f"split={split_name}, train={tr.numel()}, val={va.numel()}, test={te.numel()}"
        )

        return (
            g,
            feats.to(device),
            labs.to(device),
            tr.to(device),
            va.to(device),
            te.to(device),
        )
    
    if dataset.lower() in {"cora_ml", "coraml", "cora_full_ml"}:
        return load_cora_ml_npz_graph(device, args)
    if dataset.lower() in {
        '10k_ogbproducts',
        'ogbproducts',
        'ogbn_product',
        'ogbn_products',
        'products',
        '12k_reddit',
        'reddit',
        'reddit10k',
        'reddit12k',
        'gnia_citeseer',
        'citeseer_npz',
    }:
        return load_gnia_npz_graph(dataset.lower(), device, args)
    
    if dataset.lower() == 'pubmed79':
        _, feats, labs, tr, va, te, adj = load_data_citation_public('pubmed')

        g = dgl.from_scipy(adj).to(device)
        g.ndata['feat'] = feats.to(device)
        g.ndata['label'] = labs.to(device)

        print(
            f"[load_original_graph] dataset=pubmed79_public | "
            f"N={g.num_nodes()}, E_entries={g.num_edges()}, "
            f"F={feats.size(1)}, C={int(labs.max().item() + 1)} | "
            f"train={tr.numel()}, val={va.numel()}, test={te.numel()}"
        )

        return (
            g,
            feats.to(device),
            labs.to(device),
            tr.to(device),
            va.to(device),
            te.to(device),
        )
    else:
        root_dir = os.getcwd() + '/new_data'+'/{}'.format(dataset)
        assert root_dir is not None and os.path.isdir(root_dir), \
            f"args.geom_root 不存在：{root_dir}"
        # 1) 读原始图
        adj, feats, labs = _load_geom_graph_from_txt(root_dir, dataset)

        # 2) 读官方 split 掩码（.npz）
        split_id = int(getattr(args, 'split_id', 0))
        train_ratio = float(getattr(args, 'train_ratio', 0.6))
        val_ratio = float(getattr(args, 'val_ratio', 0.2))
        split_path = _pick_geom_split(root_dir, dataset, train_ratio, val_ratio, split_id)
        with np.load(split_path) as s:
            # 常见命名：train_mask / val_mask / test_mask
            if 'train_mask' in s:
                tr_mask = th.as_tensor(s['train_mask'], dtype=th.bool)
                va_mask = th.as_tensor(s['val_mask'], dtype=th.bool)
                te_mask = th.as_tensor(s['test_mask'], dtype=th.bool)
            else:
                tr_mask = th.as_tensor(s['train_masks'][0], dtype=th.bool)
                va_mask = th.as_tensor(s['val_masks'][0], dtype=th.bool)
                te_mask = th.as_tensor(s['test_masks'][0], dtype=th.bool)

        # 3) 构 DGL 图（默认不加自环；若模型需要，外面再 add_self_loop）
        g = dgl.from_scipy(adj).to(device)
        feats = feats.to(device)
        labs = labs.to(device)
        g.ndata['feat'] = feats
        g.ndata['label'] = labs

        # 4) 掩码 -> 索引
        tr = th.nonzero(tr_mask, as_tuple=False).view(-1).to(device)
        va = th.nonzero(va_mask, as_tuple=False).view(-1).to(device)
        te = th.nonzero(te_mask, as_tuple=False).view(-1).to(device)

        return g, feats.to(device), labs.to(device), tr.to(device), va.to(device), te.to(device)

    return g, feats.to(device), labs.to(device), tr.to(device), va.to(device), te.to(device)




def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 让 cuDNN 更可复现（如需极致复现可打开下面两行）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False



def _try_tensor_device_from_graph(g: dgl.DGLGraph) -> Optional[torch.device]:
    # 旧版 DGL 没有 g.device；尽量从 ndata 拿一个 tensor 的 device
    try:
        if hasattr(g, "device") and g.device is not None:
            return g.device
    except Exception:
        pass
    try:
        for k, v in g.ndata.items():
            if isinstance(v, torch.Tensor):
                return v.device
    except Exception:
        pass
    return None


def device_of(x: Any, default: str | torch.device = "cpu") -> torch.device:
    """
    返回对象所在 device：
      - torch.Tensor：直接返回 x.device
      - dgl.DGLGraph：优先 g.device，否则从 ndata 的任意张量推断
      - 其他：返回 default
    """
    if isinstance(x, torch.Tensor):
        return x.device
    if isinstance(x, dgl.DGLGraph):
        dev = _try_tensor_device_from_graph(x)
        if dev is not None:
            return dev
    return torch.device(default)


def to_cpu(x: Any):
    """把 Tensor / DGLGraph 安全迁移到 CPU。其他类型原样返回。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dgl.DGLGraph):
        return x.to(torch.device("cpu"))
    try:
        return x.to("cpu")
    except Exception:
        return x


def to_device(x: Any, dev: str | torch.device):
    """把 Tensor / DGLGraph 迁移到 dev。其他类型尝试 .to，失败则原样返回。"""
    if not isinstance(dev, torch.device):
        dev = torch.device(dev)
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, dgl.DGLGraph):
        return x.to(dev)
    try:
        return x.to(dev)
    except Exception:
        return x


def clone_graph_with_data(g: dgl.DGLGraph,
                          feats: Optional[torch.Tensor] = None,
                          labels: Optional[torch.Tensor] = None) -> dgl.DGLGraph:
    """
    复制图结构与数据。兼容旧版 DGL（不用 g.clone()）。
    会复制 ndata/edata（逐 key clone），并在给定时覆盖 'feat'/'label'。
    """
    u, v = g.edges()
    newg = dgl.graph((u, v), num_nodes=g.num_nodes())
    # 复制图级属性
    try:
        for k in g.ndata:
            if isinstance(g.ndata[k], torch.Tensor):
                newg.ndata[k] = g.ndata[k].clone()
            else:
                newg.ndata[k] = g.ndata[k]
    except Exception:
        pass
    try:
        for k in g.edata:
            if isinstance(g.edata[k], torch.Tensor):
                newg.edata[k] = g.edata[k].clone()
            else:
                newg.edata[k] = g.edata[k]
    except Exception:
        pass
    if feats is not None:
        newg.ndata["feat"] = feats.clone()
    if labels is not None:
        newg.ndata["label"] = labels.clone()
    # 尽量放回原 device
    dev = _try_tensor_device_from_graph(g)
    if dev is not None:
        newg = newg.to(dev)
    return newg
