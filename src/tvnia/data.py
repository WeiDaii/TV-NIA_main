"""Focused dataset loaders for the six reported TV-NIA benchmarks."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import dgl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split


DATASET_DISPLAY_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "cora_ml": "Cora-ML",
    "pubmed": "PubMed",
    "ogbn_product": "Ogbn-product",
    "reddit": "Reddit",
}
DATASET_IDS = tuple(DATASET_DISPLAY_NAMES)
CITATION_DATASETS = frozenset({"cora", "citeseer", "cora_ml", "pubmed"})


@dataclass(frozen=True)
class GraphDataset:
    dataset_id: str
    graph: dgl.DGLGraph
    features: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    test_indices: torch.Tensor

    @property
    def split(self) -> Dict[str, torch.Tensor]:
        return {
            "train": self.train_indices,
            "val": self.validation_indices,
            "test": self.test_indices,
        }


def canonical_dataset_id(name: str) -> str:
    normalized = name.lower().replace("-", "_")
    if normalized not in DATASET_DISPLAY_NAMES:
        raise ValueError(f"Unsupported dataset: {name}")
    return normalized


def _row_normalize(matrix: sp.spmatrix) -> sp.csr_matrix:
    row_sum = np.asarray(matrix.sum(1), dtype=np.float32).reshape(-1)
    with np.errstate(divide="ignore"):
        inverse = np.power(row_sum, -1)
    inverse[~np.isfinite(inverse)] = 0.0
    return sp.diags(inverse).dot(matrix).tocsr()


def _symmetrize(adjacency: sp.spmatrix) -> sp.csr_matrix:
    adjacency = adjacency.tocsr()
    result = adjacency + adjacency.T.multiply(adjacency.T > adjacency) - adjacency.multiply(
        adjacency.T > adjacency
    )
    result.data[:] = 1.0
    return result.tocsr()


def _stratified_split(
    labels: torch.Tensor,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels_np = labels.cpu().numpy()
    indices = np.arange(labels_np.shape[0])
    _, class_counts = np.unique(labels_np, return_counts=True)
    first_stratification = labels_np if class_counts.min() >= 2 else None
    train, remainder, _, remainder_labels = train_test_split(
        indices,
        labels_np,
        train_size=train_ratio,
        stratify=first_stratification,
        random_state=seed,
    )
    relative_validation = validation_ratio / (1.0 - train_ratio)
    if first_stratification is None:
        second_stratification = None
    else:
        _, remainder_counts = np.unique(remainder_labels, return_counts=True)
        second_stratification = remainder_labels if remainder_counts.min() >= 2 else None
    validation, test = train_test_split(
        remainder,
        train_size=relative_validation,
        stratify=second_stratification,
        random_state=seed,
    )
    return tuple(torch.as_tensor(values, dtype=torch.long) for values in (train, validation, test))


def _load_planetoid(
    data_root: Path, dataset_id: str
) -> tuple[sp.csr_matrix, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    objects = []
    for suffix in ("x", "y", "tx", "ty", "allx", "ally", "graph"):
        with (data_root / f"ind.{dataset_id}.{suffix}").open("rb") as stream:
            objects.append(pickle.load(stream, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph_dict = objects
    with (data_root / f"ind.{dataset_id}.test.index").open(encoding="utf-8") as stream:
        test_reordered = [int(line.strip()) for line in stream]
    test_sorted = np.sort(test_reordered)

    if dataset_id == "citeseer":
        full_range = range(min(test_reordered), max(test_reordered) + 1)
        extended_features = sp.lil_matrix((len(full_range), x.shape[1]))
        extended_features[test_sorted - min(test_sorted), :] = tx
        tx = extended_features
        extended_labels = np.zeros((len(full_range), y.shape[1]))
        extended_labels[test_sorted - min(test_sorted), :] = ty
        ty = extended_labels

    feature_matrix = sp.vstack((allx, tx)).tolil()
    feature_matrix[test_reordered, :] = feature_matrix[test_sorted, :]
    features = torch.as_tensor(
        _row_normalize(feature_matrix).toarray(), dtype=torch.float32
    )
    labels_one_hot = np.vstack((ally, ty))
    labels_one_hot[test_reordered, :] = labels_one_hot[test_sorted, :]
    labels = torch.as_tensor(labels_one_hot.argmax(axis=1), dtype=torch.long)

    adjacency = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
    adjacency = _symmetrize(adjacency) + sp.eye(adjacency.shape[0], format="csr")
    train = torch.arange(y.shape[0], dtype=torch.long)
    validation = torch.arange(y.shape[0], y.shape[0] + 500, dtype=torch.long)
    test = torch.as_tensor(test_sorted, dtype=torch.long)
    return adjacency, features, labels, train, validation, test


def _load_npz(path: Path) -> tuple[sp.csr_matrix, sp.spmatrix, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        adjacency = sp.csr_matrix(
            (archive["adj_data"], archive["adj_indices"], archive["adj_indptr"]),
            shape=archive["adj_shape"],
        )
        attributes = sp.csr_matrix(
            (archive["attr_data"], archive["attr_indices"], archive["attr_indptr"]),
            shape=archive["attr_shape"],
        )
        labels = archive["labels"]
    return adjacency, attributes, labels


def _split_from_numpy(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise TypeError(f"Expected a split dictionary in {path}")

    def indices(*names: str) -> torch.Tensor:
        selected = next((value[name] for name in names if name in value), None)
        if selected is None:
            raise KeyError(f"Missing one of {names} in {path}")
        selected = np.asarray(selected)
        if selected.dtype == np.bool_:
            selected = np.where(selected)[0]
        return torch.as_tensor(selected.astype(np.int64), dtype=torch.long)

    return indices("train", "idx_train"), indices("val", "valid", "idx_val"), indices("test", "idx_test")


def _load_cora_ml(data_root: Path, seed: int):
    adjacency, attributes, labels_np = _load_npz(data_root / "cora_ml.npz")
    adjacency = _symmetrize(adjacency)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    features = torch.as_tensor(_row_normalize(attributes).toarray(), dtype=torch.float32)
    labels_np = np.asarray(labels_np)
    labels = torch.as_tensor(
        labels_np.argmax(axis=1) if labels_np.ndim > 1 else labels_np,
        dtype=torch.long,
    ).view(-1)
    train, validation, test = _stratified_split(labels, 0.1, 0.1, seed)
    return adjacency, features, labels, train, validation, test


def _load_sampled_network(
    data_root: Path,
    dataset_id: str,
    split_strategy: str,
    split_seed: int,
):
    base = dataset_id
    adjacency, attributes, labels_np = _load_npz(data_root / f"{base}.npz")
    adjacency = _symmetrize(adjacency)
    if dataset_id == "reddit":
        _, components = connected_components(adjacency, directed=False)
        largest = np.bincount(components).argmax()
        keep = np.where(components == largest)[0]
        adjacency = adjacency[keep][:, keep].tocsr()
        attributes = attributes[keep]
        labels_np = labels_np[keep]
    adjacency = adjacency + sp.eye(adjacency.shape[0], format="csr")
    adjacency.data[:] = 1.0
    features = torch.as_tensor(attributes.toarray(), dtype=torch.float32)
    labels_np = np.asarray(labels_np)
    labels = torch.as_tensor(
        labels_np.argmax(axis=1) if labels_np.ndim > 1 else labels_np,
        dtype=torch.long,
    )

    if split_strategy == "stratified_60_20_20":
        train, validation, test = _stratified_split(labels, 0.6, 0.2, split_seed)
    elif split_strategy == "stored_split":
        train, validation, test = _split_from_numpy(data_root / f"{base}_split.npy")
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    return adjacency, features, labels, train, validation, test


def load_dataset(
    dataset: str,
    victim: str,
    data_root: Path,
    device: torch.device,
    seed: int = 42,
    split_profile: Dict[str, object] | None = None,
) -> GraphDataset:
    """Load one benchmark with the split policy from the final experiment."""
    dataset_id = canonical_dataset_id(dataset)
    victim = victim.lower()
    split_profile = split_profile or {}
    split_strategy = str(split_profile.get("strategy", ""))
    split_seed = int(split_profile.get("seed", seed))
    if dataset_id in {"cora", "citeseer", "pubmed"}:
        adjacency, features, labels, train, validation, test = _load_planetoid(
            data_root, dataset_id
        )
        if split_strategy == "stratified_10_10_80":
            train, validation, test = _stratified_split(labels, 0.1, 0.1, split_seed)
        elif split_strategy not in {"", "planetoid_public"}:
            raise ValueError(f"Unsupported split strategy: {split_strategy}")
    elif dataset_id == "cora_ml":
        adjacency, features, labels, train, validation, test = _load_cora_ml(
            data_root, split_seed
        )
    else:
        adjacency, features, labels, train, validation, test = _load_sampled_network(
            data_root, dataset_id, split_strategy, split_seed
        )

    graph = dgl.from_scipy(adjacency)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph).to(device)
    features = features.float().to(device)
    labels = labels.long().to(device)
    train, validation, test = (
        train.long().to(device),
        validation.long().to(device),
        test.long().to(device),
    )
    graph.ndata["features"] = features
    graph.ndata["labels"] = labels
    return GraphDataset(
        dataset_id,
        graph,
        features,
        labels,
        train,
        validation,
        test,
    )
