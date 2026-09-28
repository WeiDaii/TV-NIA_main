# TV-NIA clean reproduction package

This directory is an independent, minimal implementation of the final TV-NIA
protocol. It contains only the paper's main experiment and defense evaluation.
No ablation, hyperparameter sweep, visualization, or exploratory runner is
included. The original `TV-NIA_Experiment` directory is not imported or
modified at runtime.

## Project layout

```text
TV-NIA_main/
├── configs/final_no_seek_exper_3.json  # single parameter source
├── data/                               # six datasets + SHA-256 manifest
├── results/
│   ├── reference/                      # frozen final_no_seek_exper_3 results
│   ├── main/                           # new main-run output
│   └── defense/                        # new defense-run output
├── scripts/
│   ├── run_main_experiment.py
│   ├── run_defense_experiment.py
│   └── verify_results.py
└── src/tvnia/
    ├── vulnerable_topology.py
    ├── contrastive_reconstruction.py
    ├── zeroth_order_features.py
    ├── framework.py
    ├── data.py
    ├── models.py
    └── defenses/
```

## Canonical names

- Datasets: `cora`, `citeseer`, `cora_ml`, `pubmed`, `ogbn_product`, `reddit`.
- Display names: Cora, CiteSeer, Cora-ML, PubMed, Ogbn-product, Reddit.
- Victims: `gcn`, `graphsage`, `gin`.
- TV-NIA modules: vulnerable topology construction, contrastive topology
  reconstruction, and zeroth-order feature refinement.
- Defenses: `gnnguard`, `rgcn`, `prognn`, `gtrans`.

The sampled-network files also use the canonical names `ogbn_product.*` and
`reddit.*`; legacy experimental names do not appear in the new project.

## Final protocol

All defaults come from `final_no_seek_exper_3`:

- seed 42; injection ratio 5%; no injected-to-injected edges;
- GCN, GraphSAGE, and GIN hidden dimension 64;
- clean/poison victim training: 100/200 epochs, patience 30, Adam learning rate
  0.01, weight decay 0.0005;
- vulnerability weights 0.45/0.35/0.20; preselection limit 512; anchor
  refinement query budget 0;
- contrastive encoder/projection dimensions 256/32, temperature 0.4, learning
  rate 0.05, weight decay 0.00001, 20 training epochs, 3 structure iterations,
  16 actions each; contrastive feature updates 0 (step size 1.0 is inactive);
- citation networks: train-anchor scope, 16 candidates, wrong-class prototype
  initialization with mixing coefficient 0.65, unit-box projection, feature
  top-k 50, 60 feature-refinement steps, perturbation scale 0.1, learning rate
  0.8, all affected evaluation nodes, local-node cap 1024;
- sampled networks: test-anchor scope, 64 candidates, anti-class boundary
  initialization, automatic feature domain, feature top-k 80, no feature
  refinement, local-node cap 2048;
- defense training: 200 epochs, patience 30, hidden dimension 64, dropout 0.5,
  learning rate 0.01, weight decay 0.0005, threshold 0.1, with GCN as the
  attack oracle.

The injected-node count uses `ceil(0.05 × |V|)`, exactly as the source run did.
The per-injected-node degree is the rounded average directed degree after
self-loops are removed.

Model-dependent data splits are also explicit in the JSON configuration:
Cora and CiteSeer use the Planetoid public split for GCN/GIN and stratified
10/10/80 splits for GraphSAGE (seeds 42 and 0); Cora-ML uses stratified
10/10/80 with seed 42; PubMed uses its public split; Ogbn-product uses
stratified 60/20/20 with seed 1 for GCN/GIN and seed 123 for GraphSAGE; Reddit
uses its stored split.

## Run

From this directory:

```powershell
python scripts/run_main_experiment.py
python scripts/run_defense_experiment.py
```

Both runners are resumable: successful dataset/model rows already present in
their output CSV are skipped. A subset can be run explicitly, for example:

```powershell
python scripts/run_main_experiment.py --datasets cora --victims gcn
python scripts/run_defense_experiment.py --datasets cora --defenses gnnguard
```

## Verify

```powershell
python scripts/verify_results.py data
python scripts/verify_results.py main
python scripts/verify_results.py defense
```

`results/reference/main_tvnia_reference.csv` contains all 18 final TV-NIA
dataset-victim cases. `defense_tvnia_reference.csv` contains the 24 reported
dataset-defense cases with GCN as the attack oracle. The reference values were
extracted from the completed full-configuration rows of
`final_no_seek_exper_3`; no ablation result is shipped in this project.

The CUDA builds of DGL and `torch-sparse` may require installation from their
official version-specific wheel indexes. The versions in `requirements.txt`
record the validated environment; they do not replace the vendor-specific CUDA
installation instructions.
