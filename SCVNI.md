# SCVNI：稀疏协作式恶意节点注入攻击

## 1. 方法目标

SCVNI 是一个用于 GNN 节点分类任务的白盒节点注入攻击方法。它去掉了原来的强化学习流程，改成直接的可微优化。当前实现的主要目标是让代码结构更清楚，便于阅读、分析和继续修改。

主运行文件是：

```text
main_new.py
```

方法实现文件是：

```text
sc_vnia.py
```

默认评估会同时报告两个结果：

1. **逃逸攻击准确率**：固定干净受害模型，不重新训练模型，只在注入图上评估原干净受害模型。
2. **投毒攻击准确率**：在注入图上从头重新训练一个新的受害模型，然后评估重训后的模型。

其中，投毒攻击准确率是主要结果；逃逸攻击准确率只作为诊断参考，用来看生成的注入结构对固定模型是否已经有扰动效果。

## 2. 威胁模型

SCVNI 当前是一个白盒上界攻击。攻击者可以访问已经训练好的干净受害模型，并且可以对注入图中的连续变量求梯度。

攻击者不能修改：

- 原始节点；
- 原始节点特征；
- 原始节点之间的原始边。

攻击者只能控制：

- 注入节点；
- 注入节点特征；
- 注入节点与原始节点之间的跨边；
- 注入节点之间的内部边。

投毒后的图可以写成：

\[
A' =
\begin{bmatrix}
A & B_{out}^T \\
B_{out} & A_{inj}
\end{bmatrix},
\quad
X' =
\begin{bmatrix}
X \\
X_{inj}
\end{bmatrix}.
\]

其中：

- `A` 和 `X` 是固定不变的原始图结构和原始节点特征；
- `B_out` 是注入节点与原始节点之间的跨边矩阵块；
- `A_inj` 是注入节点之间的内部协作边；
- `X_inj` 是注入节点特征矩阵。

默认情况下，攻击生成阶段使用干净受害模型的预测作为伪标签，而不是使用测试节点真实标签。这个行为由参数控制：

```text
--attack_label_mode pseudo
```

如果设置为：

```text
--attack_label_mode true
```

则会使用真实标签，这是更强的“标签可见”白盒上界攻击，但现实性更弱。

## 3. 数据集加载

`main_new.py` 通过下面的接口加载数据：

```python
from utils import load_original_graph
```

默认使用：

```text
split_name="graphdc_aligned"
```

也就是说，当前数据集划分会优先复用 `utils.py` 中已经实现的 GraphDC 风格对齐逻辑。

数据加载后，DGL 图会被转换成不含自环的稠密 0/1 邻接矩阵。随后，`sc_vnia.py` 使用稠密邻接矩阵进行 `SimpleGCN` 训练和可微攻击优化。

## 4. 主流程

`main_new.py` 当前执行流程如下：

1. 使用 GraphDC 对齐加载器加载图、特征、标签、训练集、验证集和测试集。
2. 将 DGL 图转换成稠密邻接矩阵。
3. 训练一个干净的 `SimpleGCN` 受害模型。
4. 如果没有使用 `--attack_label_mode true`，则用干净受害模型预测生成伪标签。
5. 从测试集划分中选择低分类间隔的目标节点。
6. 运行 SCVNI，优化注入节点、跨边、内部边和注入特征。
7. 在固定干净受害模型上评估逃逸攻击准确率。
8. 在投毒图上从头重训新的受害模型，并报告投毒攻击准确率。

## 5. 攻击优化变量

SCVNI 不是一开始就构造一个离散的硬注入图，而是先初始化一组连续变量：

- `cross_logits`：注入节点到候选原始节点的软跨边掩码；
- `intra_logits`：注入节点之间的软内部边掩码；
- `feature_logits`：注入节点特征 logits；
- `gate_logits`：注入节点是否活跃的门控变量。

对应的软变量是：

\[
g_i = \sigma(\alpha_i),
\]

\[
P_{cross} = \sigma(\Theta_{cross}) \odot g,
\]

\[
P_{intra}^{ij} = \sigma(\Theta_{intra}^{ij}) g_i g_j,
\]

\[
X_{inj} = \sigma(\Phi).
\]

其中，活跃门控的作用是让冗余注入节点在优化过程中逐渐被压低，从而实现一定程度的稀疏化。

## 6. 候选节点池

SCVNI 不在全图所有节点上搜索跨边，而是把搜索空间限制在目标攻击节点附近的 `k` 跳候选池：

\[
\mathcal{C}=\mathcal{N}_k(\mathcal{T}).
\]

这样可以让攻击更局部，也能减少 dense 搜索空间。候选节点数量由下面的参数限制：

```text
--max_candidates
```

## 7. 优化目标

SCVNI 在选中的目标节点上最大化白盒分类攻击目标：

\[
\mathcal{L}_{atk}
= CE(f_\theta(G')_\mathcal{T}, \hat{Y}_\mathcal{T})
+ \beta \cdot
\left(\max_{c\ne \hat{y}_v} z_{v,c} - z_{v,\hat{y}_v}\right).
\]

直观理解：

- 第一项让模型在目标节点上的分类损失变大；
- 第二项鼓励错误类别输出值超过当前伪标签或真实标签对应的输出值。

同时，方法会对恶意资源进行惩罚：

\[
\mathcal{R}
= \lambda_g \sum_i g_i
+ \lambda_c \|P_{cross}\|_1
+ \lambda_i \|P_{intra}\|_1
+ \lambda_x \|X_{inj}\|_1.
\]

最终优化的损失函数是：

\[
\min -\mathcal{L}_{atk} + \mathcal{R}.
\]

也就是：一方面增强攻击目标，另一方面压缩注入节点、注入边和注入特征的规模。

## 8. 离散化

优化结束后，SCVNI 会把软变量解码成离散注入图：

1. 根据门控分数保留排名靠前的注入节点，由 `--n_inj_final` 控制。
2. 根据 `--edge_budget_cross` 或 `--cross_per_node_budget` 保留排名靠前的跨边。
3. 根据 `--edge_budget_intra` 保留排名靠前的内部边。
4. 根据 `--feature_topk` 保留排名靠前的特征维度；如果禁用 `feature_topk`，则用阈值生成注入特征。

最终得到的离散图会同时用于逃逸评估和投毒评估。

## 9. 重要文件

| 文件 | 作用 |
|---|---|
| `main_new.py` | 主实验运行文件，不包含强化学习代码 |
| `sc_vnia.py` | SCVNI 攻击实现、稠密 GCN 训练、目标选择和评估工具 |
| `utils.py` | 数据集加载和 GraphDC 对齐划分逻辑 |

## 10. 常用命令

基础运行命令：

```bash
python main_new.py \
  --dataset cora \
  --device cuda:0 \
  --split_name graphdc_aligned \
  --victim gcn \
  --attack_label_mode pseudo \
  --n_inj_max 30 \
  --n_inj_final 15 \
  --edge_budget_cross 120 \
  --edge_budget_intra 30 \
  --attack_steps 300
```

如果要运行更强但现实性较弱的标签可见白盒上界攻击：

```bash
python main_new.py --dataset cora --attack_label_mode true
```

## 11. 后续开发注意事项

当前实现使用稠密邻接矩阵，因此更适合 Cora、Citeseer、Pubmed 这类中小规模图。

对于 Reddit、Products 这类大图，下一步工程方向应该是把 SCVNI 改成稀疏优化或子图优化，否则显存和运行时间会很重。

当前投毒目标是：先利用干净受害模型的梯度生成投毒图，再在投毒图上从头重训新的受害模型。这个流程可以报告投毒准确率，但攻击优化本身并没有直接对“重训后准确率下降”求梯度。

如果要做更强的投毒攻击，后续可以考虑：

- 展开式重训练；
- 双层优化；
- 替代模型重训练奖励；
- 只在子图上做快速近似重训。

## 12. 代码分析结论：优势、弊端与 GraphDC/CaVNI 预算对齐

本节只基于当前 `SCVNIA` 目录下代码阅读得到结论，暂不修改代码。

### 12.1 当前方法实际做了什么

| 模块 | 代码位置 | 当前逻辑 |
|---|---|---|
| 主入口 | `main_new.py` | 加载 GraphDC 对齐划分，转稠密邻接矩阵，训练 `SimpleGCN`，运行 SCVNI，报告逃逸攻击和投毒攻击准确率 |
| 攻击实现 | `sc_vnia.py` | 用连续变量优化注入节点、注入特征、注入-原图跨边、注入节点内部边 |
| 干净受害模型 | `sc_vnia.py::SimpleGCN` | 使用稠密邻接矩阵的两层 GCN，不是 `victim_models.py` 里的 DGL GCN/SAGE |
| 数据划分 | `utils.py::get_graphdc_aligned_split` | 已有 GraphDC 对齐配置，`main_new.py` 默认 `--split_name graphdc_aligned` |
| 标签使用 | `main_new.py` | 默认 `--attack_label_mode pseudo`，攻击阶段使用干净受害模型伪标签；`--attack_label_mode true` 是更强但不现实的标签可见上界 |
| 投毒评估 | `main_new.py` | 注入图生成后，新的 `SimpleGCN` 从头重训，测试原始测试节点 |

### 12.2 方法优势

| 优势 | 说明 |
|---|---|
| 实现清晰 | `main_new.py` 和 `sc_vnia.py` 职责分明，去掉了 G2A2C 的慢速逐目标强化学习流程，便于继续改造 |
| 原图保护较好 | `decode()` 里拼接 `[adj_orig, B^T; B, A_inj]`，原始-原始邻接块保持来自 `adj_orig`，不直接改原始节点、原始特征、原始-原始边 |
| 白盒上界强 | 能对干净受害模型反向传播，直接优化边、特征和门控变量，理论上比纯启发式更强 |
| 支持协作式注入节点 | 有 `A_inj` 内部注入节点连接，能让恶意节点之间协同，而不是每个注入节点孤立工作 |
| 支持稀疏化 | 门控、跨边、内部边、特征都有 L1/预算惩罚，并在解码阶段做排名离散化 |
| 默认不使用测试节点真实标签 | `attack_label_mode=pseudo` 更接近黑盒或半黑盒评估；`true` 模式可以作为上界对照 |
| 同时报告逃逸与投毒 | 固定干净受害模型上的逃逸结果能诊断攻击生成是否有效，从头重训后的投毒结果更符合最终投毒评估 |
| 数据划分层面可以对齐 GraphDC | `main_new.py` 默认传 `split_name="graphdc_aligned"`，`utils.py` 里已有 Cora/Citeseer/Pubmed/10k_ogbproducts 的模型特定划分配置 |

### 12.3 方法弊端

| 弊端 | 影响 |
|---|---|
| 不是黑盒方法 | SCVNI 是白盒可微优化，需要访问干净受害模型参数和梯度；不能直接声称是 GraphDC 式黑盒攻击 |
| 不是 GraphDC 受害模型实现 | 主流程训练的是稠密 `SimpleGCN`，不是 `victim_models.py` 中已对齐的 DGL GCN/SAGE；因此干净准确率不能天然继承前面 GraphDC 干净准确率对齐结果 |
| 不支持 SAGE 真实攻击 | `--victim sage` 当前主要用于数据划分对齐，实际攻击和重训仍是 `SimpleGCN`，所以不能说已经评估 GraphSAGE 投毒 |
| 稠密邻接矩阵扩展性差 | `adj` 被转成 `[N,N]` 稠密矩阵，Products/Reddit 这类图会有很大的显存和时间压力 |
| 攻击目标仍偏逃逸攻击 | 优化目标是固定干净受害模型上选定目标节点的交叉熵和分类间隔，不是直接优化投毒重训后的验证集或测试集准确率下降 |
| 目标选择来自测试集划分 | 默认用 `idx_test` 选低分类间隔目标；虽然伪标签模式不读测试节点真实标签，但攻击构造仍利用了测试节点集合，这对严格投毒设置不够干净 |
| 预算是全局排名，不是每个注入节点严格度约束 | `edge_budget_cross` 是全局跨边数，`edge_budget_intra` 是全局内部边数；只有设置 `cross_per_node_budget` 时才限制每个注入节点的跨边数，但内部边仍可能让单个注入节点超预算 |
| 注入节点数是固定参数 | 默认 `n_inj_final=15`，没有按 GraphDC/GANI/CaVNI 常用的 `rho * num_nodes` 自动计算 |
| 特征预算未对齐 CaVNI/GraphDC | `feature_topk=50` 是固定值，不是按原图平均 feature L0 或 GraphDC/GANI 特征约束自动给出 |
| 内部注入边可能破坏不可察觉性 | 协作式内部边对攻击有利，但如果 GraphDC/CaVNI 的预算只按每个注入节点平均度控制，内部边必须计入注入节点度，否则比较不公平 |

### 12.4 是否可以对齐 GraphDC / CaVNI 的注入节点和注入边设置

结论：**当前代码不能自动对齐；但方法结构可以改造成对齐。**

#### 12.4.1 split 对齐情况

| 项目 | 当前状态 | 结论 |
|---|---|---|
| GraphDC 对齐划分 | `main_new.py` 默认 `--split_name graphdc_aligned` | 可以对齐 |
| 模型特定划分 | `utils.py` 根据数据集和受害模型选择划分与随机种子 | 可以对齐 |
| 干净准确率对齐 | 当前干净受害模型是稠密 `SimpleGCN` | 不一定对齐，需要单独测试或换成 `victim_models.py` |

#### 12.4.2 注入节点预算对齐情况

GraphDC/GANI/CaVNI 风格一般应写成：

```python
n_inj = int(round(rho * num_original_nodes))
```

当前 SCVNI 写法：

```text
--n_inj_max 30
--n_inj_final 15
```

因此：

| 数据集 | GraphDC/CaVNI 5% 注入节点应为 | 当前默认 SCVNI |
|---|---:|---:|
| Cora, N=2708 | round(0.05 * 2708) = 135 | 15 |
| Citeseer, N=3327 | round(0.05 * 3327) = 166 | 15 |
| Pubmed, N=19717 | round(0.05 * 19717) = 986 | 15 |
| 10k_ogbproducts, N=10494 | round(0.05 * 10494) = 525 | 15 |

所以当前默认设置明显不是 GraphDC/CaVNI 预算。

#### 12.4.3 注入边/度预算对齐情况

GraphDC/GANI 风格更合理的度预算是：

```python
delta_deg = int(round(2 * num_undirected_edges / num_original_nodes))
```

也就是每个注入节点最多连接 `delta_deg` 条边。

如果使用注入节点之间的内部边，这些边也必须计入每个注入节点的度预算。

CaVNI 当前主代码中也有类似平均度预算思想：

```python
topk = round(G.num_nodes() * inj_node_budget)
deg_per_node = auto average-degree based budget
```

当前 SCVNI 写法：

```text
--edge_budget_cross 120
--edge_budget_intra 30
--cross_per_node_budget -1
```

这意味着：

- 跨边默认是全局 120 条，不随 `n_inj` 和 `delta_deg` 自动变化；
- 内部边默认是全局 30 条；
- 默认没有每个注入节点的跨边度数上限；
- 内部边没有每个注入节点的度数上限；
- 解码后某些注入节点可能拿到很多跨边和内部边，另一些只有 1 条兜底边。

以 Cora 为例，GraphDC/GANI 5% 设置近似为：

```text
N = 2708
E_undirected = 5278
delta_deg = round(2 * 5278 / 2708) = 4
n_inj = 135
总关联边度预算约 = 135 * 4 = 540
```

当前 SCVNI 默认：

```text
n_inj_final = 15
cross_edges = 120
intra_edges = 30
注入节点关联边度总量 = cross_edges + 2 * intra_edges = 180
平均每个注入节点度 = 180 / 15 = 12
```

这不是同一预算：节点数更少，但每个注入节点平均度更高。若直接和 GraphDC/CaVNI 比，攻击强度和隐蔽性都不公平。

### 12.5 如果要对齐，需要怎么改

暂不改代码，但推荐后续按以下方式改：

| 需要改的点 | 推荐做法 |
|---|---|
| 注入节点数 | 增加 `--rho` 或复用 `--inj_node_budget`，自动设置 `n_inj_final = round(rho * N)` |
| 最大优化节点数 | 设置 `n_inj_max >= n_inj_final`，例如等于 `n_inj_final` 或略大后用门控剪枝 |
| 每节点度预算 | 计算 `delta_deg = round(2 * E_undirected / N)` |
| 跨边解码 | 默认使用 `cross_per_node_budget = delta_deg`，而不是全局 `edge_budget_cross` |
| 内部边解码 | 内部边也要占用每个注入节点剩余度预算，不能只用全局 `edge_budget_intra` |
| 总预算打印 | 在 `main_new.py` 中打印 `N, E_undirected, avg_degree, delta_deg, n_inj` |
| 预算校验 | 解码后检查每个注入节点的 `cross_degree + intra_degree <= delta_deg` |
| 受害模型对齐 | 若要和 GraphDC 干净准确率直接对齐，优先改成复用 `victim_models.py::Victim`，至少对 GCN/SAGE 分别跑干净准确率对齐 |
| 大图可扩展 | Pubmed/Products 不建议使用全图稠密邻接矩阵；应改成稀疏优化或子图优化 |

### 12.6 总体判断

SCVNI 是一个**适合作为白盒上界和机制验证的强攻击原型**。它的优点是可微、清晰，能同时优化跨边、内部边、特征和门控变量，并且不修改原始图块。

但是，它目前还不是一个可以直接和 GraphDC/CaVNI 公平比较的最终方法，主要原因有三点：

1. **预算未对齐**：`n_inj_final/edge_budget_cross/edge_budget_intra` 是固定手动参数，不是 `rho + delta_deg`。
2. **受害模型未对齐**：实际训练和攻击的是稠密 `SimpleGCN`，不是 GraphDC 对齐过的 GCN/SAGE 受害模型。
3. **目标偏白盒逃逸攻击**：攻击优化的是固定干净受害模型上目标节点的交叉熵和分类间隔，而投毒评估只是事后重训，不是直接优化重训后性能下降。

因此，我建议后续路线是：

```text
先保留 SCVNI 的可微优化框架；
再把预算改成 GraphDC/GANI/CaVNI 的 rho + delta_deg；
然后把受害模型和评估流程接到已对齐的 Victim；
最后再考虑稀疏版本或子图版本，以支持 Products/Reddit。
```
