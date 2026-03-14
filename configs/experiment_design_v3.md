# 轨迹质量感知数据选择 — 实验设计 v3

**基座模型**：Qwen2.5-Coder-7B-Instruct  
**微调方法**：LoRA  
**数据来源**：SWE-trajectory 数据集  
**日期**：2026年2月

---

## 一、评分体系设计

### 1.1 设计原则

评分体系回答一个核心问题：**"如果一个人类专家在做同样的任务，他会怎么评价这条轨迹？"**

人类专家评价一个开发者的 debug 过程，关注两件事：**你做得聪不聪明（Efficiency）？你的过程干不干净（Style）？** 至于"做对了没有"（Correctness）和"轨迹完不完整"（Completeness），则作为前置过滤条件，不参与连续评分。

### 1.2 前置过滤（Gate Conditions）

以下条件不参与评分，仅决定轨迹是否进入评分池：

| Gate | 条件 | 理由 |
|------|------|------|
| **Completeness Gate** | Truncation Ratio ≥ 0.9 | 截断比在数据集中几乎为常量（med=1.0, std≈0），无区分力，仅用于清理少量脏数据 |
| **Correctness Gate** | Outcome Success = 1（Resolved） | 二值变量不应与连续变量混合加权；将其作为分层条件，在 resolved 池内排序 |
| **Format Gate** | 轨迹可被解析为 thought-action-observation 结构 | 格式损坏的轨迹无法可靠打分 |

**为什么不把 Correctness 作为评分维度？**  
Outcome Success 是 binary（0/1），与连续维度做 weighted average 会导致评分被 0/1 主导。改为 gate 后，评分池内所有轨迹都是 resolved 的，评分聚焦于"如何更好地完成任务"。

### 1.3 连续评分维度（4维）

过滤后剩余 **32,161 条 resolved 轨迹**，在池内用以下 4 个维度评分：

#### Efficiency（效率）— 达成目标的路径是否精简

| 子维度 | 定义 | 打分方式 | 分布特征 |
|--------|------|----------|----------|
| **B2: Error-Retry Cycles** | 出错后反复重试的代价 | 统计 "action→error observation→类似action" 的循环数，归一化取反 | std=0.286, med=0.300，**区分力最强** |
| **B3: Step Count Ratio** | 步骤数的合理性 | 本轨迹步数 / 同 task 所有 resolved 轨迹的中位步数，clip 后归一化取反 | std=0.063, med=0.800 |

#### Style（风格质量）— 轨迹作为训练数据的干净程度

| 子维度 | 定义 | 打分方式 | 分布特征 |
|--------|------|----------|----------|
| **C2: Action Diversity** | 工具使用是否合理多样 | action type 的 entropy，归一化到 [0,1] | std=0.046, med=0.655 |
| **C3: Observation Utilization** | 是否有效利用了 observation 信息 | observation 中出现的文件名（basename）/错误类名在后续 action 中被引用的比例 | std=0.118, med=0.313 |

#### 被排除的维度及原因

| 维度 | 排除原因 |
|------|----------|
| **B1: Redundant Commands** | std=0.033, med=0.962，几乎无区分力。数据集中的 agent 很少执行完全重复的命令 |
| **C1: Observation Cleanliness** | std=0.043, med=0.967，几乎无区分力。绝大多数 observation 都是干净的 |

> **论文表述**：我们设计了 6 个候选子维度，通过方差分析发现 B1 和 C1 在该数据集上缺乏区分力（std < 0.05），因此排除。这本身是一个发现——SWE-trajectory 数据集的 agent 在命令冗余和 observation 干净程度上已高度同质化。

### 1.4 评分聚合

```
Efficiency = mean(B2, B3)           # std=0.152, med=0.529
Style      = mean(C2, C3)           # std=0.063, med=0.485
Composite  = 0.5 × Efficiency + 0.5 × Style   # std=0.083, med=0.507
```

### 1.5 C3 实现细节

初版 C3 使用完整路径匹配（如 `src/utils.py`），导致 agent 引用 `utils.py`（无路径前缀）时匹配失败，median 仅 0.201。修复后改为 basename 匹配，median 提升至 0.313。剩余的低利用率反映了 agent 普遍存在的"读了不用"问题，本身是一个值得讨论的发现。

---

## 二、实验分组设计

### 2.1 总览（13组 + 1 baseline）

| # | 实验名 | 样本池 | 选择方式 | 样本数 | 所属 Block |
|---|--------|--------|----------|--------|------------|
| 0 | baseline | — | 无微调 | — | — |
| 1 | Random-500 | 全量 | 随机 | 500 | Block 1 |
| 2 | Random-1000 | 全量 | 随机 | 1000 | Block 1 |
| 3 | TopQ-500 | resolved | composite 排序 top | 500 | Block 1 |
| 4 | TopQ-1000 | resolved | composite 排序 top | 1000 | Block 1 |
| 5 | ResolvedOnly-500 | resolved | 随机 | 500 | Block 1 |
| 6 | ResolvedOnly-1000 | resolved | 随机 | 1000 | Block 1 |
| 7 | BottomQ-500 | resolved | composite 倒序 bottom | 500 | Block 1 |
| 8 | Ablation-NoEfficiency-500 | resolved | 仅 Style 排序 | 500 | Block 2 |
| 9 | Ablation-NoStyle-500 | resolved | 仅 Efficiency 排序 | 500 | Block 2 |
| 10 | Ablation-NoB2-500 | resolved | Efficiency=B3 only | 500 | Block 3 |
| 11 | Ablation-NoB3-500 | resolved | Efficiency=B2 only | 500 | Block 3 |
| 12 | Ablation-NoC2-500 | resolved | Style=C3 only | 500 | Block 3 |
| 13 | Ablation-NoC3-500 | resolved | Style=C2 only | 500 | Block 3 |

### 2.2 各 Block 的研究问题

#### Block 1：数据量与策略对比（7组）

| 对比 | 研究问题 |
|------|----------|
| exp1 vs exp5 | **Gate 有没有用？** 全量随机 vs resolved 池随机 |
| exp5 vs exp3 | **评分有没有用？** resolved 随机 vs resolved 排序 |
| exp1→exp2, exp3→exp4, exp5→exp6 | **数据量 scaling 效果**，三种策略下 500→1000 的提升幅度 |
| exp3 vs exp7 | **Sanity check**：最好 vs 最差，验证评分体系有效性 |

#### Block 2：大维度消融（2组）

| 对比 | 研究问题 |
|------|----------|
| exp8 vs exp9 vs exp3 | **Efficiency vs Style 哪个更重要？** 单用 vs 组合 |

#### Block 3：子维度消融（4组）

| 对比 | 研究问题 |
|------|----------|
| exp10 vs exp11 vs exp3 | **Efficiency 内部**：Error-Retry Cycles vs Step Count Ratio |
| exp12 vs exp13 vs exp3 | **Style 内部**：Action Diversity vs Observation Utilization |

### 2.3 可复用的旧实验

| 实验 | 是否可复用 | 原因 |
|------|------------|------|
| baseline | ✅ 复用 | 无微调 |
| Random-500 (exp1) | ✅ 复用 | 随机采样与评分体系无关 |
| Random-1000 (exp2) | ✅ 复用 | 同上 |
| 其余所有 | ❌ 需重训 | 评分公式变化导致选出的样本不同 |

**实际需要新训练：11 组**

---

## 三、评测方案

### 3.1 困惑度评测（Perplexity / Cross-Entropy Loss）

在三个独立测试集上计算 assistant token 的平均交叉熵损失：

| 测试集 | 样本数 | 来源 |
|--------|--------|------|
| Gold | 200 | **新 composite score** 最高的轨迹 |
| Random | 200 | 随机采样 |
| Low-Q | 200 | **新 composite score** 最低的轨迹 |

> **注意**：测试集也需要根据新评分重新构建，确保 Gold/Low-Q 反映的是新体系下的质量排序。

### 3.2 预期验证

所有模型应呈现 **Gold < Random < Low-Q** 的 loss 梯度，以验证新评分体系的合理性。

---

## 四、论文故事线

```
第一层：微调本身有没有用？     baseline vs 任意微调模型
第二层：Gate 有没有用？        Random-500 vs ResolvedOnly-500
第三层：评分有没有用？         ResolvedOnly-500 vs TopQ-500
第四层：数据量 vs 质量？       500→1000 scaling curve（三种策略）
第五层：哪个大维度重要？       EfficiencyOnly vs StyleOnly vs TopQ
第六层：哪个子维度重要？       子维度消融（B2/B3/C2/C3）
验证层：评分体系有效吗？       TopQ vs BottomQ + 测试集质量梯度
```

---

## 五、设计决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|----------|------|
| Truncation Ratio 处理 | Gate（过滤） | 作为评分维度 | std≈0，无区分力 |
| Outcome Success 处理 | Gate（分层） | 作为连续评分维度 | binary 变量不宜与连续变量加权平均 |
| B1/C1 排除 | 从 composite 移除 | rank normalization 强制拉平 | rank normalization 会放大噪声，移除更诚实 |
| C3 文件匹配 | basename 匹配 | 完整路径匹配 | agent 常省略路径前缀，完整匹配导致 C3 系统性偏低 |
| 聚合方式 | 层级 mean + 等权 | 加权平均 / 学习权重 | 等权作为默认选择，权重差异可通过消融实验间接体现 |
| 数据规模 | 500 / 1000 | 2000 / 5000 | GPU 资源限制 |
