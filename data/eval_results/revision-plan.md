# ICIC 论文修改计划（6 天）

> **截止日期：** 2026-04-01
> **论文：** A Systematic Evaluation of Trajectory Scoring for LoRA Fine-Tuning of Code Agents
> **目标：** 回应四位审稿人的核心质疑，补充关键实验，提升论文说服力

---

## 审稿人核心问题汇总

| # | 问题 | 提出者 | 严重程度 | 应对方式 |
|---|------|--------|----------|----------|
| 1 | Proxy metric（CE loss）缺乏下游验证 | R1,R2,R3,R4 | **致命** | 补实验 + 文献论证 |
| 2 | Scale 太小（仅 500–1000） | R1,R2,R3 | **高** | 补 2000 规模实验 |
| 3 | 缺少外部 baseline（IFD/DEITA） | R1,R2,R3 | **高** | Discussion 论证 |
| 4 | Composite score 是否优于 B2-only | R1,R2 | **中** | 补实验 |
| 5 | 可视化 / 统计不充分（缺 CI、boxplot） | R1,R2,R3 | **中** | 改图 + 补统计量 |
| 6 | 结论过于绝对 | R1,R2,R3 | **中** | 改措辞 |
| 7 | format_score=0.0 讨论位置不当 | R1,R2 | **低** | 调整章节 |
| 8 | 术语不统一 / ROUGE-L justification | R3 | **低** | 校对 |
| 9 | 缺框架流程图 | R4 | **低** | 画图 |
| 10 | 去 watermark | R1,R2 | **低** | 直接删除 |

---

## Day 1–2：补实验（最高优先级）

### 实验 A：First-Action Evaluation（回应问题 #1）

**目的：** 证明 CE loss 的下降确实反映了模型下游 action 质量的提升，绕开 SWE-bench resolve rate 为 0 的困境。

**原理：** 不跑完整 SWE-bench pipeline，只让模型看 issue + context，生成第一步 action，和 ground truth trajectory 的第一步对比。

#### 步骤 1：构造评估数据集

- 从 Gold / Random / Low-Q test split 中各抽取样本，总计 50–100 个 instance
- 每个 instance 提取 ground truth trajectory 的第一个 action

```python
# 伪代码
for instance in test_instances:
    prompt = instance["system_prompt"] + instance["issue_description"]
    gt_first_action = instance["trajectory"][0]["action"]
```

#### 步骤 2：定义评估指标（三个层级）

| 指标 | 含义 | 严格程度 | 示例 |
|------|------|----------|------|
| **Action Type Match** | 生成的 action 类型是否正确 | 宽松 | GT: `find_file xxx` → 模型: `find_file yyy` ✓ |
| **Target File Match** | 是否定位到正确文件/目录 | 中等 | GT: `open_file src/utils.py` → 模型: `find_file utils.py` ✓ |
| **Exact Match / ROUGE-L** | action 内容精确匹配度 | 严格 | 完全一致才算 match |

#### 步骤 3：对关键 checkpoint 做推理

选择 3–5 个有代表性的 checkpoint：

- `baseline`（未微调 / 微调前）
- `random_500`
- `topq_500`
- `random_1000`
- （可选）`topq_1000`

每个 checkpoint 推理 50–100 条，单卡 7B 模型几分钟即可完成。

```python
for checkpoint in checkpoints:
    model = load_model(checkpoint)
    for instance in test_instances:
        prompt = build_prompt(instance)  # 和训练时格式一致
        generated = model.generate(prompt, max_new_tokens=256)
        first_action = parse_first_action(generated)
        # 逐条打分
```

#### 步骤 4：相关性分析

- 将每个 checkpoint 的 CE loss 与三个 action-level 指标画 scatter plot
- 计算 Spearman 相关系数 + p-value
- 期望结果：ρ > 0.7，证明 loss 下降 → action 质量提升

```
Checkpoint     | CE Loss | Type Match% | File Match% | Exact Match%
baseline       |  2.85   |    42%      |    28%      |    15%
random_500     |  2.51   |    51%      |    35%      |    22%
topq_500       |  2.49   |    53%      |    36%      |    23%
random_1000    |  2.19   |    60%      |    44%      |    30%
```

**论文中的呈现：** 新增 Section "Proxy Metric Validation"，放在 Results 开头或作为 4.x 节，包含 scatter plot + 相关系数表。

---

### 实验 B：Scale 扩展到 2000（回应问题 #2）

**目的：** 验证 quantity-quality crossover 假设，将结论的适用范围从 500–1000 扩展到 2000。

#### 实验设计

| 实验组 | 数据量 | 筛选策略 | 目的 |
|--------|--------|----------|------|
| Random-2000 | 2000 | 随机采样 | quantity 继续扩大的效果 |
| TopQ-1000 | 1000 | composite score top 1000 | quality filtering 在 1000 时的表现 |
| TopQ-2000 | 2000 | composite score top 2000 | quality + quantity 同时扩大 |

- 最少跑 `Random-2000` vs `TopQ-1000`，这是回应审稿人最关键的一组
- 如果时间允许，加 `TopQ-2000`

**关键问题：**
- 在 2000 规模下，quality filtering 是否开始产生显著差异？
- 如果是 → 找到了 crossover point，论文结论大幅增强
- 如果否 → quantity dominance 的适用范围扩展到 2000，也是有价值的发现

---

### 实验 C：B2-only Baseline（回应问题 #4）

**目的：** 验证 composite score 是否优于单独使用 error-retry rate (B2)。

| 实验组 | 筛选指标 | 数据量 |
|--------|----------|--------|
| Composite-Top500 | 现有 composite score 选 top 500 | 500 |
| B2-Only-Top500 | 仅用 B2 (error-retry rate) 选 top 500 | 500 |

- 对比两者的 CE loss
- 如果差异不显著 → 承认 B2 单指标已足够，简化框架（这本身是一个有价值的发现）
- 如果 composite 更优 → 证明多维评分的必要性

---

## Day 3：可视化 + 统计升级（回应问题 #5）

### 需要修改的图表

- [ ] 所有关键对比图从 bar chart 改为 **boxplot 或 violin plot**
- [ ] 所有 key comparison 添加 **95% confidence interval**
- [ ] 添加 **effect size（Cohen's d）** alongside p-value
- [ ] 绘制 **框架流程图**（回应 R4），展示：
  - 数据输入 → Scoring Framework → Quality Ranking → Filtering/Sampling → LoRA Fine-tuning → Evaluation
- [ ] First-Action Evaluation 的 scatter plot（实验 A 的输出）

### 统计呈现标准

每个 key comparison 报告：
```
Mean ± SD, 95% CI [lower, upper], Cohen's d = x.xx, p = x.xxx (Mann-Whitney U)
```

---

## Day 4–5：改论文

### 结构调整

- [ ] **新增 Section：Proxy Metric Validation**（实验 A 的结果）
  - 放在 Results 开头，先建立 proxy metric 的可信度，再展开其他结果
- [ ] **format_score = 0.0 的发现提前**
  - 从 Section 5.3 移到 Results 主体或作为一个独立 Finding
  - 讨论其对实际 agent pipeline 部署的意义
- [ ] **新增 2000-scale 实验结果**（实验 B）
  - 扩展现有的 Scaling 分析 section
- [ ] **新增 B2-only baseline 对比**（实验 C）
  - 放在 Ablation Study 中

### 措辞修改

- [ ] **结论限定 scope：**
  - ~~"data quantity dominates quality"~~
  - → "In the small-to-medium scale regime (500–2000 trajectories), increasing data quantity yields substantially larger improvements than quality-ranked filtering"
- [ ] **Limitation section 加强 proxy metric 的 justification：**
  - 引用 Scaling Laws 相关论文（Kaplan et al. 2020, Hoffmann et al. 2022）中 loss 与下游任务表现的相关性证据
  - 引用 code generation 领域的类似发现（如有）
  - 明确说明：7B 模型在 SWE-bench 上 resolve rate 接近 0，端到端评估在当前规模下不可行
  - 展示 First-Action Evaluation 的结果作为 proxy validity 的直接证据

### IFD/DEITA 的论证（回应问题 #3）

在 Related Work 或 Discussion 中新增一段，核心论点：

> IFD 和 DEITA 针对单轮 instruction-response 对设计。Trajectory 数据具有本质差异：
> 1. **多步交互结构：** 一条 trajectory 包含多个 action-observation 循环，质量不是单一维度
> 2. **状态依赖：** 每步 action 的质量取决于之前的 observation，不能独立评估
> 3. **Action space 特殊性：** code agent 的 action 是结构化命令（find_file, edit_file 等），不是自然语言 response
>
> 因此，直接迁移 IFD/DEITA 到 trajectory 场景存在根本性的 mismatch。本文的 scoring framework 是针对 trajectory 结构特点设计的领域特定方案。

### 其他修改

- [ ] 术语统一：全文统一为 `TopQ` / `BottomQ`（不用 Top-Q / Bottom-Q）
- [ ] Figure 1 caption 修正：与正文描述的 metric 数量一致
- [ ] ROUGE-L 用于 action prediction 的 justification 加强（或考虑替换为 Exact Match + Action Type Match）
- [ ] 去除内部 watermark

---

## Day 6：通读 + 提交

- [ ] 全文通读，检查逻辑连贯性
- [ ] 检查所有新增实验的数据是否在文中正确引用
- [ ] 检查所有图表编号、引用是否正确
- [ ] 检查 Reference 是否有遗漏
- [ ] 确认 camera-ready 格式符合 ICIC 要求
- [ ] 提交

---

## 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|----------|
| 2000-scale 训练时间不够 | 无法回应 scale 问题 | 至少跑 1500，或在 discussion 中用外推分析论证趋势 |
| First-Action Evaluation 相关性不显著 | proxy metric 仍然存疑 | 加强文献论证（方案 B），承认 limitation 但指出这是该规模下的最优评估方式 |
| B2-only 和 composite 无显著差异 | scoring 框架受质疑 | 坦诚承认，将其作为发现呈现："practitioners can use B2 alone as a lightweight proxy" |
| 整体时间不够 | 某些修改无法完成 | 优先级：实验 A > 实验 B > 论文改写 > 实验 C > 可视化升级 |

---

## 快速 Checklist

```
[ ] 实验 A：First-Action Evaluation 代码 + 推理 + 相关性分析
[ ] 实验 B：2000-scale 训练 + 评估
[ ] 实验 C：B2-only baseline 对比
[ ] 图表：boxplot / violin plot + CI + effect size
[ ] 图表：框架流程图
[ ] 论文：新增 Proxy Metric Validation section
[ ] 论文：结论 scope 限定
[ ] 论文：IFD/DEITA discussion
[ ] 论文：format_score 发现提前
[ ] 论文：术语统一 + 校对
[ ] 论文：去 watermark
[ ] 最终通读 + 提交
```
