# Statistical Significance Report (v3)

## All Models vs Baseline (Gold Test Set, two-sided)

| Model | Direction | U-stat | p-value | Significant | Effect Size r |
|-------|-----------|--------|---------|-------------|---------------|
| Random 500 | ↓ better | 39875 | 0.0000 | **YES** | 0.997 |
| Random 1000 | ↓ better | 39978 | 0.0000 | **YES** | 0.999 |
| TopQ 500 | ↓ better | 39894 | 0.0000 | **YES** | 0.997 |
| TopQ 1000 | ↓ better | 39982 | 0.0000 | **YES** | 1.000 |
| ResolvedOnly 500 | ↓ better | 39875 | 0.0000 | **YES** | 0.997 |
| ResolvedOnly 1000 | ↓ better | 39977 | 0.0000 | **YES** | 0.999 |
| BottomQ 500 | ↓ better | 39863 | 0.0000 | **YES** | 0.997 |
| Ablation NoEfficiency | ↓ better | 39882 | 0.0000 | **YES** | 0.997 |
| Ablation NoStyle | ↓ better | 39890 | 0.0000 | **YES** | 0.997 |
| Ablation NoB2 | ↓ better | 39882 | 0.0000 | **YES** | 0.997 |
| Ablation NoB3 | ↓ better | 39890 | 0.0000 | **YES** | 0.997 |
| Ablation NoC2 | ↓ better | 39891 | 0.0000 | **YES** | 0.997 |
| Ablation NoC3 | ↓ better | 39893 | 0.0000 | **YES** | 0.997 |

---
## Block 1: Gate & Score Effect

### H1: Gate 有效性 — ResolvedOnly-500 < Random-500 (Gold set)

- Mann-Whitney U=20905, p=0.7832 → Not supported ✗
- Mean loss: ResolvedOnly-500=0.4786, Random-500=0.4737

### H2: 评分有效性 — TopQ-500 < ResolvedOnly-500 (Gold set)

- Mann-Whitney U=18551, p=0.1051 → Not supported ✗
- Mean loss: TopQ-500=0.4704, ResolvedOnly-500=0.4786

### H4: Sanity Check — TopQ-500 < BottomQ-500 (Gold set)

- Mann-Whitney U=17139, p=0.0067 → **Supported** ✓
- Mean loss: TopQ-500=0.4704, BottomQ-500=0.4869

### H3: Scaling — 500 → 1000 均值对比

| Strategy | 500 mean loss | 1000 mean loss | Δ |
|----------|--------------|----------------|---|
| Random | 0.4737 | 0.4140 | -0.0597 |
| TopQ | 0.4704 | 0.4106 | -0.0598 |
| ResolvedOnly | 0.4786 | 0.4175 | -0.0610 |

---
## Block 2: Efficiency vs Style

### H5: Efficiency 贡献 > Style 贡献 — NoEfficiency > NoStyle on Gold set

- Mann-Whitney U=21177, p=0.1544 → Not supported ✗
- Mean loss: NoEfficiency=0.4774, NoStyle=0.4711

---
## Block 3: Sub-dimension Ablation

### H6: B2(error_retry) 贡献 > B3(step_count) — NoB2 > NoB3 on Gold set

- Mann-Whitney U=20572, p=0.3105 → Not supported ✗
- Mean loss: NoB2=0.4749, NoB3=0.4720

### H7: C3(obs_utilization) 贡献 > C2(action_diversity) — NoC3 > NoC2 on Gold set

- Mann-Whitney U=19331, p=0.7187 → Not supported ✗
- Mean loss: NoC3=0.4700, NoC2=0.4735

### 子维度 Impact 排名（Δ vs TopQ-500，gold set）

| Ablation | Mean Loss | Δ vs TopQ | Rank |
|----------|-----------|-----------|------|
| No Efficiency | 0.4774 | +0.0070 | #1 |
| No B2 (error_retry) | 0.4749 | +0.0045 | #2 |
| No C2 (action_div) | 0.4735 | +0.0031 | #3 |
| No B3 (step_count) | 0.4720 | +0.0016 | #4 |
| No Style | 0.4711 | +0.0007 | #5 |
| No C3 (obs_util) | 0.4700 | -0.0004 | #6 |

**最关键维度**: 移除 **No Efficiency** 导致 loss 增幅最大。

---
## H8: 测试集质量梯度（所有模型应满足 gold < random < low_q）

| Model | Gold | Random | Low-Q | H8 Satisfied |
|-------|------|--------|-------|--------------|
| Baseline (no SFT) | 0.9100 | 0.9661 | 1.0812 | ✓ |
| Random 500 | 0.4737 | 0.4875 | 0.5232 | ✓ |
| TopQ 500 | 0.4704 | 0.4878 | 0.5260 | ✓ |
| ResolvedOnly 500 | 0.4786 | 0.4886 | 0.5212 | ✓ |
| BottomQ 500 | 0.4869 | 0.4916 | 0.5185 | ✓ |