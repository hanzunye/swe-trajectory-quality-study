# Experiment A: First-Action Evaluation Report

## Summary Table

| Checkpoint | CE Loss | Type Acc | File Match | ROUGE-L | N |
|---|---|---|---|---|---|
| baseline | 0.9100 | 0.000 | 0.573 | 0.137 | 96 |
| exp1 | 0.4737 | 0.000 | 0.680 | 0.200 | 100 |
| exp3 | 0.4704 | 0.000 | 0.670 | 0.212 | 100 |
| exp2 | 0.4140 | 0.000 | 0.650 | 0.248 | 100 |

## Spearman Correlation (CE Loss ↔ Action Quality)

| Metric | Spearman ρ | p-value | Interpretation |
|---|---|---|---|
| action_type_acc | nan | nan | strong positive |
| file_match_score | -0.200 | 0.8000 | weak |
| rouge_l | -1.000 | 0.0000 | strong negative |

## Notes
- CE Loss from `eval_results/perplexity_results.csv` (gold split)
- Spearman ρ < 0 is expected: lower loss → higher action quality
- ρ < −0.7 indicates strong proxy validity
