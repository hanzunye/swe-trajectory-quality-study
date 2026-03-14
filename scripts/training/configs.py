"""
Experiment configurations matching the Training Matrix (§3.4.3).

Each experiment maps to a QLoRA training run on a specific data subset.
"""
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str
    subset: str
    hypothesis: str
    context_length: int = 32768
    est_gpu_hours: float = 6.0

    # ── Model ──
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    # ── QLoRA ──
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # ── Training ──
    num_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # effective batch size = 8
    learning_rate: float = 2e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Memory optimization ──
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"

    # ── Checkpointing ──
    save_strategy: str = "steps"
    save_steps: int = 10
    save_total_limit: int = 4  # keep last 4 checkpoints (space management)
    logging_steps: int = 10

    # ── Flash attention ──
    attn_implementation: str = "flash_attention_2"
    #"flash_attention_2""sdpa"


# ── Experiment Matrix (§3.4.3) ────────────────────────────
# 13 experiments total:
#   exp1–exp4  : scale/quality baselines (Random, TopQ)
#   exp5–exp7  : filter-strategy comparisons (ResolvedOnly, BottomQ sanity check)
#   exp8–exp9  : group ablations (no Efficiency, no Style)
#   exp10–exp13: dimension ablations (no B2/B3/C2/C3)

EXPERIMENTS = {
    # ── Baselines ────────────────────────────────────────────
    "exp1": ExperimentConfig(
        name="exp1",
        subset="Random-500",
        hypothesis="Baseline: random 500",
        est_gpu_hours=6.0,
    ),
    "exp2": ExperimentConfig(
        name="exp2",
        subset="Random-1000",
        hypothesis="Scale baseline: does more random data help?",
        est_gpu_hours=12.0,
    ),
    "exp3": ExperimentConfig(
        name="exp3",
        subset="TopQ-500",
        hypothesis="H1: Quality filtering > random",
        est_gpu_hours=6.0,
    ),
    "exp4": ExperimentConfig(
        name="exp4",
        subset="TopQ-1000",
        hypothesis="H1 at scale: quality + more data",
        est_gpu_hours=12.0,
    ),
    # ── Filter-strategy comparisons ──────────────────────────
    "exp5": ExperimentConfig(
        name="exp5",
        subset="ResolvedOnly-500",
        hypothesis="H2: Resolved-only filter sufficient without quality ranking?",
        est_gpu_hours=6.0,
    ),
    "exp6": ExperimentConfig(
        name="exp6",
        subset="ResolvedOnly-1000",
        hypothesis="Resolved-only at scale",
        est_gpu_hours=12.0,
    ),
    "exp7": ExperimentConfig(
        name="exp7",
        subset="BottomQ-500",
        hypothesis="Sanity check: bottom quality should perform worst",
        est_gpu_hours=6.0,
    ),
    # ── Group ablations ──────────────────────────────────────
    "exp8": ExperimentConfig(
        name="exp8",
        subset="Ablation-NoEfficiency-500",
        hypothesis="Ablation: Style score only (drop Efficiency group)",
        est_gpu_hours=6.0,
    ),
    "exp9": ExperimentConfig(
        name="exp9",
        subset="Ablation-NoStyle-500",
        hypothesis="Ablation: Efficiency score only (drop Style group)",
        est_gpu_hours=6.0,
    ),
    # ── Dimension ablations ──────────────────────────────────
    "exp10": ExperimentConfig(
        name="exp10",
        subset="Ablation-NoB2-500",
        hypothesis="Ablation: remove B2 (error_retry) from Efficiency",
        est_gpu_hours=6.0,
    ),
    "exp11": ExperimentConfig(
        name="exp11",
        subset="Ablation-NoB3-500",
        hypothesis="Ablation: remove B3 (step_count_ratio) from Efficiency",
        est_gpu_hours=6.0,
    ),
    "exp12": ExperimentConfig(
        name="exp12",
        subset="Ablation-NoC2-500",
        hypothesis="Ablation: remove C2 (action_diversity) from Style",
        est_gpu_hours=6.0,
    ),
    "exp13": ExperimentConfig(
        name="exp13",
        subset="Ablation-NoC3-500",
        hypothesis="Ablation: remove C3 (observation_utilization) from Style",
        est_gpu_hours=6.0,
    ),
}


def get_experiment(name: str) -> ExperimentConfig:
    """Get experiment config by name (exp1..exp13)."""
    if name not in EXPERIMENTS:
        available = ", ".join(sorted(EXPERIMENTS.keys()))
        raise ValueError(f"Unknown experiment '{name}'. Available: {available}")
    return EXPERIMENTS[name]


def list_experiments():
    """Print all experiments."""
    print(f"{'Exp':<8} {'Subset':<35} {'Hypothesis':<40} {'~Hours'}")
    print("-" * 90)
    for name, cfg in EXPERIMENTS.items():
        print(f"{name:<8} {cfg.subset:<35} {cfg.hypothesis:<40} {cfg.est_gpu_hours:.0f}h")
