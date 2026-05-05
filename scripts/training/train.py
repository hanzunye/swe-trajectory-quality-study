"""
QLoRA Fine-tuning Script with Spot Instance Resume Support.

Trains Qwen2.5-Coder-7B-Instruct with QLoRA on trajectory subsets.
Automatically resumes from latest checkpoint if interrupted.

Data is pre-tokenized by prepare_data.py with assistant-only labels:
  - input_ids: full sequence token IDs
  - attention_mask: 1 for real tokens, 0 for padding
  - labels: input_ids for assistant tokens, -100 for everything else

This means we use the Trainer directly (not SFTTrainer's tokenization),
since the data is already in the exact format the model needs.

Usage:
    python train.py --experiment exp1
    python train.py --experiment exp3 --resume
    python train.py --experiment exp5 --epochs 3
"""
import argparse
import gc
import json
import logging
import os
import signal
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from configs import ExperimentConfig, get_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────
WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
DATA_DIR = WORKSPACE / "data"
CHECKPOINT_DIR = WORKSPACE / "checkpoints"
OUTPUT_DIR = WORKSPACE / "outputs"

# Reduce CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Graceful shutdown on SIGTERM (spot preemption) ────────

class GracefulShutdown:
    """Handle SIGTERM from spot instance preemption."""
    shutdown_requested = False

    def __init__(self):
        signal.signal(signal.SIGTERM, self._handler)
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, signum, frame):
        logger.warning("Received signal %d — saving checkpoint and shutting down...", signum)
        GracefulShutdown.shutdown_requested = True


class SpotCheckpointCallback(TrainerCallback):
    """
    Callback that monitors for shutdown signals and forces a checkpoint save.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if GracefulShutdown.shutdown_requested:
            logger.warning("Shutdown requested — forcing checkpoint save at step %d", state.global_step)
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        # Write a resume marker with step info
        marker = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "experiment": os.environ.get("CURRENT_EXPERIMENT", "unknown"),
        }
        marker_path = Path(args.output_dir) / "resume_marker.json"
        with open(marker_path, "w") as f:
            json.dump(marker, f, indent=2)
        logger.info("Resume marker saved at step %d", state.global_step)


def find_latest_checkpoint(checkpoint_base: Path) -> str | None:
    """Find the latest checkpoint directory for resume."""
    if not checkpoint_base.exists():
        return None

    checkpoints = sorted(
        [d for d in checkpoint_base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if checkpoints:
        latest = str(checkpoints[-1])
        logger.info("Found checkpoint for resume: %s", latest)
        return latest
    return None


def load_training_data(subset_name: str):
    """Load pre-tokenized Arrow dataset from prepare_data.py output."""
    data_path = DATA_DIR / subset_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            f"Run prepare_data.py first: python prepare_data.py --subset {subset_name}"
        )
    ds = load_from_disk(str(data_path))
    ds.set_format("torch")
    logger.info("Loaded %d pre-tokenized examples from %s", len(ds), data_path)

    # Quick sanity check
    sample = ds[0]
    num_unmasked = (sample["labels"] != -100).sum().item()
    total = len(sample["labels"])
    logger.info("  Sample check: %d/%d tokens have loss (%.1f%% assistant)",
                num_unmasked, total, 100 * num_unmasked / total)

    return ds


def create_model_and_tokenizer(cfg: ExperimentConfig):
    """Load base model with 4-bit quantization and tokenizer."""
    logger.info("Loading model: %s", cfg.base_model)

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # Allow TF32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Memory info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info("GPU Memory: %.2f GB allocated / %.2f GB total", allocated, total)

    return model, tokenizer


def train(cfg: ExperimentConfig, resume: bool = False):
    """Run training for a single experiment."""
    os.environ["CURRENT_EXPERIMENT"] = cfg.name

    # Paths
    exp_checkpoint_dir = CHECKPOINT_DIR / cfg.name
    exp_output_dir = OUTPUT_DIR / cfg.name
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup graceful shutdown
    shutdown_handler = GracefulShutdown()

    # Load model & tokenizer
    model, tokenizer = create_model_and_tokenizer(cfg)

    # Load pre-tokenized data (already has input_ids, attention_mask, labels)
    dataset = load_training_data(cfg.subset)

    # Check for existing checkpoint
    resume_from = None
    if resume:
        resume_from = find_latest_checkpoint(exp_checkpoint_dir)
        if resume_from:
            logger.info("Resuming from checkpoint: %s", resume_from)
        else:
            logger.info("No checkpoint found, starting from scratch.")

    # Data collator: pads to longest in batch, uses -100 for label padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=cfg.context_length,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(exp_checkpoint_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optim,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"{cfg.name}_{cfg.subset}",
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Clean memory before training
    gc.collect()
    torch.cuda.empty_cache()

    # Use standard Trainer since data is pre-tokenized with labels
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[SpotCheckpointCallback()],
    )

    # Train
    logger.info("=" * 60)
    logger.info("Starting experiment: %s", cfg.name)
    logger.info("Subset: %s (%s)", cfg.subset, cfg.hypothesis)
    logger.info("Data: %d examples, pre-tokenized with assistant-only labels", len(dataset))
    logger.info("Context length: %d", cfg.context_length)
    logger.info("Effective batch size: %d",
                cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    logger.info("Loss: assistant tokens only (labels=-100 for system/user/tool)")
    logger.info("=" * 60)

    trainer.train(resume_from_checkpoint=resume_from)

    # If SIGINT/SIGTERM was received, stop here with an interrupted exit code.
    # This prevents writing a "completed" summary and lets the runner halt cleanly.
    if GracefulShutdown.shutdown_requested:
        logger.warning("Training interrupted for %s; checkpoint saved. Exiting with code 130.", cfg.name)
        raise SystemExit(130)

    # Save final model
    final_path = exp_output_dir / "final"
    logger.info("Saving final adapter to %s", final_path)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save training summary
    summary = {
        "experiment": cfg.name,
        "subset": cfg.subset,
        "hypothesis": cfg.hypothesis,
        "base_model": cfg.base_model,
        "final_step": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
        "train_loss": trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None,
        "total_steps": trainer.state.max_steps,
        "lora_rank": cfg.lora_rank,
        "learning_rate": cfg.learning_rate,
        "context_length": cfg.context_length,
        "loss_masking": "assistant_only",
        "num_examples": len(dataset),
    }
    with open(exp_output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training complete for %s!", cfg.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning with Spot Resume")
    parser.add_argument("--experiment", required=True, help="Experiment name (exp1..exp16)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    args = parser.parse_args()

    if args.list:
        from configs import list_experiments
        list_experiments()
        return

    cfg = get_experiment(args.experiment)

    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.lr:
        cfg.learning_rate = args.lr

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
