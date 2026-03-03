#!/usr/bin/env python
"""
GPU Training Script — Breakout CNN (EfficientNetV2-S Hybrid)
=============================================================
Standalone script to train the breakout CNN on GPU with improved
hyperparameters tuned for the ~3k+ image dataset on an RTX 2070 Super.

Usage:
    cd futures
    PYTHONPATH=src .venv/bin/python scripts/train_gpu.py

    # Or with custom options:
    PYTHONPATH=src .venv/bin/python scripts/train_gpu.py \
        --epochs 25 --batch-size 64 --lr 2e-4 --freeze-epochs 3

Key improvements over the default CLI trainer:
  - Mixed precision (AMP) for ~2x speedup on RTX cards
  - Larger batch size (64) to better utilise 8GB VRAM
  - More epochs (25) with cosine annealing + warmup
  - Class-weighted loss to handle good/bad imbalance (~65/35)
  - Progress bars via tqdm
  - Saves best model by val accuracy AND by val loss
  - Prints a summary table at the end
  - Early stopping if val loss doesn't improve for N epochs
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from lib.analysis.breakout_cnn import (
    DEFAULT_MODEL_DIR,
    MODEL_PREFIX,
    BreakoutDataset,
    HybridBreakoutCNN,
    get_device,
    get_inference_transform,
    get_training_transform,
)

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_gpu")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    logger.info("tqdm not installed — using plain logging (pip install tqdm for progress bars)")


def compute_class_weights(csv_path: str) -> torch.Tensor:
    """Compute inverse-frequency class weights for the binary good/bad split."""
    df = pd.read_csv(csv_path)
    labels = df["label"].apply(lambda x: 1 if str(x).startswith("good") else 0)
    counts = labels.value_counts().sort_index()  # 0=bad, 1=good
    total = len(labels)
    weights = []
    for cls in range(2):
        c = counts.get(cls, 1)
        weights.append(total / (2.0 * c))
    w = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights: bad=%.3f, good=%.3f (from %d samples)", w[0], w[1], total)
    return w


def train(
    train_csv: str = "dataset/train.csv",
    val_csv: str = "dataset/val.csv",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    freeze_epochs: int = 3,
    warmup_epochs: int = 1,
    model_dir: str = DEFAULT_MODEL_DIR,
    num_workers: int = 4,
    patience: int = 8,
    label_smoothing: float = 0.05,
    use_amp: bool = True,
    dropout: float = 0.4,
) -> str | None:
    """Train the HybridBreakoutCNN with GPU-optimised settings.

    Returns the path to the best saved model, or None on failure.
    """
    # --- Device ---
    device_str = get_device()
    device = torch.device(device_str)
    logger.info("=" * 70)
    logger.info("BREAKOUT CNN — GPU TRAINING")
    logger.info("=" * 70)
    logger.info("Device:          %s", device)
    if device.type == "cuda":
        logger.info("GPU:             %s", torch.cuda.get_device_name(0))
        logger.info("VRAM:            %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
        logger.info("PyTorch:         %s", torch.__version__)
        logger.info("CUDA:            %s", torch.version.cuda)
    logger.info("Epochs:          %d (freeze: %d, warmup: %d)", epochs, freeze_epochs, warmup_epochs)
    logger.info("Batch size:      %d", batch_size)
    logger.info("Learning rate:   %.1e", lr)
    logger.info("Weight decay:    %.1e", weight_decay)
    logger.info("Label smoothing: %.2f", label_smoothing)
    logger.info("Dropout:         %.2f", dropout)
    logger.info("Mixed precision: %s", "ON" if use_amp else "OFF")
    logger.info("Early stopping:  patience=%d", patience)
    logger.info("-" * 70)

    # Disable AMP if not on CUDA
    if device.type != "cuda":
        use_amp = False

    # --- Datasets ---
    train_transform = get_training_transform()
    val_transform = get_inference_transform()

    train_dataset = BreakoutDataset(train_csv, transform=train_transform)
    val_dataset = BreakoutDataset(val_csv, transform=val_transform)

    logger.info("Train samples:   %d", len(train_dataset))
    logger.info("Val samples:     %d", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )

    # --- Model ---
    model = HybridBreakoutCNN(pretrained=True, dropout=dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters:      %.1fM total", total_params / 1e6)
    logger.info("                 %.1fM trainable", trainable_params / 1e6)
    logger.info("-" * 70)

    # --- Class-weighted loss ---
    class_weights = compute_class_weights(train_csv).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    # --- Optimizer: separate LR groups for backbone vs head ---
    backbone_params = list(model.cnn.parameters())
    head_params = list(model.tabular_head.parameters()) + list(model.classifier.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1, "name": "backbone"},
            {"params": head_params, "lr": lr, "name": "head"},
        ],
        weight_decay=weight_decay,
    )

    # --- Scheduler: cosine annealing with warmup ---
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Mixed precision scaler ---
    scaler = GradScaler("cuda", enabled=use_amp)

    # --- Training state ---
    os.makedirs(model_dir, exist_ok=True)
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_acc_path: str | None = None
    best_loss_path: str | None = None
    epochs_without_improvement = 0
    history: list[dict] = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Phase management ---
        if epoch < freeze_epochs:
            if epoch == 0:
                model.freeze_backbone()
                # Zero out backbone LR during freeze
                optimizer.param_groups[0]["lr"] = 0.0
            phase = "frozen"
        elif epoch == freeze_epochs:
            model.unfreeze_backbone()
            # Restore backbone LR
            optimizer.param_groups[0]["lr"] = lr * 0.1 * lr_lambda(epoch)
            phase = "fine-tune"
        else:
            phase = "fine-tune"

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loader_iter = train_loader
        if HAS_TQDM:
            loader_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1:2d}/{epochs} [{phase:9s}] Train",
                leave=False,
                ncols=100,
            )

        for batch_idx, batch in enumerate(loader_iter):
            # skip_invalid_collate returns None when every sample was invalid
            if batch is None:
                continue
            imgs, tabs, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                outputs = model(imgs, tabs)
                loss = criterion(outputs, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss at batch %d — skipping", batch_idx)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            if HAS_TQDM:
                loader_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100.0 * train_correct / max(train_total, 1):.1f}%",
                )

        scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # true positives (good predicted as good)
        val_fp = 0  # false positives (bad predicted as good)
        val_fn = 0  # false negatives (good predicted as bad)

        loader_iter_val = val_loader
        if HAS_TQDM:
            loader_iter_val = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1:2d}/{epochs} [{phase:9s}]   Val",
                leave=False,
                ncols=100,
            )

        with torch.no_grad():
            for batch in loader_iter_val:
                if batch is None:
                    continue
                imgs, tabs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast("cuda", enabled=use_amp):
                    outputs = model(imgs, tabs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                # Precision/recall for class 1 (good)
                val_tp += ((predicted == 1) & (labels == 1)).sum().item()
                val_fp += ((predicted == 1) & (labels == 0)).sum().item()
                val_fn += ((predicted == 0) & (labels == 1)).sum().item()

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)

        precision = val_tp / max(val_tp + val_fp, 1) * 100
        recall = val_tp / max(val_tp + val_fn, 1) * 100

        epoch_time = time.time() - epoch_start
        backbone_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]

        # Log
        logger.info(
            "Epoch %2d/%d [%s] — "
            "Train: loss=%.4f acc=%.1f%% | "
            "Val: loss=%.4f acc=%.1f%% prec=%.1f%% rec=%.1f%% | "
            "LR: %.1e/%.1e | %.0fs",
            epoch + 1,
            epochs,
            phase,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            precision,
            recall,
            backbone_lr,
            head_lr,
            epoch_time,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "phase": phase,
                "train_loss": round(avg_train_loss, 4),
                "train_acc": round(train_acc, 1),
                "val_loss": round(avg_val_loss, 4),
                "val_acc": round(val_acc, 1),
                "precision": round(precision, 1),
                "recall": round(recall, 1),
                "backbone_lr": backbone_lr,
                "head_lr": head_lr,
                "epoch_time": round(epoch_time, 1),
            }
        )

        # --- Save best by accuracy ---
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_acc_path = os.path.join(
                model_dir,
                f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_acc{val_acc:.0f}.pt",
            )
            torch.save(model.state_dict(), best_acc_path)
            logger.info("  ★ New best accuracy: %.1f%% → %s", val_acc, best_acc_path)
            improved = True

        # --- Save best by loss ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_path = os.path.join(
                model_dir,
                f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_bestloss.pt",
            )
            torch.save(model.state_dict(), best_loss_path)
            if not improved:
                logger.info("  ↓ New best val loss: %.4f → %s", avg_val_loss, best_loss_path)
            improved = True

        # --- Early stopping ---
        if improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping — no improvement for %d epochs (best acc: %.1f%%)",
                    patience,
                    best_val_acc,
                )
                break

    # --- Save final model ---
    final_path = os.path.join(model_dir, f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_final.pt")
    torch.save(model.state_dict(), final_path)

    # --- Also save as breakout_cnn_best.pt for easy inference ---
    if best_acc_path:
        best_link = os.path.join(model_dir, "breakout_cnn_best.pt")
        import shutil

        shutil.copy2(best_acc_path, best_link)
        logger.info("Best model copied to: %s", best_link)

    total_time = time.time() - start_time

    # --- Print summary ---
    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Total time:      %.1f min", total_time / 60)
    logger.info("Best val acc:    %.1f%%", best_val_acc)
    logger.info("Best val loss:   %.4f", best_val_loss)
    logger.info("Best acc model:  %s", best_acc_path or "N/A")
    logger.info("Best loss model: %s", best_loss_path or "N/A")
    logger.info("Final model:     %s", final_path)
    logger.info("")

    # Print training history table
    print("\n" + "=" * 110)
    print(
        f"{'Epoch':>5} | {'Phase':>9} | {'Train Loss':>10} | {'Train Acc':>9} | "
        f"{'Val Loss':>8} | {'Val Acc':>7} | {'Prec':>5} | {'Rec':>5} | {'Time':>5}"
    )
    print("-" * 110)
    for h in history:
        marker = " ★" if h["val_acc"] == best_val_acc else ""
        print(
            f"{h['epoch']:>5} | {h['phase']:>9} | {h['train_loss']:>10.4f} | {h['train_acc']:>8.1f}% | "
            f"{h['val_loss']:>8.4f} | {h['val_acc']:>6.1f}% | {h['precision']:>4.1f}% | "
            f"{h['recall']:>4.1f}% | {h['epoch_time']:>4.0f}s{marker}"
        )
    print("=" * 110)

    # Save history CSV for analysis
    history_path = os.path.join(model_dir, "training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    logger.info("Training history saved to: %s", history_path)

    return best_acc_path


def main():
    parser = argparse.ArgumentParser(
        description="GPU Training — Breakout CNN (EfficientNetV2-S Hybrid)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv", default="dataset/train.csv", help="Training CSV path")
    parser.add_argument("--val-csv", default="dataset/val.csv", help="Validation CSV path")
    parser.add_argument("--epochs", type=int, default=25, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (64 fits 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for head")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Epochs with frozen backbone")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="LR warmup epochs")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--dropout", type=float, default=0.4, help="Classifier dropout")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Model output directory")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.train_csv):
        logger.error("Training CSV not found: %s", args.train_csv)
        sys.exit(1)
    if not os.path.isfile(args.val_csv):
        logger.error("Validation CSV not found: %s", args.val_csv)
        sys.exit(1)

    result = train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_epochs=args.freeze_epochs,
        warmup_epochs=args.warmup_epochs,
        model_dir=args.model_dir,
        num_workers=args.workers,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        use_amp=not args.no_amp,
        dropout=args.dropout,
    )

    if result:
        print(f"\n✅ Best model saved to: {result}")
    else:
        print("\n❌ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
