"""
Breakout CNN — Hybrid EfficientNetV2 + Tabular Model for ORB Pattern Recognition
==================================================================================
Classifies Ruby-style chart snapshots as "good breakout" (high-probability
follow-through) or "bad breakout" (likely to fail / chop) using a hybrid
architecture that combines:

  1. **EfficientNetV2-S** (image backbone) — pre-trained on ImageNet, fine-tuned
     on Ruby-rendered candlestick charts.  Extracts 1280-dim visual features from
     the exact same chart images a human trader sees (ORB box, VWAP, EMA9,
     quality badge, volume panel).

  2. **Tabular head** — a small feed-forward network that ingests numeric features
     the image cannot easily capture (quality %, volume ratio, ATR %, CVD delta,
     NR7 flag, direction bias).

  3. **Classifier** — merges the two feature vectors and outputs a probability
     of "clean breakout" (0.0–1.0).

Dependencies:
  - torch >= 2.0
  - torchvision >= 0.15
  - Pillow
  - pandas, numpy (already in project)

All torch imports are guarded so the module can be imported on machines
without CUDA (it will log a warning and the functions will return None /
raise informative errors).

Public API:
    from lib.analysis.breakout_cnn import (
        HybridBreakoutCNN,
        BreakoutDataset,
        train_model,
        predict_breakout,
        predict_breakout_batch,
        get_device,
        DEFAULT_THRESHOLD,
        TABULAR_FEATURES,
    )

Design:
  - Training produces a ``.pt`` state-dict file under ``models/``.
  - Inference loads the latest model automatically (or a specific path).
  - Thread-safe: model loading uses a module-level lock.
  - Graceful degradation: if torch is missing, all functions return None
    with a logged warning — the rest of the engine keeps running.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    # Always visible to the type checker — these are never actually
    # executed at runtime when TYPE_CHECKING is True.
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

logger = logging.getLogger("analysis.breakout_cnn")

# ---------------------------------------------------------------------------
# Guard torch imports — allow the module to be imported without GPU/torch
# ---------------------------------------------------------------------------

try:
    import torch  # type: ignore[no-redef]
    import torch.nn as nn  # type: ignore[no-redef]
    import torchvision.models as models  # type: ignore[no-redef]
    import torchvision.transforms as T  # type: ignore[no-redef]
    from PIL import Image  # type: ignore[no-redef]
    from torch.utils.data import DataLoader, Dataset  # type: ignore[no-redef]

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch / torchvision not installed — CNN features disabled.  "
        "Install with: pip install torch torchvision Pillow"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered list of tabular feature names expected by the model.
# The dataset and inference code must provide them in exactly this order.
TABULAR_FEATURES: list[str] = [
    "quality_pct",  # 0–100, normalised to 0–1
    "volume_ratio",  # breakout bar vol / 20-bar avg vol
    "atr_pct",  # ATR as % of price
    "cvd_delta",  # cumulative volume delta (normalised)
    "nr7_flag",  # 1.0 if NR7 day, else 0.0
    "direction_flag",  # 1.0 = LONG, 0.0 = SHORT
]

NUM_TABULAR = len(TABULAR_FEATURES)

# Image pre-processing — matches ImageNet stats used by EfficientNetV2
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference threshold — probability above this → "send signal"
DEFAULT_THRESHOLD = 0.82

# Model output directory
DEFAULT_MODEL_DIR = "models"
MODEL_PREFIX = "breakout_cnn_"

# Thread lock for model loading
_model_lock = threading.Lock()
_cached_model: Any | None = None
_cached_model_path: str | None = None


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_inference_transform():
    """Standard image transform for inference (resize + normalise)."""
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_training_transform():
    """Image transform for training with mild augmentation.

    Augmentations are intentionally light — we want the CNN to learn from
    the chart structure (candlestick bodies, ORB box, overlays), not from
    random crops or heavy colour jitter that would destroy that information.
    """
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
            T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.0),  # disabled — chart direction matters
            T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def get_device() -> str:
    """Return the best available device string ('cuda', 'mps', or 'cpu')."""
    if not _TORCH_AVAILABLE:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class BreakoutDataset(Dataset):  # type: ignore[no-redef]
        """PyTorch Dataset for breakout chart images + tabular features.

        Expects a CSV with columns:
          - image_path: path to the PNG chart snapshot
          - label: "good_long", "good_short", "bad_long", "bad_short"
          - quality_pct, volume_ratio, atr_pct, cvd_delta, nr7_flag, direction
            (the tabular features)

        The binary target is:
          - 1 if label starts with "good_" (clean breakout)
          - 0 otherwise (failed breakout)
        """

        def __init__(
            self,
            csv_path: str,
            transform=None,
            image_root: str | None = None,
        ):
            self.df = pd.read_csv(csv_path)
            self.transform = transform or get_inference_transform()
            self.image_root = image_root

            # Pre-validate: drop rows with missing image paths
            self.df = self.df.dropna(subset=["image_path"])
            if "label" not in self.df.columns:
                raise ValueError("CSV must have a 'label' column")

            logger.info("BreakoutDataset loaded: %d samples from %s", len(self.df), csv_path)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]

            # --- Image ---
            img_path = row["image_path"]
            if self.image_root and not os.path.isabs(img_path):
                img_path = os.path.join(self.image_root, img_path)

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                logger.warning("Failed to load image %s: %s — using blank", img_path, exc)
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(15, 15, 26))

            if self.transform:
                img = self.transform(img)

            # --- Tabular features (normalised to ~0–1 range) ---
            # volume_ratio: raw can be 0–30+; log-scale then clamp to [0, 1]
            _vol_raw = max(float(row.get("volume_ratio", 1.0)), 0.01)
            _vol_norm = min(np.log1p(_vol_raw) / np.log1p(10.0), 1.0)  # log1p(10)≈2.4

            # atr_pct: raw is typically 0.0001–0.01; scale ×100 then clamp
            _atr_norm = min(float(row.get("atr_pct", 0.0)) * 100.0, 1.0)

            # cvd_delta: clamp to [-1, 1]
            _cvd_raw = float(row.get("cvd_delta", 0.0))
            _cvd_norm = max(min(_cvd_raw, 1.0), -1.0)

            tabular = torch.tensor(
                [
                    float(row.get("quality_pct", 0)) / 100.0,  # already 0–1
                    _vol_norm,
                    _atr_norm,
                    _cvd_norm,
                    float(row.get("nr7_flag", 0)),  # 0 or 1
                    1.0 if str(row.get("direction", "")).upper().startswith("L") else 0.0,
                ],
                dtype=torch.float32,
            )

            # Guard against NaN / Inf from corrupt data
            if torch.isnan(tabular).any() or torch.isinf(tabular).any():
                logger.warning("NaN/Inf in tabular features for row %d — zeroing", idx)
                tabular = torch.zeros(NUM_TABULAR, dtype=torch.float32)

            # --- Label ---
            label_str = str(row.get("label", "bad"))
            target = 1 if label_str.startswith("good") else 0

            return img, tabular, torch.tensor(target, dtype=torch.long)

else:
    # Stub when torch is not available
    class BreakoutDataset(Dataset):  # type: ignore[no-redef,misc]
        """Stub for environments without PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed — cannot create BreakoutDataset")

        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> Any:
            raise RuntimeError("PyTorch is not installed")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class HybridBreakoutCNN(nn.Module):  # type: ignore[no-redef]
        """Hybrid image + tabular model for breakout classification.

        Architecture:
          Image branch:  EfficientNetV2-S (pre-trained) → 1280-dim features
          Tabular branch: Linear(6→64) → ReLU → Dropout → Linear(64→32)
          Classifier:     Linear(1280+32→256) → ReLU → Dropout → Linear(256→2)

        The model outputs raw logits for 2 classes:
          - Class 0: bad breakout (fail / chop)
          - Class 1: good breakout (clean follow-through)

        Use ``torch.softmax(output, dim=1)[:, 1]`` to get P(good breakout).
        """

        def __init__(
            self,
            num_tabular: int = NUM_TABULAR,
            dropout: float = 0.4,
            pretrained: bool = True,
            freeze_backbone_epochs: int = 0,
        ):
            super().__init__()
            self.num_tabular = num_tabular
            self._freeze_backbone_epochs = freeze_backbone_epochs

            # --- Image backbone: EfficientNetV2-S ---
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)

            # Remove the original classifier head → get 1280-dim features
            # EfficientNetV2-S: features → avgpool → classifier
            # We keep features + avgpool, replace classifier with Identity
            backbone.classifier = nn.Identity()  # type: ignore[assignment]
            self.cnn = backbone
            self._cnn_out_dim = 1280  # EfficientNetV2-S feature dim

            # --- Tabular branch ---
            self.tabular_head = nn.Sequential(
                nn.Linear(num_tabular, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
            )

            # --- Classifier (merges image + tabular) ---
            combined_dim = self._cnn_out_dim + 32
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 2),
            )

        def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                image: (B, 3, 224, 224) normalised image tensor.
                tabular: (B, NUM_TABULAR) float tensor.

            Returns:
                (B, 2) logits tensor.
            """
            img_features = self.cnn(image)  # (B, 1280)
            tab_features = self.tabular_head(tabular)  # (B, 32)
            combined = torch.cat([img_features, tab_features], dim=1)  # (B, 1312)
            return self.classifier(combined)  # (B, 2)

        def freeze_backbone(self):
            """Freeze the CNN backbone (useful for first N epochs of fine-tuning)."""
            for param in self.cnn.parameters():
                param.requires_grad = False
            logger.info("CNN backbone frozen")

        def unfreeze_backbone(self):
            """Unfreeze the CNN backbone for full fine-tuning."""
            for param in self.cnn.parameters():
                param.requires_grad = True
            logger.info("CNN backbone unfrozen")

else:

    class HybridBreakoutCNN(nn.Module):  # type: ignore[no-redef,misc]
        """Stub for environments without PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed — cannot create HybridBreakoutCNN")

        def freeze_backbone(self) -> None:
            raise RuntimeError("PyTorch is not installed")

        def unfreeze_backbone(self) -> None:
            raise RuntimeError("PyTorch is not installed")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    data_csv: str,
    val_csv: str | None = None,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    freeze_epochs: int = 2,
    model_dir: str = DEFAULT_MODEL_DIR,
    image_root: str | None = None,
    num_workers: int = 4,
    save_best: bool = True,
) -> str | None:
    """Train the HybridBreakoutCNN model.

    Two-phase training:
      1. Freeze CNN backbone for ``freeze_epochs`` epochs — trains only the
         tabular head and classifier on your data.
      2. Unfreeze backbone and fine-tune everything at a lower LR.

    Args:
        data_csv: Path to training CSV (see BreakoutDataset for format).
        val_csv: Optional validation CSV.  If None, 15% of training data
                 is held out automatically.
        epochs: Total training epochs (default 8).
        batch_size: Batch size (default 32).
        lr: Learning rate (default 3e-4).
        weight_decay: AdamW weight decay (default 1e-5).
        freeze_epochs: Number of epochs to freeze the CNN backbone (default 2).
        model_dir: Directory to save the trained model (default "models").
        image_root: Optional root directory to prepend to image_path values.
        num_workers: DataLoader workers (default 4).
        save_best: If True and val_csv is provided, save the best model by
                   validation accuracy instead of the final epoch.

    Returns:
        Path to the saved model file, or None if training failed.
    """
    if not _TORCH_AVAILABLE:
        logger.error("PyTorch is not installed — cannot train model")
        return None

    device = torch.device(get_device())
    logger.info("Training on device: %s", device)

    # --- Datasets ---
    train_transform = get_training_transform()
    val_transform = get_inference_transform()

    train_dataset = BreakoutDataset(data_csv, transform=train_transform, image_root=image_root)

    if val_csv:
        val_dataset = BreakoutDataset(val_csv, transform=val_transform, image_root=image_root)
    else:
        # Auto-split: 85% train / 15% val
        n = len(train_dataset)
        n_val = max(1, int(n * 0.15))
        n_train = n - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [n_train, n_val])
        logger.info("Auto-split: %d train / %d val", n_train, n_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Model ---
    model = HybridBreakoutCNN(pretrained=True)
    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # --- Loss ---
    # Use label smoothing to prevent overconfident predictions
    criterion: Any = nn.CrossEntropyLoss(label_smoothing=0.05)

    # --- Training loop ---
    best_val_acc = 0.0
    best_model_path: str | None = None
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        # Phase management: freeze/unfreeze backbone
        if epoch < freeze_epochs:
            if epoch == 0:
                model.freeze_backbone()
        elif epoch == freeze_epochs:
            model.unfreeze_backbone()
            # Lower LR for backbone fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = lr * 0.1

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, tabs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs, tabs)
            loss = criterion(outputs, labels)

            # NaN guard — skip batch if loss explodes
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss at batch %d — skipping", batch_idx)
                continue

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1) * 100

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, tabs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(imgs, tabs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1) * 100

        current_lr = optimizer.param_groups[0]["lr"]
        phase = "frozen" if epoch < freeze_epochs else "fine-tune"

        logger.info(
            "Epoch %d/%d [%s] — Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%% | LR: %.2e",
            epoch + 1,
            epochs,
            phase,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
            current_lr,
        )

        # --- Save best model ---
        if save_best and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(
                model_dir, f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_acc{val_acc:.0f}.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            logger.info("New best model saved: %s (val_acc=%.1f%%)", best_model_path, val_acc)

    # --- Save final model (if not saving best, or as fallback) ---
    final_path = os.path.join(model_dir, f"{MODEL_PREFIX}{datetime.now():%Y%m%d_%H%M%S}_final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved: %s", final_path)

    result_path = best_model_path if (save_best and best_model_path) else final_path
    logger.info(
        "Training complete — best val accuracy: %.1f%% — model: %s",
        best_val_acc,
        result_path,
    )

    # Invalidate cached model so next inference picks up the new one
    global _cached_model, _cached_model_path
    with _model_lock:
        _cached_model = None
        _cached_model_path = None

    return result_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_latest_model(model_dir: str = DEFAULT_MODEL_DIR) -> str | None:
    """Find the most recently modified .pt file in the model directory."""
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return None

    pt_files = list(model_path.glob(f"{MODEL_PREFIX}*.pt"))
    if not pt_files:
        return None

    # Sort by modification time, newest first
    pt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(pt_files[0])


def _load_model(
    model_path: str | None = None,
    device: str | None = None,
) -> Any | None:
    """Load a HybridBreakoutCNN model from disk.

    Uses a module-level cache to avoid reloading on every inference call.
    Thread-safe via _model_lock.

    Args:
        model_path: Explicit path to a .pt file.  If None, finds the latest.
        device: Device to load onto.  If None, auto-detects.

    Returns:
        The loaded model in eval mode, or None if loading failed.
    """
    if not _TORCH_AVAILABLE:
        return None

    global _cached_model, _cached_model_path

    if model_path is None:
        model_path = _find_latest_model()

    if model_path is None:
        logger.warning("No trained model found in %s", DEFAULT_MODEL_DIR)
        return None

    with _model_lock:
        # Return cached model if same path
        if _cached_model is not None and _cached_model_path == model_path:
            return _cached_model

        dev = torch.device(device or get_device())

        try:
            model = HybridBreakoutCNN(pretrained=False)
            state_dict = torch.load(model_path, map_location=dev, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            model = model.to(dev)

            _cached_model = model
            _cached_model_path = model_path

            logger.info("Model loaded: %s → %s", model_path, dev)
            return model

        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_path, exc)
            return None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _normalise_tabular_for_inference(raw_features: Sequence[float]) -> list[float]:
    """Apply the same normalisation used in BreakoutDataset to raw inference features.

    Input order: [quality_pct_norm, volume_ratio, atr_pct, cvd_delta, nr7_flag, direction_flag]
    The caller is expected to pass quality_pct already divided by 100.

    Returns a list of 6 floats in normalised form.
    """
    f = list(raw_features)
    if len(f) != NUM_TABULAR:
        return f  # pass through — caller will get an error downstream

    quality_norm = max(min(f[0], 1.0), 0.0)

    # volume_ratio: log-scale, same as dataset
    vol_raw = max(f[1], 0.01)
    vol_norm = min(float(np.log1p(vol_raw) / np.log1p(10.0)), 1.0)

    # atr_pct: ×100 then clamp
    atr_norm = min(f[2] * 100.0, 1.0)

    # cvd_delta: clamp [-1, 1]
    cvd_norm = max(min(f[3], 1.0), -1.0)

    return [quality_norm, vol_norm, atr_norm, cvd_norm, f[4], f[5]]


def predict_breakout(
    image_path: str,
    tabular_features: Sequence[float],
    model_path: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any] | None:
    """Predict whether a chart snapshot shows a high-quality breakout.

    Args:
        image_path: Path to the PNG chart snapshot.
        tabular_features: List/tuple of 6 floats in TABULAR_FEATURES order:
            [quality_pct_norm, volume_ratio, atr_pct, cvd_delta, nr7_flag, direction_flag]
            - quality_pct_norm: quality_pct / 100 (0.0–1.0)
            - volume_ratio: breakout bar volume / 20-bar average
            - atr_pct: ATR as fraction of price
            - cvd_delta: normalised CVD delta
            - nr7_flag: 1.0 if NR7 day, 0.0 otherwise
            - direction_flag: 1.0 for LONG, 0.0 for SHORT
        model_path: Explicit model path (default: latest in models/).
        threshold: Probability threshold for "signal" verdict (default 0.82).

    Returns:
        Dict with:
          - prob: float (0.0–1.0) — probability of clean breakout
          - signal: bool — True if prob >= threshold
          - confidence: str — "high", "medium", or "low"
          - model_path: str — which model was used
        Or None if inference failed.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — cannot run inference")
        return None

    model = _load_model(model_path)
    if model is None:
        return None

    device = next(model.parameters()).device
    transform = get_inference_transform()

    try:
        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        assert transform is not None, "Transform must not be None"
        img_tensor = transform(img).unsqueeze(0).to(device)  # type: ignore[union-attr]  # (1, 3, 224, 224)

        # Normalise tabular features (same transforms as training dataset)
        tab_list = _normalise_tabular_for_inference(tabular_features)
        if len(tab_list) != NUM_TABULAR:
            logger.error("Expected %d tabular features, got %d", NUM_TABULAR, len(tab_list))
            return None

        tab_tensor = torch.tensor([tab_list], dtype=torch.float32).to(device)  # (1, 6)

        # Inference
        with torch.no_grad():
            logits = model(img_tensor, tab_tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=1)
            prob_good = float(probs[0, 1].item())

        # Confidence bucketing
        if prob_good >= 0.90:
            confidence = "high"
        elif prob_good >= 0.75:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "prob": round(prob_good, 4),
            "signal": prob_good >= threshold,
            "confidence": confidence,
            "threshold": threshold,
            "model_path": _cached_model_path or "",
        }

    except Exception as exc:
        logger.error("Inference failed for %s: %s", image_path, exc, exc_info=True)
        return None


def predict_breakout_batch(
    image_paths: Sequence[str],
    tabular_features_batch: Sequence[Sequence[float]],
    model_path: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = 16,
) -> list[dict[str, Any] | None]:
    """Batch inference for multiple chart snapshots.

    More efficient than calling ``predict_breakout`` in a loop because it
    batches the GPU forward passes.

    Args:
        image_paths: List of PNG paths.
        tabular_features_batch: List of tabular feature vectors (one per image).
        model_path: Explicit model path (default: latest).
        threshold: Signal threshold.
        batch_size: Max images per GPU forward pass.

    Returns:
        List of result dicts (same format as predict_breakout), or None entries
        for images that failed to load.
    """
    if not _TORCH_AVAILABLE:
        return [None] * len(image_paths)

    if len(image_paths) != len(tabular_features_batch):
        logger.error("Mismatched lengths: %d images vs %d tabular", len(image_paths), len(tabular_features_batch))
        return [None] * len(image_paths)

    model = _load_model(model_path)
    if model is None:
        return [None] * len(image_paths)

    device = next(model.parameters()).device
    transform = get_inference_transform()
    results: list[dict[str, Any] | None] = [None] * len(image_paths)

    # Process in batches
    for batch_start in range(0, len(image_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_indices = list(range(batch_start, batch_end))

        img_tensors = []
        tab_tensors = []
        valid_indices = []

        for i in batch_indices:
            try:
                img = Image.open(image_paths[i]).convert("RGB")
                assert transform is not None
                img_t = transform(img)
                tab_list = _normalise_tabular_for_inference(tabular_features_batch[i])
                if len(tab_list) != NUM_TABULAR:
                    continue
                tab_t = torch.tensor(tab_list, dtype=torch.float32)

                img_tensors.append(img_t)
                tab_tensors.append(tab_t)
                valid_indices.append(i)
            except Exception as exc:
                logger.debug("Failed to load image %s: %s", image_paths[i], exc)

        if not valid_indices:
            continue

        img_batch = torch.stack(img_tensors).to(device)  # (B, 3, 224, 224)
        tab_batch = torch.stack(tab_tensors).to(device)  # (B, 6)

        with torch.no_grad():
            logits = model(img_batch, tab_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # (B,)

        for j, global_idx in enumerate(valid_indices):
            prob_good = float(probs[j].item())
            if prob_good >= 0.90:
                confidence = "high"
            elif prob_good >= 0.75:
                confidence = "medium"
            else:
                confidence = "low"

            results[global_idx] = {
                "prob": round(prob_good, 4),
                "signal": prob_good >= threshold,
                "confidence": confidence,
                "threshold": threshold,
                "model_path": _cached_model_path or "",
            }

    return results


# ---------------------------------------------------------------------------
# Model info / diagnostics
# ---------------------------------------------------------------------------


def model_info(model_path: str | None = None) -> dict[str, Any]:
    """Return diagnostic information about the current or specified model.

    Useful for the dashboard / health checks.
    """
    if not _TORCH_AVAILABLE:
        return {"available": False, "error": "PyTorch not installed"}

    path = model_path or _find_latest_model()
    if path is None:
        return {"available": False, "error": "No trained model found"}

    try:
        file_stat = os.stat(path)
        size_mb = file_stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    except OSError:
        size_mb = 0.0
        modified = ""

    info = {
        "available": True,
        "model_path": path,
        "size_mb": round(size_mb, 1),
        "modified": modified,
        "device": get_device(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "image_size": IMAGE_SIZE,
        "num_tabular_features": NUM_TABULAR,
        "tabular_features": TABULAR_FEATURES,
        "default_threshold": DEFAULT_THRESHOLD,
    }

    if torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    return info


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    """Simple CLI for training and inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Breakout CNN — Train or Predict")
    sub = parser.add_subparsers(dest="command")

    # Train
    train_parser = sub.add_parser("train", help="Train the model")
    train_parser.add_argument("--csv", required=True, help="Path to training CSV")
    train_parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--freeze-epochs", type=int, default=2)
    train_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    train_parser.add_argument("--image-root", default=None)
    train_parser.add_argument("--workers", type=int, default=4)

    # Predict
    pred_parser = sub.add_parser("predict", help="Predict on a single image")
    pred_parser.add_argument("--image", required=True, help="Path to chart PNG")
    pred_parser.add_argument(
        "--features",
        nargs=NUM_TABULAR,
        type=float,
        required=True,
        help=f"Tabular features: {', '.join(TABULAR_FEATURES)}",
    )
    pred_parser.add_argument("--model", default=None, help="Model path (default: latest)")
    pred_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    # Info
    sub.add_parser("info", help="Show model info")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.command == "train":
        result = train_model(
            data_csv=args.csv,
            val_csv=args.val_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_epochs=args.freeze_epochs,
            model_dir=args.model_dir,
            image_root=args.image_root,
            num_workers=args.workers,
        )
        if result:
            print(f"\nModel saved to: {result}")
        else:
            print("\nTraining failed")
            exit(1)

    elif args.command == "predict":
        result = predict_breakout(
            image_path=args.image,
            tabular_features=args.features,
            model_path=args.model,
            threshold=args.threshold,
        )
        if result:
            signal_str = "SIGNAL" if result["signal"] else "NO SIGNAL"
            print(
                f"\n{signal_str} — P(good breakout) = {result['prob']:.4f} "
                f"(threshold={result['threshold']}, confidence={result['confidence']})"
            )
        else:
            print("\nPrediction failed")
            exit(1)

    elif args.command == "info":
        info = model_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
