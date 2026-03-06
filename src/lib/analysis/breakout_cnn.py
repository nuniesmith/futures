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

import contextlib
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
#
# v6 contract (12 features) — adds:
#   breakout_type_ord    — ordinal of BreakoutType enum (0–12) / 12, normalised
#   asset_volatility_class — 0=low, 0.5=medium, 1.0=high (based on ATR vs peer group)
#   range_atr_ratio      — range_size / ATR — how "wide" the setup range is
#   hour_of_day          — ET hour normalised to [0, 1] (0h=0.0 … 23h≈0.96)
#
# v5 → v6 is a BREAKING change for the model weights.  Old 8-feature models
# will be auto-padded to 12 by NT8's OrbCnnPredictor (runtime dim detection).
# Retrain required to benefit from the new features.
TABULAR_FEATURES: list[str] = [
    # ── v5 features (original 8) ──────────────────────────────────────────
    "quality_pct",  # 0–100, normalised to 0–1
    "volume_ratio",  # breakout bar vol / 20-bar avg vol
    "atr_pct",  # ATR as % of price
    "cvd_delta",  # cumulative volume delta (normalised, −1 to +1)
    "nr7_flag",  # 1.0 if NR7 day, else 0.0
    "direction_flag",  # 1.0 = LONG, 0.0 = SHORT
    "session_flag",  # ordinal position in Globex day cycle [0, 1]
    "london_overlap_flag",  # 1.0 if breakout in 08:00–09:00 ET overlap
    # ── v6 additions ──────────────────────────────────────────────────────
    "breakout_type_ord",  # BreakoutType ordinal / 12  → [0.0, 1.0]
    "asset_volatility_class",  # 0.0=low  0.5=medium  1.0=high
    "range_atr_ratio",  # range_size / ATR  (clamped to [0, 3], /3 → [0,1])
    "hour_of_day",  # ET hour / 23  → [0, 1]
]

NUM_TABULAR = len(TABULAR_FEATURES)

# Feature contract version — bump whenever TABULAR_FEATURES changes.
FEATURE_CONTRACT_VERSION = 6

# BreakoutType ordinal map — mirrors lib.services.engine.breakout.BreakoutType
# and the C# enum in OrbCnnPredictor.  Using a local mapping avoids a circular
# import between analysis and services layers.
BREAKOUT_TYPE_ORDINALS: dict[str, float] = {
    "ORB": 0.0 / 12,
    "PDR": 1.0 / 12,
    "IB": 2.0 / 12,
    "CONS": 3.0 / 12,
    "WEEKLY": 4.0 / 12,
    "MONTHLY": 5.0 / 12,
    "ASIAN": 6.0 / 12,
    "BBSQUEEZE": 7.0 / 12,
    "VA": 8.0 / 12,
    "INSIDE": 9.0 / 12,
    "GAP": 10.0 / 12,
    "PIVOT": 11.0 / 12,
    "FIB": 12.0 / 12,
}

# Asset volatility classes — maps ticker → class float.
# Low (0.0): stable, narrow daily ranges (ZN, 6E, M6B)
# Medium (0.5): moderate volatility (MES, MYM, MGC, MCL)
# High (1.0): wide intraday ranges (MNQ, M2K, MBT, BTC/ETH crypto)
ASSET_VOLATILITY_CLASS: dict[str, float] = {
    # Low
    "ZN": 0.0,
    "M6B": 0.0,
    "M6E": 0.0,
    # Medium
    "MES": 0.5,
    "MYM": 0.5,
    "MGC": 0.5,
    "MCL": 0.5,
    "SIL": 0.5,
    "M2K": 0.5,
    # High
    "MNQ": 1.0,
    "MBT": 1.0,
    "MHG": 1.0,
    # Kraken crypto — treat as high vol
    "BTC": 1.0,
    "ETH": 1.0,
    "SOL": 1.0,
    "LINK": 1.0,
    "AVAX": 1.0,
    "DOT": 1.0,
    "ADA": 1.0,
    "MATIC": 1.0,
    "XRP": 1.0,
}


def get_breakout_type_ordinal(breakout_type: str) -> float:
    """Return the normalised ordinal [0, 1] for a BreakoutType string.

    Accepts both upper-case ("ORB", "PDR") and lower-case / mixed variants.
    Falls back to 0.0 (ORB) for unknown types.
    """
    return BREAKOUT_TYPE_ORDINALS.get(str(breakout_type).upper().strip(), 0.0)


def get_asset_volatility_class(ticker: str) -> float:
    """Return the volatility class [0.0 low, 0.5 medium, 1.0 high] for *ticker*.

    Falls back to 0.5 (medium) for unknown tickers.
    """
    return ASSET_VOLATILITY_CLASS.get(str(ticker).upper().strip(), 0.5)


# Image pre-processing — matches ImageNet stats used by EfficientNetV2
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference threshold — probability above this → "send signal"
DEFAULT_THRESHOLD = 0.82

# Per-session inference thresholds.
#
# Rationale: overnight sessions (CME open, Sydney, Tokyo, Shanghai) have
# thinner markets and noisier price action than the primary London / US
# sessions, so we require lower confidence to avoid over-filtering the
# smaller signal pool while still maintaining signal quality.  Daytime
# sessions (Frankfurt, London, London-NY, US, CME Settlement) keep the
# full 0.82 bar.
#
# These are *starting* values — tune after 2 weeks of paper-trade data
# by comparing CNN probability distributions per session in Grafana and
# adjusting to match the 58-65% win-rate target.
#
# Keys match ORBSession.key values in orb.py / SESSION_BY_KEY.
SESSION_THRESHOLDS: dict[str, float] = {
    # ── Overnight sessions (18:00–03:00 ET, thin markets) ──────────────
    "cme": 0.75,  # CME Globex re-open 18:00 ET — first bars, wide spread
    "sydney": 0.72,  # Sydney/ASX 18:30 ET — thinnest session
    "tokyo": 0.74,  # Tokyo/TSE 19:00 ET — narrow range, metals/JPY
    "shanghai": 0.74,  # Shanghai/HK 21:00 ET — copper/gold driver
    # ── Primary daytime sessions ────────────────────────────────────────
    "frankfurt": 0.80,  # Frankfurt/Xetra 03:00 ET — pre-London, good vol
    "london": 0.82,  # London Open 03:00 ET — PRIMARY, highest conviction
    "london_ny": 0.82,  # London-NY Crossover 08:00 ET — highest volume
    "us": 0.82,  # US Equity Open 09:30 ET — classic ORB session
    "cme_settle": 0.78,  # CME Settlement 14:00 ET — metals/energy only
}


def get_session_threshold(session_key: str | None) -> float:
    """Return the CNN inference threshold for *session_key*.

    Falls back to ``DEFAULT_THRESHOLD`` (0.82) for unknown or None keys.
    This is the single authoritative lookup used by both
    ``predict_breakout()`` and ``predict_breakout_batch()``.

    Args:
        session_key: ORBSession.key string (e.g. "london", "tokyo", "us").
                     None or empty string returns DEFAULT_THRESHOLD.

    Returns:
        Float probability threshold in [0, 1].

    Example::

        >>> get_session_threshold("tokyo")
        0.74
        >>> get_session_threshold("london")
        0.82
        >>> get_session_threshold(None)
        0.82
    """
    if not session_key:
        return DEFAULT_THRESHOLD
    return SESSION_THRESHOLDS.get(session_key.lower().strip(), DEFAULT_THRESHOLD)


# Ordinal session encoding — maps ORBSession.key → float in [0, 1].
# Encodes the session's position in the 24-hour Globex day cycle so the
# tabular head can learn time-of-day patterns.  Used by BreakoutDataset
# and _normalise_tabular_for_inference in place of the old binary
# session_flag (1.0 = US, 0.0 = London).
#
# Ordered chronologically within the Globex day (18:00 ET start):
#   cme(18:00) → sydney(18:30) → tokyo(19:00) → shanghai(21:00) →
#   frankfurt(03:00) → london(03:00) → london_ny(08:00) →
#   us(09:30) → cme_settle(14:00)
SESSION_ORDINAL: dict[str, float] = {
    "cme": 0.0 / 8,  # 18:00 ET — position 0
    "sydney": 1.0 / 8,  # 18:30 ET — position 1
    "tokyo": 2.0 / 8,  # 19:00 ET — position 2
    "shanghai": 3.0 / 8,  # 21:00 ET — position 3
    "frankfurt": 4.0 / 8,  # 03:00 ET — position 4
    "london": 5.0 / 8,  # 03:00 ET — position 5
    "london_ny": 6.0 / 8,  # 08:00 ET — position 6
    "us": 7.0 / 8,  # 09:30 ET — position 7
    "cme_settle": 8.0 / 8,  # 14:00 ET — position 8
}

# Backward-compat aliases so old callers that pass session_flag=1.0 (US)
# or session_flag=0.0 (London) still get sensible ordinal values.
_SESSION_FLAG_COMPAT = {1.0: SESSION_ORDINAL["us"], 0.0: SESSION_ORDINAL["london"]}


def get_session_ordinal(session_key: str | None) -> float:
    """Return the ordinal encoding [0, 1] for *session_key*.

    Falls back to the US session ordinal (0.875) for unknown keys.
    Accepts the legacy float values ``1.0`` (US) and ``0.0`` (London)
    for backward compatibility with old callers.

    Args:
        session_key: ORBSession.key string, or None.

    Returns:
        Float in [0.0, 1.0] representing session position in the Globex day.
    """
    if session_key is None:
        return SESSION_ORDINAL["us"]
    # Legacy float-as-string passthrough (e.g. "1.0", "0.0")
    with contextlib.suppress(ValueError):
        fval = float(session_key)
        if fval in _SESSION_FLAG_COMPAT:
            return _SESSION_FLAG_COMPAT[fval]
    return SESSION_ORDINAL.get(str(session_key).lower().strip(), SESSION_ORDINAL["us"])


# Model output directory
DEFAULT_MODEL_DIR = "models"
MODEL_PREFIX = "breakout_cnn_"

# Thread lock for model loading
_model_lock = threading.Lock()
_cached_model: Any | None = None
_cached_model_path: str | None = None
_cached_model_mtime: float = 0.0


def invalidate_model_cache() -> bool:
    """Invalidate the cached CNN model so the next inference reloads from disk.

    Thread-safe.  Returns True if a cached model was actually evicted,
    False if the cache was already empty.

    Called by the engine's hot-reload watcher when it detects that the
    champion model file has changed on disk (new ``st_mtime``).
    """
    global _cached_model, _cached_model_path, _cached_model_mtime
    with _model_lock:
        had_model = _cached_model is not None
        _cached_model = None
        _cached_model_path = None
        _cached_model_mtime = 0.0
    if had_model:
        logger.info("CNN model cache invalidated — next inference will reload from disk")
    return had_model


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

    Augmentation strategy:
      - Mild random crop (±16 px): simulates slight chart zoom variation
      - No horizontal flip: chart direction (left→right time) must be preserved
      - Mild brightness/contrast jitter (±8%): handles monitor calibration
        differences and theme variations (dark vs light dashboard)
      - Mild saturation jitter (±5%): Kraken/crypto pairs use different
        colour palettes from futures charts
      - Random rotation (±1.5°): simulates slight chart panel tilt in screenshots
      - Random erasing (p=0.05, max 10% area): simulates minor UI overlays
        or partial occlusions on the dashboard
    """
    if not _TORCH_AVAILABLE:
        return None
    return T.Compose(
        [
            T.Resize((IMAGE_SIZE + 16, IMAGE_SIZE + 16)),
            T.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomHorizontalFlip(p=0.0),  # disabled — chart direction matters
            T.RandomRotation(degrees=1.5, fill=0),  # tiny tilt only
            T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            # Random erasing AFTER ToTensor (operates on tensor, not PIL)
            # Simulates minor UI overlays / partial occlusion.
            # p=0.05: only 5% of images affected; scale capped at 10% area.
            T.RandomErasing(p=0.05, scale=(0.01, 0.10), ratio=(0.3, 3.3), value=0),
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

            if "label" not in self.df.columns:
                raise ValueError("CSV must have a 'label' column")

            # --- Pre-validate: aggressively remove rows without usable images ---
            initial_count = len(self.df)

            # 1. Drop rows where image_path is NaN or empty string
            self.df = self.df.dropna(subset=["image_path"])
            self.df = self.df[self.df["image_path"].astype(str).str.strip().ne("")]
            dropped_empty = initial_count - len(self.df)

            # 2. Verify image files actually exist on disk
            def _resolve(p: str) -> str:
                p = str(p).strip()
                if self.image_root and not os.path.isabs(p):
                    return os.path.join(self.image_root, p)
                return p

            exists_mask = self.df["image_path"].apply(lambda p: os.path.isfile(_resolve(str(p))))
            dropped_missing = int((~exists_mask).sum())
            self.df = self.df[exists_mask]

            # 3. Reset index so iloc is contiguous
            self.df = self.df.reset_index(drop=True)

            if dropped_empty > 0 or dropped_missing > 0:
                logger.warning(
                    "BreakoutDataset: dropped %d empty-path + %d missing-file rows from %s (kept %d / %d)",
                    dropped_empty,
                    dropped_missing,
                    csv_path,
                    len(self.df),
                    initial_count,
                )

            logger.info("BreakoutDataset loaded: %d samples from %s", len(self.df), csv_path)

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int):
            row = self.df.iloc[idx]

            # --- Image ---
            img_path = str(row["image_path"]).strip()
            if self.image_root and not os.path.isabs(img_path):
                img_path = os.path.join(self.image_root, img_path)

            valid = True  # tracks whether this sample should be used
            try:
                img = Image.open(img_path).convert("RGB")

                # Detect degenerate / near-blank images: if the image has
                # essentially zero variance it contains no chart information
                # (e.g. a solid-colour fallback from a render failure).
                img_arr = np.asarray(img)
                if img_arr.std() < 2.0:
                    logger.warning(
                        "Near-blank image detected (std=%.2f): %s — marking invalid",
                        img_arr.std(),
                        img_path,
                    )
                    valid = False
            except Exception as exc:
                logger.warning("Failed to load image %s: %s — marking invalid", img_path, exc)
                # Create a dummy image so tensor pipeline doesn't crash;
                # the valid=False flag tells the collate_fn to drop this sample.
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
                valid = False

            if self.transform:
                img = self.transform(img)

            # --- Tabular features v6 (normalised to ~0–1 range) ---
            # volume_ratio: raw can be 0–30+; log-scale then clamp to [0, 1]
            _vol_raw = max(float(row.get("volume_ratio", 1.0)), 0.01)
            _vol_norm = min(np.log1p(_vol_raw) / np.log1p(10.0), 1.0)  # log1p(10)≈2.4

            # atr_pct: raw is typically 0.0001–0.01; scale ×100 then clamp
            _atr_norm = min(float(row.get("atr_pct", 0.0)) * 100.0, 1.0)

            # cvd_delta: clamp to [-1, 1]
            _cvd_raw = float(row.get("cvd_delta", 0.0))
            _cvd_norm = max(min(_cvd_raw, 1.0), -1.0)

            # Determine session ordinal + hour from breakout_time
            _session = SESSION_ORDINAL["us"]  # default
            _bt = ""
            _hour = 10  # default to US-session hour
            try:
                _bt = str(row.get("breakout_time", ""))
                if _bt and " " in _bt:
                    # Parse hour from timestamp like "2026-01-29 03:30:00-05:00"
                    _hour = int(_bt.split(" ")[1].split(":")[0])
                    # Map hour to approximate session ordinal
                    if _hour < 3:
                        _session = SESSION_ORDINAL["shanghai"]
                    elif _hour < 8:
                        _session = SESSION_ORDINAL["london"]
                    elif _hour < 9:
                        _session = SESSION_ORDINAL["london_ny"]
                    else:
                        _session = SESSION_ORDINAL["us"]
            except Exception:
                _session = SESSION_ORDINAL["us"]

            # London/NY overlap flag: 08:00–09:00 ET is historically the
            # strongest breakout window when both sessions are active.
            _london_overlap = 1.0 if 8 <= _hour <= 9 else 0.0

            # ── v6 features ───────────────────────────────────────────────
            # breakout_type_ord: BreakoutType ordinal / 12
            _btype_str = str(row.get("breakout_type", "ORB")).upper().strip()
            _btype_ord = BREAKOUT_TYPE_ORDINALS.get(_btype_str, 0.0)

            # asset_volatility_class: 0.0=low, 0.5=medium, 1.0=high
            _ticker = str(row.get("ticker", row.get("symbol", ""))).upper().strip()
            _vol_class = ASSET_VOLATILITY_CLASS.get(_ticker, 0.5)

            # range_atr_ratio: range_size / ATR, clamped to [0, 3] then /3 → [0, 1]
            _range_size = float(row.get("range_size", row.get("or_range", 0.0)))
            _atr_raw = float(row.get("atr_value", row.get("atr_pct", 0.0)))
            _range_atr = min(_range_size / _atr_raw, 3.0) / 3.0 if _atr_raw > 0 else 0.5

            # hour_of_day: ET hour / 23
            _hour_norm = min(_hour, 23) / 23.0

            tabular = torch.tensor(
                [
                    float(row.get("quality_pct", 0)) / 100.0,  # already 0–1
                    _vol_norm,
                    _atr_norm,
                    _cvd_norm,
                    float(row.get("nr7_flag", 0)),  # 0 or 1
                    1.0 if str(row.get("direction", "")).upper().startswith("L") else 0.0,
                    _session,  # session ordinal [0, 1]
                    _london_overlap,  # 1.0 if 08:00–09:00 ET
                    # v6
                    _btype_ord,  # breakout_type / 12
                    _vol_class,  # 0.0/0.5/1.0
                    _range_atr,  # range/ATR clamped [0,1]
                    _hour_norm,  # ET hour / 23
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

            return img, tabular, torch.tensor(target, dtype=torch.long), torch.tensor(valid, dtype=torch.bool)

        @staticmethod
        def skip_invalid_collate(batch):
            """Custom collate_fn that drops samples flagged as invalid.

            Each sample is a tuple ``(img, tabular, target, valid_flag)``.
            Only samples where ``valid_flag`` is True are kept.  If the entire
            batch is invalid we return ``None`` — the training loop must check
            for this and skip the batch.
            """
            # batch is a list of (img, tabular, target, valid) tuples
            filtered = [s for s in batch if s[3].item()]
            if not filtered:
                return None  # entire batch was invalid — caller must skip

            imgs = torch.stack([s[0] for s in filtered])
            tabs = torch.stack([s[1] for s in filtered])
            targets = torch.stack([s[2] for s in filtered])
            return imgs, tabs, targets


else:
    # Stub when torch is not available
    class BreakoutDataset:  # type: ignore[no-redef,misc]
        """Stub for environments without PyTorch."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed — cannot create BreakoutDataset")

        def __len__(self) -> int:
            return 0

        def __getitem__(self, idx: int) -> Any:
            raise RuntimeError("PyTorch is not installed")

        @staticmethod
        def skip_invalid_collate(batch: Any) -> Any:
            """Stub — PyTorch is not installed."""
            raise RuntimeError("PyTorch is not installed")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Number of distinct breakout types — used to size the embedding table.
# Mirrors len(BreakoutType) in lib.core.breakout_types.
NUM_BREAKOUT_TYPES = 13

# Learned embedding dimension for breakout type.
# Replaces the single scalar ``breakout_type_ord`` feature with a richer
# NUM_BREAKOUT_TYPES × BREAKOUT_EMBED_DIM lookup table that the model
# trains end-to-end.  The tabular vector's ``breakout_type_ord`` slot is
# still consumed (so the input dimension stays at NUM_TABULAR = 12) but
# when ``use_type_embedding=True`` that slot is ignored and the embedding
# replaces it in the combined representation.
BREAKOUT_EMBED_DIM = 8


if _TORCH_AVAILABLE:

    class HybridBreakoutCNN(nn.Module):  # type: ignore[no-redef]
        """Hybrid image + tabular model for breakout classification.

        Architecture (use_type_embedding=False, default v5 compat):
          Image branch:   EfficientNetV2-S (pre-trained) → 1280-dim features
          Tabular branch: Linear(NUM_TABULAR→64) → ReLU → Dropout → Linear(64→32)
          Classifier:     Linear(1280+32→256) → ReLU → Dropout → Linear(256→2)

        Architecture (use_type_embedding=True, v6 recommended):
          Image branch:   EfficientNetV2-S (pre-trained) → 1280-dim features
          Type embedding: Embedding(13, 8) — learned per-type representation
                          The scalar ``breakout_type_ord`` slot in the tabular
                          vector is excluded; the embedding is concatenated
                          instead, keeping the combined width the same.
          Tabular branch: Linear(NUM_TABULAR-1→64) → ReLU → Dropout →
                          Linear(64→32) → concat(embed_8) → Linear(40→32)
          Classifier:     Linear(1280+32→256) → ReLU → Dropout → Linear(256→2)

        The model outputs raw logits for 2 classes:
          - Class 0: bad breakout (fail / chop)
          - Class 1: good breakout (clean follow-through)

        Use ``torch.softmax(output, dim=1)[:, 1]`` to get P(good breakout).

        Args:
            num_tabular: Number of tabular features (default NUM_TABULAR=12).
            dropout: Dropout rate for the classifier head (default 0.4).
            pretrained: Load ImageNet pre-trained backbone weights (default True).
            freeze_backbone_epochs: Epochs to freeze backbone at start of training.
            use_type_embedding: If True, replace the scalar ``breakout_type_ord``
                feature at position 8 (index 8 in TABULAR_FEATURES) with a
                learned Embedding(NUM_BREAKOUT_TYPES, BREAKOUT_EMBED_DIM).
                Requires the caller to pass integer breakout type IDs separately
                via the ``type_ids`` argument in forward().  Default False for
                backward compatibility with existing checkpoints.
        """

        # Index of the ``breakout_type_ord`` feature in TABULAR_FEATURES.
        # When use_type_embedding=True this slot is dropped from the tabular
        # input and replaced by the embedding lookup.
        BREAKOUT_TYPE_ORD_IDX: int = 8  # position in v6 TABULAR_FEATURES list

        def __init__(
            self,
            num_tabular: int = NUM_TABULAR,
            dropout: float = 0.4,
            pretrained: bool = True,
            freeze_backbone_epochs: int = 0,
            use_type_embedding: bool = False,
        ):
            super().__init__()
            self.num_tabular = num_tabular
            self._freeze_backbone_epochs = freeze_backbone_epochs
            self.use_type_embedding = use_type_embedding

            # --- Image backbone: EfficientNetV2-S ---
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_v2_s(weights=weights)

            # Remove the original classifier head → get 1280-dim features
            # EfficientNetV2-S: features → avgpool → classifier
            # We keep features + avgpool, replace classifier with Identity
            backbone.classifier = nn.Identity()  # type: ignore[assignment]
            self.cnn = backbone
            self._cnn_out_dim = 1280  # EfficientNetV2-S feature dim

            # --- Breakout type embedding (optional) ---
            if use_type_embedding:
                self.type_embedding: nn.Embedding | None = nn.Embedding(
                    NUM_BREAKOUT_TYPES, BREAKOUT_EMBED_DIM, padding_idx=None
                )
                # Tabular head receives (num_tabular - 1) scalar features
                # (breakout_type_ord slot excluded) plus BREAKOUT_EMBED_DIM
                # embedding dims, fused before the final projection.
                _tab_scalar_dim = num_tabular - 1
                self.tabular_head = nn.Sequential(
                    nn.Linear(_tab_scalar_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                )
                # Fusion: merge 32-dim scalar encoding + 8-dim type embedding
                self.type_fusion = nn.Sequential(
                    nn.Linear(32 + BREAKOUT_EMBED_DIM, 32),
                    nn.ReLU(inplace=True),
                )
            else:
                self.type_embedding = None
                self.type_fusion = None  # type: ignore[assignment]
                # --- Tabular branch (scalar-only, v5 compat) ---
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

        def forward(
            self,
            image: torch.Tensor,
            tabular: torch.Tensor,
            type_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                image: (B, 3, 224, 224) normalised image tensor.
                tabular: (B, NUM_TABULAR) float tensor.
                type_ids: (B,) long tensor of breakout type ordinals (0–12).
                          Required when ``use_type_embedding=True``.
                          Ignored when ``use_type_embedding=False``.

            Returns:
                (B, 2) logits tensor.
            """
            img_features = self.cnn(image)  # (B, 1280)

            if self.use_type_embedding and self.type_embedding is not None:
                # Drop the scalar breakout_type_ord slot from tabular
                idx = self.BREAKOUT_TYPE_ORD_IDX
                tab_scalar = torch.cat([tabular[:, :idx], tabular[:, idx + 1 :]], dim=1)  # (B, NUM_TABULAR-1)
                scalar_enc = self.tabular_head(tab_scalar)  # (B, 32)

                # Breakout type embedding
                if type_ids is None:
                    # Fallback: treat ordinal at idx as integer index
                    type_ids = (tabular[:, idx] * (NUM_BREAKOUT_TYPES - 1)).long().clamp(0, NUM_BREAKOUT_TYPES - 1)
                type_emb = self.type_embedding(type_ids)  # (B, BREAKOUT_EMBED_DIM)

                # Fuse scalar encoding + type embedding → 32-dim
                assert self.type_fusion is not None, "type_fusion must be set when use_type_embedding=True"
                tab_features = self.type_fusion(torch.cat([scalar_enc, type_emb], dim=1))  # (B, 32)
            else:
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

    class HybridBreakoutCNN:  # type: ignore[no-redef,misc]
        """Stub for environments without PyTorch."""

        BREAKOUT_TYPE_ORD_IDX: int = 8

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is not installed — cannot create HybridBreakoutCNN")

        def freeze_backbone(self) -> None:
            raise RuntimeError("PyTorch is not installed")

        def unfreeze_backbone(self) -> None:
            raise RuntimeError("PyTorch is not installed")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _detect_docker() -> bool:
    """Return True if we appear to be running inside a Docker container."""
    try:
        if os.path.exists("/.dockerenv"):
            return True
        with open("/proc/1/cgroup") as f:
            return any("docker" in line or "containerd" in line for line in f)
    except Exception:
        return False


def _safe_num_workers(requested: int) -> int:
    """Clamp DataLoader num_workers to 0 inside Docker when /dev/shm is small.

    PyTorch multiprocess DataLoader workers communicate via shared memory.
    Docker's default /dev/shm is only 64 MB, which causes:
        RuntimeError: unable to allocate shared memory(shm)
    Setting num_workers=0 forces single-process loading (slower but safe).
    """
    if requested == 0:
        return 0

    if not _detect_docker():
        return requested

    # Check /dev/shm size — need at least 512 MB for multi-worker loading
    try:
        stat = os.statvfs("/dev/shm")
        shm_bytes = stat.f_frsize * stat.f_blocks
        shm_mb = shm_bytes / (1024 * 1024)
        if shm_mb >= 512:
            logger.info("Docker detected with %.0f MB /dev/shm — using %d DataLoader workers", shm_mb, requested)
            return requested
        else:
            logger.warning(
                "Docker detected with only %.0f MB /dev/shm (need ≥512 MB) — "
                "forcing num_workers=0 to avoid shared memory crash. "
                "Fix: add 'shm_size: 2gb' to docker-compose.yml",
                shm_mb,
            )
            return 0
    except Exception:
        logger.warning("Docker detected but cannot check /dev/shm — forcing num_workers=0 for safety")
        return 0


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

    # Clamp num_workers for Docker shared memory safety
    num_workers = _safe_num_workers(num_workers)

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
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=BreakoutDataset.skip_invalid_collate,
    )

    # --- Model ---
    use_type_emb = os.getenv("CNN_TYPE_EMBEDDING", "0").strip() in ("1", "true", "yes")
    model = HybridBreakoutCNN(pretrained=True, use_type_embedding=use_type_emb)
    if use_type_emb:
        logger.info("HybridBreakoutCNN: using learned BreakoutType embedding (dim=%d)", BREAKOUT_EMBED_DIM)
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

        for batch_idx, batch in enumerate(train_loader):
            # skip_invalid_collate returns None when every sample was invalid
            if batch is None:
                continue
            imgs, tabs, labels = batch
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Derive type_ids from breakout_type_ord slot for embedding models
            type_ids = None
            if use_type_emb and tabs.shape[1] > HybridBreakoutCNN.BREAKOUT_TYPE_ORD_IDX:
                idx = HybridBreakoutCNN.BREAKOUT_TYPE_ORD_IDX
                type_ids = (tabs[:, idx] * (NUM_BREAKOUT_TYPES - 1)).long().clamp(0, NUM_BREAKOUT_TYPES - 1)
            outputs = model(imgs, tabs, type_ids=type_ids)
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
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, tabs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                val_type_ids = None
                if use_type_emb and tabs.shape[1] > HybridBreakoutCNN.BREAKOUT_TYPE_ORD_IDX:
                    idx = HybridBreakoutCNN.BREAKOUT_TYPE_ORD_IDX
                    val_type_ids = (tabs[:, idx] * (NUM_BREAKOUT_TYPES - 1)).long().clamp(0, NUM_BREAKOUT_TYPES - 1)
                outputs = model(imgs, tabs, type_ids=val_type_ids)
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
    invalidate_model_cache()

    return result_path


def evaluate_model(
    model_path: str,
    val_csv: str,
    image_root: str | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, Any] | None:
    """Evaluate a trained model checkpoint against a validation CSV.

    Computes accuracy, precision, and recall (macro-averaged) on the
    validation set.  Intended to be called by the trainer server after
    :func:`train_model` completes, so the pipeline can gate promotion on
    concrete metrics.

    Args:
        model_path: Path to a ``.pt`` state-dict checkpoint.
        val_csv: Path to the validation CSV (same format as training CSV).
        image_root: Optional root directory prepended to ``image_path``
                    values in the CSV.
        batch_size: Evaluation batch size (default 32).
        num_workers: DataLoader workers (default 4).

    Returns:
        Dict with keys ``val_accuracy``, ``val_precision``, ``val_recall``
        (all 0.0–1.0 floats), or ``None`` if evaluation failed.
    """
    if not _TORCH_AVAILABLE:
        logger.error("PyTorch is not installed — cannot evaluate model")
        return None

    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        device = torch.device(get_device())
        num_workers = _safe_num_workers(num_workers)

        # Load model
        model = HybridBreakoutCNN(pretrained=False)
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        # Build validation loader
        val_transform = get_inference_transform()
        val_dataset = BreakoutDataset(val_csv, transform=val_transform, image_root=image_root)

        if len(val_dataset) == 0:
            logger.warning("Validation dataset is empty — cannot evaluate")
            return None

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=BreakoutDataset.skip_invalid_collate,
        )

        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                imgs, tabs, labels = batch
                imgs = imgs.to(device, non_blocking=True)
                tabs = tabs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(imgs, tabs)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        if not all_labels:
            logger.warning("No valid samples evaluated — cannot compute metrics")
            return None

        acc = accuracy_score(all_labels, all_preds)
        # Use zero_division=0.0 so we don't crash when a class is absent
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0.0)  # type: ignore[arg-type]
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0.0)  # type: ignore[arg-type]

        logger.info(
            "Evaluation complete — accuracy: %.1f%%, precision: %.1f%%, recall: %.1f%% (%d samples)",
            acc * 100,
            prec * 100,
            rec * 100,
            len(all_labels),
        )

        return {
            "val_accuracy": round(float(acc), 4),
            "val_precision": round(float(prec), 4),
            "val_recall": round(float(rec), 4),
            "num_samples": len(all_labels),
        }

    except Exception as exc:
        logger.error("Model evaluation failed: %s", exc, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _find_best_model(model_dir: str = DEFAULT_MODEL_DIR) -> str | None:
    """Find the best available model in *model_dir*.

    Selection priority:
      1. ``breakout_cnn_best.pt`` (the promoted champion) — always preferred
         when it exists.
      2. The checkpoint with the highest ``val_accuracy`` recorded in its
         companion ``<stem>_meta.json`` sidecar file.
      3. The most recently modified ``.pt`` file (mtime fallback when no
         meta JSON is available for any checkpoint).

    Args:
        model_dir: Directory to search (default ``"models"``).

    Returns:
        Absolute path to the chosen model file, or ``None`` if no models
        are found.
    """
    model_path = Path(model_dir)
    if not model_path.is_dir():
        return None

    # 1. Prefer the promoted champion if it exists
    champion = model_path / "breakout_cnn_best.pt"
    if champion.is_file():
        logger.debug("Model selection: using champion %s", champion)
        return str(champion)

    # 2. Scan all breakout_cnn_*.pt checkpoints
    pt_files = list(model_path.glob(f"{MODEL_PREFIX}*.pt"))
    if not pt_files:
        return None

    # Try to rank by val_accuracy from companion meta JSON sidecars.
    # A meta JSON for checkpoint "breakout_cnn_20260101_020000_acc87.pt"
    # would be named "breakout_cnn_20260101_020000_acc87_meta.json".
    # Also check the global "breakout_cnn_best_meta.json" as a fallback.
    global_meta_path = model_path / "breakout_cnn_best_meta.json"

    def _val_accuracy(pt: Path) -> float:
        # Sidecar meta: same stem + _meta.json
        sidecar = pt.with_name(pt.stem + "_meta.json")
        for candidate in (sidecar, global_meta_path):
            if candidate.is_file():
                try:
                    import json as _json

                    meta = _json.loads(candidate.read_text())
                    return float(meta.get("val_accuracy", 0.0))
                except Exception:
                    pass
        # Try to parse accuracy from filename: "..._accNN.pt"
        import re as _re

        m = _re.search(r"_acc(\d+(?:\.\d+)?)", pt.stem)
        if m:
            return float(m.group(1))
        return 0.0

    scored = [(pt, _val_accuracy(pt)) for pt in pt_files]

    # Check if any checkpoint has a meaningful accuracy score
    if any(score > 0.0 for _, score in scored):
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        logger.debug(
            "Model selection: chose %s (val_acc=%.1f%%) from %d checkpoints",
            best.name,
            scored[0][1],
            len(pt_files),
        )
        return str(best)

    # 3. Fallback: most recently modified file
    pt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    logger.debug("Model selection: fallback to newest mtime %s", pt_files[0].name)
    return str(pt_files[0])


def _find_latest_model(model_dir: str = DEFAULT_MODEL_DIR) -> str | None:
    """Alias for :func:`_find_best_model` kept for backwards compatibility."""
    return _find_best_model(model_dir)


def _build_model_from_checkpoint(state_dict: dict) -> Any:
    """Instantiate a HybridBreakoutCNN whose architecture matches *state_dict*.

    Detects whether the checkpoint was trained with ``use_type_embedding=True``
    by checking for the ``type_embedding.weight`` key in the state dict.

    Returns a model with weights loaded (eval mode, not moved to device yet).
    """
    if not _TORCH_AVAILABLE:
        return None

    use_type_emb = "type_embedding.weight" in state_dict

    # Infer num_tabular from first tabular_head Linear weight shape.
    # For scalar-only models the key is ``tabular_head.0.weight`` with shape
    # (64, num_tabular).  For embedding models it is (64, num_tabular-1).
    num_tabular = NUM_TABULAR
    tab_key = "tabular_head.0.weight"
    if tab_key in state_dict:
        in_features = state_dict[tab_key].shape[1]
        num_tabular = in_features + 1 if use_type_emb else in_features

    model = HybridBreakoutCNN(
        num_tabular=num_tabular,
        pretrained=False,  # weights come from checkpoint
        use_type_embedding=use_type_emb,
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


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

    v6 input order (12 features):
        [0]  quality_pct_norm      — quality_pct / 100 (caller pre-divides)
        [1]  volume_ratio          — raw ratio, will be log-normalised here
        [2]  atr_pct               — ATR / price (will be ×100 clamped)
        [3]  cvd_delta             — cumulative volume delta (clamped ±1)
        [4]  nr7_flag              — 0 or 1
        [5]  direction_flag        — 1.0 LONG / 0.0 SHORT
        [6]  session_flag          — ordinal [0, 1] via get_session_ordinal()
        [7]  london_overlap_flag   — 0 or 1
        [8]  breakout_type_ord     — BreakoutType ordinal / 12  [0, 1]
        [9]  asset_volatility_class — 0.0 / 0.5 / 1.0
        [10] range_atr_ratio       — range/ATR clamped to [0, 1]  (/3 applied here)
        [11] hour_of_day           — ET hour / 23  [0, 1]

    For backward compatibility, vectors of length 8 (v5) are zero-padded to 12.
    The caller is expected to pass quality_pct already divided by 100.

    Returns a list of 12 floats in normalised form.
    """
    f = list(raw_features)

    # v5 → v6 backward compat: zero-pad the 4 new features
    if len(f) == 8:
        f.extend([0.0, 0.5, 0.5, 0.5])  # sensible defaults for new fields

    if len(f) != NUM_TABULAR:
        raise ValueError(
            f"Expected {NUM_TABULAR} tabular features (v6) or 8 (v5 compat), "
            f"got {len(f)}. Required order: {TABULAR_FEATURES}"
        )

    quality_norm = max(min(f[0], 1.0), 0.0)

    # volume_ratio: log-scale, same as dataset
    vol_raw = max(f[1], 0.01)
    vol_norm = min(float(np.log1p(vol_raw) / np.log1p(10.0)), 1.0)

    # atr_pct: ×100 then clamp
    atr_norm = min(f[2] * 100.0, 1.0)

    # cvd_delta: clamp [-1, 1]
    cvd_norm = max(min(f[3], 1.0), -1.0)

    # f[4] = nr7_flag (0 or 1), f[5] = direction_flag (0 or 1),
    # f[6] = session ordinal [0, 1], f[7] = london_overlap_flag (0 or 1)
    # All pass through unchanged.

    # f[8] = breakout_type_ord [0, 1] — pass through
    btype_ord = max(min(f[8], 1.0), 0.0)

    # f[9] = asset_volatility_class — 0.0 / 0.5 / 1.0 — pass through
    vol_class = max(min(f[9], 1.0), 0.0)

    # f[10] = range_atr_ratio — caller passes raw ratio; clamp to [0, 3] then /3
    # If caller already normalised to [0, 1], the clamp is a no-op.
    raw_range_atr = f[10]
    range_atr = min(raw_range_atr / 3.0, 1.0) if raw_range_atr > 1.0 else max(min(raw_range_atr, 1.0), 0.0)

    # f[11] = hour_of_day — caller passes ET hour / 23; clamp to [0, 1]
    hour_norm = max(min(f[11], 1.0), 0.0)

    return [
        quality_norm,
        vol_norm,
        atr_norm,
        cvd_norm,
        f[4],
        f[5],
        f[6],
        f[7],
        btype_ord,
        vol_class,
        range_atr,
        hour_norm,
    ]


def predict_breakout(
    image_path: str,
    tabular_features: Sequence[float],
    model_path: str | None = None,
    threshold: float | None = None,
    session_key: str | None = None,
) -> dict[str, Any] | None:
    """Predict whether a chart snapshot shows a high-quality breakout.

    Args:
        image_path: Path to the PNG chart snapshot.
        tabular_features: List/tuple of 8 floats in TABULAR_FEATURES order:
            [quality_pct_norm, volume_ratio, atr_pct, cvd_delta, nr7_flag,
             direction_flag, session_ordinal, london_overlap_flag]
            - quality_pct_norm: quality_pct / 100 (0.0–1.0)
            - volume_ratio: breakout bar volume / 20-bar average
            - atr_pct: ATR as fraction of price
            - cvd_delta: normalised CVD delta (-1 to 1)
            - nr7_flag: 1.0 if NR7 day, 0.0 otherwise
            - direction_flag: 1.0 for LONG, 0.0 for SHORT
            - session_ordinal: ordinal position in Globex day via
              ``get_session_ordinal(session_key)`` — replaces old binary
              session_flag (1.0=US / 0.0=London).  Old binary values are
              accepted for backward compatibility.
            - london_overlap_flag: 1.0 if 08:00–09:00 ET, 0.0 otherwise
        model_path: Explicit model path (default: latest in models/).
        threshold: Probability threshold for "signal" verdict.
                   When None (default), the per-session threshold from
                   ``SESSION_THRESHOLDS`` is used via ``session_key``.
                   Passing an explicit float overrides the session default.
        session_key: ORBSession.key (e.g. "london", "tokyo", "us").
                     Used to look up the per-session threshold and for
                     logging.  Ignored if *threshold* is explicitly set.

    Returns:
        Dict with:
          - prob: float (0.0–1.0) — probability of clean breakout
          - signal: bool — True if prob >= threshold
          - confidence: str — "high", "medium", or "low"
          - threshold: float — the threshold that was applied
          - session_key: str — which session threshold was used
          - model_path: str — which model was used
        Or None if inference failed.
    """
    if not _TORCH_AVAILABLE:
        logger.warning("PyTorch not available — cannot run inference")
        return None

    model = _load_model(model_path)
    if model is None:
        return None

    # Resolve the effective threshold — explicit arg wins, then per-session,
    # then global default.
    effective_threshold = threshold if threshold is not None else get_session_threshold(session_key)

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

        tab_tensor = torch.tensor([tab_list], dtype=torch.float32).to(device)  # (1, 8)

        # Inference
        with torch.no_grad():
            logits = model(img_tensor, tab_tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=1)
            prob_good = float(probs[0, 1].item())

        # Confidence bucketing — relative to the effective threshold so that
        # a "high" confidence call means the same quality bar regardless of
        # which session we're in.
        if prob_good >= effective_threshold + 0.08:
            confidence = "high"
        elif prob_good >= effective_threshold - 0.04:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "prob": round(prob_good, 4),
            "signal": prob_good >= effective_threshold,
            "confidence": confidence,
            "threshold": effective_threshold,
            "session_key": session_key or "",
            "model_path": _cached_model_path or "",
        }

    except Exception as exc:
        logger.error("Inference failed for %s: %s", image_path, exc, exc_info=True)
        return None


def predict_breakout_batch(
    image_paths: Sequence[str],
    tabular_features_batch: Sequence[Sequence[float]],
    model_path: str | None = None,
    threshold: float | None = None,
    session_key: str | None = None,
    batch_size: int = 16,
) -> list[dict[str, Any] | None]:
    """Batch inference for multiple chart snapshots.

    More efficient than calling ``predict_breakout`` in a loop because it
    batches the GPU forward passes.

    Args:
        image_paths: List of PNG paths.
        tabular_features_batch: List of tabular feature vectors (one per image).
        model_path: Explicit model path (default: latest).
        threshold: Signal threshold.  When None (default), the per-session
                   threshold from ``SESSION_THRESHOLDS`` is used via
                   ``session_key``.  Passing an explicit float overrides it.
        session_key: ORBSession.key (e.g. "london", "tokyo", "us") used to
                     look up the per-session threshold.  Ignored if
                     *threshold* is explicitly set.
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

    # Resolve the effective threshold once for the whole batch.
    effective_threshold = threshold if threshold is not None else get_session_threshold(session_key)

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
            # Relative confidence bucketing — mirrors predict_breakout so
            # "high"/"medium"/"low" mean the same quality bar across sessions.
            if prob_good >= effective_threshold + 0.08:
                confidence = "high"
            elif prob_good >= effective_threshold - 0.04:
                confidence = "medium"
            else:
                confidence = "low"

            results[global_idx] = {
                "prob": round(prob_good, 4),
                "signal": prob_good >= effective_threshold,
                "confidence": confidence,
                "threshold": effective_threshold,
                "session_key": session_key or "",
                "model_path": _cached_model_path or "",
            }

    return results


# ---------------------------------------------------------------------------
# Model info / diagnostics
# ---------------------------------------------------------------------------


def generate_feature_contract(output_path: str | None = None) -> dict[str, Any]:
    """Generate and optionally write the ``feature_contract.json`` v6 file.

    The contract encodes every parameter needed by consumers (engine, NT8
    C# OrbCnnPredictor, rb trainer) to correctly prepare the tabular feature
    vector and interpret model outputs.

    Structure::

        {
          "version":          6,
          "num_tabular":      12,
          "tabular_features": [...],
          "default_threshold": 0.82,
          "image_size":       224,
          "breakout_types":   {<BreakoutType.name>: {ordinal, box_style, tp1_atr_mult, ...}},
          "sessions":         {<session_key>: {ordinal, session_ordinal, cnn_threshold, ...}},
          "asset_volatility": {<ticker>: 0.0|0.5|1.0},
          "generated_at":     "<ISO timestamp>",
        }

    Args:
        output_path: If given, write the JSON to this path (creates parent
            directories as needed).  If ``None``, only return the dict.

    Returns:
        The contract as a Python dict (always returned regardless of
        ``output_path``).
    """
    import json as _json
    from datetime import datetime as _dt

    # ── breakout_types section ─────────────────────────────────────────────
    try:
        from lib.core.breakout_types import to_feature_contract_dict as _bt_contract

        bt_section = _bt_contract()
    except Exception:
        # Fallback: use the local ordinal map
        bt_section = {
            name: {"ordinal": round(ord_val * 12), "breakout_type_ord": round(ord_val, 6)}
            for name, ord_val in BREAKOUT_TYPE_ORDINALS.items()
        }

    # ── sessions section ───────────────────────────────────────────────────
    try:
        from lib.core.multi_session import to_feature_contract_dict as _sess_contract

        sess_section = _sess_contract()
    except Exception:
        sess_section = {}

    contract: dict[str, Any] = {
        "version": FEATURE_CONTRACT_VERSION,
        "num_tabular": NUM_TABULAR,
        "tabular_features": TABULAR_FEATURES,
        "default_threshold": DEFAULT_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "breakout_types": bt_section,
        "sessions": sess_section,
        "asset_volatility": ASSET_VOLATILITY_CLASS,
        "generated_at": _dt.now(tz=__import__("datetime").timezone.utc).isoformat(),
    }

    if output_path:
        import os as _os

        _os.makedirs(_os.path.dirname(_os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            _json.dump(contract, fh, indent=2)

    return contract


def get_type_embedding_weights(model_path: str | None = None) -> dict[str, Any] | None:
    """Return the learned BreakoutType embedding matrix from a checkpoint.

    Returns a dict mapping breakout type name → embedding vector (list of
    floats), or None if the model was not trained with type embeddings or
    torch is not available.

    Useful for debugging/visualising what the model has learned about each
    breakout type.
    """
    if not _TORCH_AVAILABLE:
        return None

    mp = model_path or _find_best_model()
    if not mp or not os.path.isfile(mp):
        return None

    try:
        try:
            sd = torch.load(mp, map_location="cpu", weights_only=True)
        except TypeError:
            sd = torch.load(mp, map_location="cpu")  # type: ignore[call-overload]

        if "type_embedding.weight" not in sd:
            return None

        emb_weight = sd["type_embedding.weight"].numpy()  # (13, BREAKOUT_EMBED_DIM)

        # Map ordinal index → breakout type name
        _ordinal_to_name = {int(round(v * 12)): k for k, v in BREAKOUT_TYPE_ORDINALS.items()}
        result: dict[str, Any] = {}
        for i, vec in enumerate(emb_weight):
            name = _ordinal_to_name.get(i, f"type_{i}")
            result[name] = vec.tolist()
        return result
    except Exception as exc:
        logger.warning("Failed to extract type embedding weights: %s", exc)
        return None


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
    train_parser = sub.add_parser("train", help="Train the CNN model")
    train_parser.add_argument("--csv", required=True, help="Path to training CSV")
    train_parser.add_argument("--val-csv", default=None, help="Path to validation CSV")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--freeze-epochs", type=int, default=2)
    train_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    train_parser.add_argument("--image-root", default=None)
    train_parser.add_argument(
        "--type-embedding",
        action="store_true",
        default=False,
        help="Use learned BreakoutType embedding instead of scalar ordinal feature. "
        "Recommended for v6 models trained on all 13 breakout types.",
    )
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

    # Embedding inspection
    sub.add_parser(
        "embedding",
        help="Print learned BreakoutType embedding weights from the current champion model",
    )

    # Contract
    contract_parser = sub.add_parser("contract", help="Generate feature_contract.json v6")
    contract_parser.add_argument(
        "--output",
        default="feature_contract.json",
        help="Output path for feature_contract.json (default: ./feature_contract.json)",
    )
    contract_parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the contract to stdout without writing a file",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.command == "train":
        # CNN_TYPE_EMBEDDING env var is read inside train_model(); set it
        # from the CLI flag so the caller doesn't have to export it manually.
        if getattr(args, "type_embedding", False):
            os.environ["CNN_TYPE_EMBEDDING"] = "1"
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

    elif args.command == "embedding":
        weights = get_type_embedding_weights()
        if weights is None:
            print("No type embedding found in checkpoint (model not trained with --type-embedding).")
        else:
            import json as _json

            print(_json.dumps(weights, indent=2))

    elif args.command == "info":
        info = model_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

    elif args.command == "contract":
        import json as _json

        if args.print_only:
            contract = generate_feature_contract(output_path=None)
            print(_json.dumps(contract, indent=2))
        else:
            contract = generate_feature_contract(output_path=args.output)
            print(f"✅ feature_contract.json v{contract['version']} written to: {args.output}")
            print(f"   tabular features : {contract['num_tabular']}")
            print(f"   breakout types   : {len(contract['breakout_types'])}")
            print(f"   sessions         : {len(contract['sessions'])}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
