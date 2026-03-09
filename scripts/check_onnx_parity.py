#!/usr/bin/env python3
"""
check_onnx_parity.py — Validate ONNX ↔ PyTorch inference parity
=================================================================

Runs the same 18-feature v6 tabular batch through both the .pt and .onnx
champion models and asserts that the outputs agree to within 1e-4.

Usage:
    python scripts/check_onnx_parity.py
    python scripts/check_onnx_parity.py --models-dir models/
    python scripts/check_onnx_parity.py --n-samples 256 --verbose

Exit codes:
    0 — parity check passed (max abs diff < 1e-4)
    1 — parity check failed or models could not be loaded

Dependencies:
    torch, torchvision, onnxruntime, numpy
    (all present in the trainer Docker image / .venv)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path so lib.* imports resolve when run from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# isort: split
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODELS_DIR = _REPO_ROOT / "models"
PT_FILENAME = "breakout_cnn_best.pt"
ONNX_FILENAME = "breakout_cnn_best.onnx"
CONTRACT_FILENAME = "feature_contract.json"

PARITY_THRESHOLD = 1e-4  # max absolute difference allowed
IMAGE_SIZE = 224  # EfficientNetV2-S input
NUM_TABULAR = 18  # v6 contract
DEFAULT_N_SAMPLES = 64
BATCH_SIZE = 32
SEED = 42

GREEN = "\033[0;32m"
RED = "\033[0;31m"
CYAN = "\033[0;36m"
YELLOW = "\033[1;33m"
DIM = "\033[2m"
NC = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[  ✓ ]{NC} {msg}")


def err(msg: str) -> None:
    print(f"{RED}[FAIL]{NC} {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"{CYAN}[info]{NC} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[warn]{NC} {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_deps() -> bool:
    """Verify all required packages are importable."""
    missing = []
    for pkg in ("torch", "torchvision", "onnxruntime", "numpy", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        err(f"Missing dependencies: {', '.join(missing)}")
        err("Install with: pip install torch torchvision onnxruntime Pillow")
        return False
    return True


def _make_synthetic_batch(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a batch of (image_tensor, tabular_tensor) with valid ranges.

    Images: uint8 RGB in [0, 255] — will be normalised to ImageNet stats.
    Tabular: float32 in [0, 1] per-feature (as the normaliser produces).
    """
    # Synthetic chart images — random noise mimics a flattened chart pixel grid.
    # We use structured noise (gradients) so the backbone doesn't always see
    # a flat field, exercising more of the conv path.
    imgs = np.zeros((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i in range(n):
        for c in range(3):
            base = rng.uniform(0.2, 0.8)
            noise = rng.normal(0, 0.15, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
            imgs[i, c] = np.clip(base + noise, 0.0, 1.0)

    # Apply ImageNet normalisation  (mean/std per channel)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    # Tabular features — all in [0, 1] matching _normalise_tabular_for_inference.
    # Feature-specific plausible ranges:
    #   [0]  quality_pct_norm    [0, 1]
    #   [1]  volume_ratio        [0, 5]  → clip to [0, 1] after /5
    #   [2]  atr_pct             [0, 0.05]
    #   [3]  cvd_delta           [-1, 1] → normalised to [0, 1]
    #   [4]  nr7_flag            {0, 1}
    #   [5]  direction_flag      {0, 1}
    #   [6]  session_ordinal     [0, 1]
    #   [7]  london_overlap_flag {0, 1}
    #   [8]  or_range_atr_ratio  [0, 3]  → normalised
    #   [9]  premarket_range_ratio [0, 2] → normalised
    #   [10] bar_of_day          [0, 1]
    #   [11] day_of_week         [0, 1]
    #   [12] vwap_distance       [-3, 3] → normalised to [0, 1]
    #   [13] asset_class_id      [0, 1]
    #   [14] breakout_type_ord   [0, 1]
    #   [15] asset_volatility_class [0, 1]
    #   [16] hour_of_day         [0, 1]
    #   [17] tp3_atr_mult_norm   [0, 1]
    tab = rng.uniform(0.0, 1.0, (n, NUM_TABULAR)).astype(np.float32)
    # Binary flags — snap to 0 or 1
    for flag_idx in (4, 5, 7):
        tab[:, flag_idx] = (tab[:, flag_idx] > 0.5).astype(np.float32)

    return imgs, tab


def _load_pytorch_model(pt_path: Path, device: str) -> tuple[Any, Any]:
    """Load the .pt checkpoint and return (model, torch_module)."""
    import torch

    from lib.analysis.breakout_cnn import _build_model_from_checkpoint  # noqa: PLC0415

    info(f"Loading PyTorch model from {pt_path.name} ...")
    t0 = time.perf_counter()
    checkpoint = torch.load(str(pt_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model = _build_model_from_checkpoint(state_dict)  # type: ignore[arg-type]
    if model is None:
        raise RuntimeError(f"_build_model_from_checkpoint returned None for {pt_path}")
    model.eval()
    elapsed = time.perf_counter() - t0
    ok(f"PyTorch model loaded in {elapsed:.2f}s  (device={device})")
    return model, torch


def _load_onnx_session(onnx_path: Path) -> Any:
    """Load the ONNX model via onnxruntime and return the InferenceSession."""
    import onnxruntime as ort

    info(f"Loading ONNX model from {onnx_path.name} ...")
    t0 = time.perf_counter()

    # Use CPU provider — we only need correctness here, not speed.
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # suppress verbose ORT logging
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    elapsed = time.perf_counter() - t0
    ok(f"ONNX session loaded in {elapsed:.2f}s")

    # Print input/output names for debugging
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    info(f"  ONNX inputs:  {input_names}")
    info(f"  ONNX outputs: {output_names}")

    return session


def _pt_infer_batch(
    model: Any,
    torch_mod: Any,
    imgs: np.ndarray,
    tab: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run a batch through PyTorch and return raw logit probabilities [N]."""
    torch = torch_mod
    with torch.no_grad():
        img_t = torch.tensor(imgs, dtype=torch.float32).to(device)
        tab_t = torch.tensor(tab, dtype=torch.float32).to(device)
        logits = model(img_t, tab_t)
        # Model outputs raw logits (shape [N, 1] or [N, 2])
        probs = torch.sigmoid(logits).squeeze(-1) if logits.shape[-1] == 1 else torch.softmax(logits, dim=-1)[:, 1]
        return probs.cpu().numpy().astype(np.float32)


def _onnx_infer_batch(
    session: Any,
    imgs: np.ndarray,
    tab: np.ndarray,
) -> np.ndarray:
    """Run a batch through ONNX Runtime and return probabilities [N]."""
    import onnxruntime as ort  # noqa: F401

    input_names = [inp.name for inp in session.get_inputs()]

    # The ONNX export wraps the model in _ExportWrapper which concatenates
    # image and tabular inputs.  We need to match whatever input names
    # the export used.  Typically: ["image", "tabular"] or ["input_image", "input_tabular"].
    if len(input_names) == 2:
        feed = {input_names[0]: imgs, input_names[1]: tab}
    elif len(input_names) == 1:
        # Some exports concatenate everything — fall back to single-input mode.
        # This shouldn't happen for our HybridBreakoutCNN export.
        feed = {input_names[0]: imgs}
        warn(
            "ONNX model has only 1 input — expected 2 (image + tabular). "
            "Tabular features will NOT be used in ONNX inference."
        )
    else:
        raise ValueError(f"Unexpected number of ONNX inputs: {len(input_names)}")

    outputs = session.run(None, feed)
    raw = outputs[0]  # shape [N, 1] or [N, 2] or [N]

    if raw.ndim == 1:
        return raw.astype(np.float32)
    elif raw.shape[-1] == 1:
        # Raw logits — apply sigmoid
        return (1.0 / (1.0 + np.exp(-raw.squeeze(-1)))).astype(np.float32)
    else:
        # Softmax already applied (check the export wrapper)
        # If shape is [N, 2] the export likely already applied sigmoid/softmax
        if np.all((raw >= 0) & (raw <= 1)):
            return raw[:, 1].astype(np.float32)
        else:
            # Raw logits in [N, 2]
            shifted = raw - raw.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            softmax = exp / exp.sum(axis=-1, keepdims=True)
            return softmax[:, 1].astype(np.float32)


def _run_parity_check(
    pt_path: Path,
    onnx_path: Path,
    n_samples: int,
    threshold: float,
    verbose: bool,
    device: str,
) -> bool:
    """Run the full parity check.  Returns True on pass, False on failure."""
    rng = np.random.default_rng(SEED)

    # Load models
    try:
        model, torch_mod = _load_pytorch_model(pt_path, device)
    except Exception as exc:
        err(f"Failed to load PyTorch model: {exc}")
        return False

    try:
        onnx_session = _load_onnx_session(onnx_path)
    except Exception as exc:
        err(f"Failed to load ONNX model: {exc}")
        return False

    print()
    info(f"Running parity check: {n_samples} synthetic samples in batches of {BATCH_SIZE}")
    info(f"Parity threshold: max abs diff < {threshold:.0e}")
    print()

    all_pt_probs: list[np.ndarray] = []
    all_onnx_probs: list[np.ndarray] = []

    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_samples)
        n_this = end - start

        imgs, tab = _make_synthetic_batch(n_this, rng)

        # PyTorch inference
        t_pt0 = time.perf_counter()
        pt_probs = _pt_infer_batch(model, torch_mod, imgs, tab, device)
        t_pt = time.perf_counter() - t_pt0

        # ONNX inference
        t_onnx0 = time.perf_counter()
        onnx_probs = _onnx_infer_batch(onnx_session, imgs, tab)
        t_onnx = time.perf_counter() - t_onnx0

        all_pt_probs.append(pt_probs)
        all_onnx_probs.append(onnx_probs)

        if verbose:
            diff = np.abs(pt_probs - onnx_probs)
            print(
                f"  {DIM}batch {batch_idx + 1:2d}/{n_batches}  "
                f"n={n_this:3d}  "
                f"PT [{pt_probs.min():.4f}–{pt_probs.max():.4f}]  "
                f"ONNX [{onnx_probs.min():.4f}–{onnx_probs.max():.4f}]  "
                f"max_diff={diff.max():.2e}  "
                f"pt={t_pt * 1000:.1f}ms  onnx={t_onnx * 1000:.1f}ms"
                f"{NC}"
            )

    # Aggregate results
    pt_all = np.concatenate(all_pt_probs)
    onnx_all = np.concatenate(all_onnx_probs)
    abs_diff = np.abs(pt_all - onnx_all)

    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    p99_diff = float(np.percentile(abs_diff, 99))

    print()
    print("═" * 60)
    print("  Parity Report")
    print("═" * 60)
    print(f"  Samples       : {len(pt_all)}")
    print(f"  Max abs diff  : {max_diff:.6e}   {'✓' if max_diff < threshold else '✗ FAIL'}")
    print(f"  Mean abs diff : {mean_diff:.6e}")
    print(f"  P99 abs diff  : {p99_diff:.6e}")
    print(f"  PT  probs     : [{pt_all.min():.4f}–{pt_all.max():.4f}]  mean={pt_all.mean():.4f}")
    print(f"  ONNX probs    : [{onnx_all.min():.4f}–{onnx_all.max():.4f}]  mean={onnx_all.mean():.4f}")
    print("═" * 60)

    if max_diff < threshold:
        ok(f"PARITY PASSED — max diff {max_diff:.2e} < {threshold:.0e}")
        return True
    else:
        err(f"PARITY FAILED — max diff {max_diff:.2e} >= {threshold:.0e}")
        # Show the worst offenders
        worst_idxs = np.argsort(abs_diff)[-5:][::-1]
        print()
        print("  Top-5 divergent samples:")
        for i, idx in enumerate(worst_idxs):
            print(
                f"    [{i + 1}] sample={idx:4d}  "
                f"PT={pt_all[idx]:.6f}  "
                f"ONNX={onnx_all[idx]:.6f}  "
                f"diff={abs_diff[idx]:.2e}"
            )
        return False


def _validate_contract(models_dir: Path, verbose: bool) -> bool:
    """Load feature_contract.json and verify it matches TABULAR_FEATURES."""
    import json

    contract_path = models_dir / CONTRACT_FILENAME
    if not contract_path.exists():
        warn(f"{CONTRACT_FILENAME} not found — skipping contract validation")
        return True

    try:
        with open(contract_path) as f:
            contract = json.load(f)
    except Exception as exc:
        warn(f"Could not read {CONTRACT_FILENAME}: {exc}")
        return True

    version = contract.get("version", "?")
    features = contract.get("tabular_features", [])
    n_features = len(features)

    info(f"Feature contract: version={version}, features={n_features}")

    if n_features != NUM_TABULAR:
        err(f"Contract has {n_features} features but expected {NUM_TABULAR} (v6)")
        return False

    if verbose:
        try:
            from lib.analysis.breakout_cnn import TABULAR_FEATURES  # noqa: PLC0415

            mismatches = []
            for i, (cf, pf) in enumerate(zip(features, TABULAR_FEATURES, strict=True)):
                if cf != pf:
                    mismatches.append((i, cf, pf))
            if mismatches:
                err("Feature name mismatches between contract and Python:")
                for idx, cf, pf in mismatches:
                    err(f"  [{idx}] contract='{cf}' != python='{pf}'")
                return False
            else:
                ok(f"All {NUM_TABULAR} feature names match Python TABULAR_FEATURES")
        except ImportError:
            warn("Could not import TABULAR_FEATURES — skipping name check")

    ok(f"Feature contract v{version}: {n_features} features ✓")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate ONNX ↔ PyTorch inference parity for the champion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing .pt, .onnx, and feature_contract.json (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic samples to test (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=PARITY_THRESHOLD,
        help=f"Max allowed absolute difference (default: {PARITY_THRESHOLD:.0e})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device to use for PyTorch inference (default: auto)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-batch statistics during the parity check",
    )
    parser.add_argument(
        "--skip-contract",
        action="store_true",
        help="Skip feature_contract.json validation",
    )
    args = parser.parse_args()

    print()
    print("═" * 60)
    print("  ONNX ↔ PyTorch Parity Check")
    print("═" * 60)
    print()

    # Dependency check
    if not _check_deps():
        return 1

    # Resolve device
    if args.device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device
    info(f"Using device: {device}")
    print()

    models_dir: Path = args.models_dir
    pt_path = models_dir / PT_FILENAME
    onnx_path = models_dir / ONNX_FILENAME

    # Verify files exist
    missing_files = []
    if not pt_path.exists():
        missing_files.append(str(pt_path))
    if not onnx_path.exists():
        missing_files.append(str(onnx_path))

    if missing_files:
        for f in missing_files:
            err(f"Model file not found: {f}")
        err("Run training first or copy the champion model files into the models/ directory.")
        return 1

    info(f"PT   model : {pt_path}")
    info(f"ONNX model : {onnx_path}")
    print()

    # Feature contract validation
    if not args.skip_contract:
        contract_ok = _validate_contract(models_dir, args.verbose)
        print()
        if not contract_ok:
            return 1

    # Main parity check
    passed = _run_parity_check(
        pt_path=pt_path,
        onnx_path=onnx_path,
        n_samples=args.n_samples,
        threshold=args.threshold,
        verbose=args.verbose,
        device=device,
    )
    print()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
