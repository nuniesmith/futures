#!/usr/bin/env python3
"""
export_onnx.py — Export HybridBreakoutCNN to ONNX for NinjaTrader 8 inference
===============================================================================

Converts a trained HybridBreakoutCNN .pt checkpoint to an ONNX model that
OrbCnnPredictor.cs can load directly inside NinjaTrader 8.

The exported model has exactly two inputs (matching OrbCnnPredictor.cs):
    "image"   : float32 [batch, 3, 224, 224]   (ImageNet-normalised RGB)
    "tabular" : float32 [batch, 8]              (see TABULAR_FEATURES)

And one output:
    "output"  : float32 [batch, 2]             (logits for [bad, good])

Usage
-----
    # Export the latest model from models/ with default settings:
    python scripts/export_onnx.py

    # Export a specific checkpoint:
    python scripts/export_onnx.py --model models/breakout_cnn_v3_acc0.847.pt

    # Export to a custom location:
    python scripts/export_onnx.py --output C:/NT8/Models/orb_breakout_cnn.onnx

    # Export with ONNX opset 17 and run a quick self-test:
    python scripts/export_onnx.py --opset 17 --verify

    # Export and immediately copy to the NT8 bin/Custom folder:
    python scripts/export_onnx.py --copy-to "C:/Users/you/Documents/NinjaTrader 8/bin/Custom"

After export
------------
Copy the following files to Documents\\NinjaTrader 8\\bin\\Custom\\ :
    orb_breakout_cnn.onnx              (exported by this script)
    Microsoft.ML.OnnxRuntime.dll       (from your OrbCnnPredictor project build)
    onnxruntime.dll                    (NuGet cache: Microsoft.ML.OnnxRuntime/runtimes/win-x64/native/)
    onnxruntime_providers_shared.dll   (same NuGet native folder)

The .onnx file can also live anywhere on disk — set the CnnModelPath property
in BreakoutStrategy to point to it.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure project src/ is on the import path regardless of where
# the script is invoked from.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC = _PROJECT_ROOT / "src"

for _p in (_SRC, _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_onnx")

# ---------------------------------------------------------------------------
# Imports (guarded — give a clear error if torch is missing)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
except ImportError:
    log.error("PyTorch is not installed.  Install with:\n    pip install torch torchvision\nthen re-run this script.")
    sys.exit(1)

try:
    import onnx
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False
    log.warning(
        "onnx / onnxruntime not installed — verification will be skipped.\nInstall with:  pip install onnx onnxruntime"
    )

from lib.analysis.breakout_cnn import (
    HybridBreakoutCNN,
    NUM_TABULAR,
    IMAGE_SIZE,
    TABULAR_FEATURES,
    _find_best_model,
    DEFAULT_MODEL_DIR,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OPSET = 17
DEFAULT_OUTPUT_NAME = "orb_breakout_cnn.onnx"
DEFAULT_OUTPUT_FOLDER = _PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(pt_path: str) -> HybridBreakoutCNN:
    """Load a .pt state-dict checkpoint into a HybridBreakoutCNN instance."""
    log.info("Loading checkpoint: %s", pt_path)

    device = torch.device("cpu")  # always export from CPU for portability

    model = HybridBreakoutCNN(pretrained=False)

    state = torch.load(pt_path, map_location=device, weights_only=True)

    # Support both bare state-dicts and wrapped checkpoints
    # (some training runs save {"model_state_dict": ..., "epoch": ..., ...})
    if isinstance(state, dict) and "model_state_dict" in state:
        log.info("  Wrapped checkpoint detected — extracting model_state_dict")
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        log.info("  Wrapped checkpoint detected — extracting state_dict")
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    model = model.to(device)

    log.info("  Loaded OK — num_tabular=%d, dropout=%.2f", model.num_tabular, model.classifier[2].p)
    return model


def _build_dummy_inputs(batch_size: int = 1):
    """Return (image_tensor, tabular_tensor) dummy inputs for tracing."""
    img = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    tab = torch.randn(batch_size, NUM_TABULAR)
    return img, tab


def _export(
    model: HybridBreakoutCNN,
    output_path: str,
    opset: int,
) -> None:
    """Run torch.onnx.export with the correct input/output names."""

    img_dummy, tab_dummy = _build_dummy_inputs(batch_size=1)

    log.info("Exporting to ONNX (opset %d) → %s", opset, output_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        model,
        args=(img_dummy, tab_dummy),
        f=output_path,
        # Input names must match what OrbCnnPredictor.cs reads from
        # _session.InputMetadata — the C# code uses positional order
        # (index 0 = image, index 1 = tabular) but also logs names.
        input_names=["image", "tabular"],
        output_names=["output"],
        # Dynamic batch axis so OrbCnnPredictor can pass batch_size=1
        # and onnxruntime won't complain about fixed shapes.
        dynamic_axes={
            "image": {0: "batch_size"},
            "tabular": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        export_params=True,
        do_constant_folding=True,
        opset_version=opset,
        # Verbose=False keeps the log clean; flip to True for debugging
        verbose=False,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info("  Export complete — %.1f MB", size_mb)


def _verify_onnx_model(onnx_path: str) -> None:
    """Check the ONNX model graph with onnx.checker and print metadata."""
    if not _ORT_AVAILABLE:
        log.warning("Skipping verification — onnx/onnxruntime not installed")
        return

    log.info("Verifying ONNX model graph…")
    model_proto = onnx.load(onnx_path)

    try:
        onnx.checker.check_model(model_proto)
        log.info("  onnx.checker: PASSED")
    except onnx.checker.ValidationError as e:
        log.error("  onnx.checker FAILED: %s", e)
        raise

    # Print input / output metadata so the user can cross-check against
    # the C# predictor.
    log.info("  Inputs:")
    for inp in model_proto.graph.input:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in inp.type.tensor_type.shape.dim]
        log.info("    %-12s  shape=%s", inp.name, shape)

    log.info("  Outputs:")
    for out in model_proto.graph.output:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in out.type.tensor_type.shape.dim]
        log.info("    %-12s  shape=%s", out.name, shape)


def _verify_inference(
    onnx_path: str,
    torch_model: HybridBreakoutCNN,
) -> None:
    """
    Run a dummy forward pass through both the PyTorch model and ONNX Runtime,
    then compare outputs to make sure they agree to within floating-point tolerance.
    """
    if not _ORT_AVAILABLE:
        log.warning("Skipping inference verification — onnxruntime not installed")
        return

    log.info("Running inference verification (PyTorch vs ONNX Runtime)…")

    import numpy as np

    img_t, tab_t = _build_dummy_inputs(batch_size=1)

    # PyTorch reference output
    with torch.no_grad():
        pt_logits = torch_model(img_t, tab_t).numpy()

    # ONNX Runtime output
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "image": img_t.numpy(),
        "tabular": tab_t.numpy(),
    }
    ort_logits = sess.run(["output"], ort_inputs)[0]

    # Compare
    max_diff = float(np.abs(pt_logits - ort_logits).max())
    rtol = 1e-4
    atol = 1e-4

    if max_diff <= atol:
        log.info("  Max absolute diff: %.2e  ✓  (within tolerance %.2e)", max_diff, atol)
    else:
        log.warning(
            "  Max absolute diff: %.2e  ⚠  (tolerance %.2e) — "
            "outputs differ more than expected; try a lower opset or check model.",
            max_diff,
            atol,
        )

    # Also check softmax probabilities to make sure the model output is sane
    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    ort_probs = _softmax(ort_logits)
    log.info(
        "  ORT probs: bad=%.4f  good=%.4f  (P(good breakout)=%.4f)",
        ort_probs[0, 0],
        ort_probs[0, 1],
        ort_probs[0, 1],
    )
    log.info("  Inference verification complete")


def _benchmark(onnx_path: str, n_runs: int = 100) -> None:
    """Print average inference latency for a single sample."""
    if not _ORT_AVAILABLE:
        return

    import numpy as np

    log.info("Benchmarking ONNX Runtime (%d runs, batch_size=1)…", n_runs)

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 2
    sess = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

    img_np = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    tab_np = np.random.randn(1, NUM_TABULAR).astype(np.float32)
    inputs = {"image": img_np, "tabular": tab_np}

    # Warmup
    for _ in range(5):
        sess.run(["output"], inputs)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(["output"], inputs)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms

    log.info("  Average latency: %.2f ms / inference  (CPU, 2 threads)", elapsed)

    if elapsed > 5.0:
        log.warning(
            "  Latency %.2f ms is higher than the 5 ms target for real-time use.\n"
            "  Consider enabling CUDA (--cuda) or reducing model complexity.",
            elapsed,
        )


def _print_tabular_feature_summary() -> None:
    """Print the ordered feature list so the user can cross-check C# code."""
    log.info("Tabular features (must match OrbCnnPredictor.NormaliseTabular order):")
    for i, name in enumerate(TABULAR_FEATURES):
        log.info("  [%d] %s", i, name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export HybridBreakoutCNN to ONNX for NinjaTrader 8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--model",
        "-m",
        default=None,
        metavar="PATH",
        help=(f"Path to a .pt checkpoint.  Default: best model found in {DEFAULT_MODEL_DIR}/"),
    )

    p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help=(f"Output .onnx file path.  Default: {DEFAULT_OUTPUT_FOLDER}/{DEFAULT_OUTPUT_NAME}"),
    )

    p.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        metavar="N",
        help=f"ONNX opset version (default: {DEFAULT_OPSET}).  11–17 are safe.",
    )

    p.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help=("Run onnx.checker + OnnxRuntime verification after export (default: on).  Use --no-verify to skip."),
    )
    p.add_argument("--no-verify", dest="verify", action="store_false")

    p.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Benchmark ONNX Runtime inference latency after export.",
    )

    p.add_argument(
        "--copy-to",
        default=None,
        metavar="DIR",
        help=(
            "After export, copy the .onnx file to this directory "
            r"(e.g. 'C:\Users\you\Documents\NinjaTrader 8\bin\Custom')."
        ),
    )

    p.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Export and verify using CUDA (requires onnxruntime-gpu).",
    )

    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Reduce log verbosity.",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # ── Resolve model path ────────────────────────────────────────────────────
    pt_path = args.model
    if pt_path is None:
        pt_path = _find_best_model(str(DEFAULT_MODEL_DIR))
        if pt_path is None:
            log.error(
                "No trained model found in %s/\n"
                "Train a model first with:  python scripts/train_gpu.py\n"
                "Or specify one with:       --model path/to/model.pt",
                DEFAULT_MODEL_DIR,
            )
            return 1
        log.info("Auto-selected model: %s", pt_path)

    if not os.path.isfile(pt_path):
        log.error("Checkpoint not found: %s", pt_path)
        return 1

    # ── Resolve output path ───────────────────────────────────────────────────
    if args.output:
        out_path = str(args.output)
        # If the user gave a directory, append the default filename
        if os.path.isdir(out_path):
            out_path = os.path.join(out_path, DEFAULT_OUTPUT_NAME)
    else:
        out_path = str(DEFAULT_OUTPUT_FOLDER / DEFAULT_OUTPUT_NAME)

    # ── Print feature summary ─────────────────────────────────────────────────
    _print_tabular_feature_summary()

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = _load_checkpoint(pt_path)
    except Exception as e:
        log.error("Failed to load checkpoint: %s", e, exc_info=True)
        return 1

    # ── Export ────────────────────────────────────────────────────────────────
    try:
        _export(model, out_path, opset=args.opset)
    except Exception as e:
        log.error("Export failed: %s", e, exc_info=True)
        return 1

    # ── Verify ────────────────────────────────────────────────────────────────
    if args.verify:
        try:
            _verify_onnx_model(out_path)
            _verify_inference(out_path, model)
        except Exception as e:
            log.error("Verification failed: %s", e, exc_info=True)
            return 1

    # ── Benchmark ─────────────────────────────────────────────────────────────
    if args.benchmark:
        try:
            _benchmark(out_path)
        except Exception as e:
            log.warning("Benchmark failed: %s", e)

    # ── Optional copy ─────────────────────────────────────────────────────────
    if args.copy_to:
        dest_dir = args.copy_to
        os.makedirs(dest_dir, exist_ok=True)
        dest_file = os.path.join(dest_dir, os.path.basename(out_path))
        try:
            shutil.copy2(out_path, dest_file)
            log.info("Copied → %s", dest_file)
        except Exception as e:
            log.warning("Copy failed: %s", e)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 60)
    log.info("Export complete.")
    log.info("  Source checkpoint : %s", pt_path)
    log.info("  ONNX model        : %s", os.path.abspath(out_path))
    log.info("  Opset             : %d", args.opset)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Copy %s to a stable location", os.path.basename(out_path))
    log.info("     e.g.  C:\\NT8\\Models\\orb_breakout_cnn.onnx")
    log.info("  2. Copy OnnxRuntime DLLs to NinjaTrader 8\\bin\\Custom\\")
    log.info("     (Microsoft.ML.OnnxRuntime.dll, onnxruntime.dll,")
    log.info("      onnxruntime_providers_shared.dll)")
    log.info("  3. In BreakoutStrategy → Group 6 AI Filter:")
    log.info("     CnnModelPath   = C:\\NT8\\Models\\orb_breakout_cnn.onnx")
    log.info("     CnnThreshold   = 0.82  (or use per-session defaults)")
    log.info("     EnableCnnFilter = true")
    log.info("═" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
