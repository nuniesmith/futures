# CNN Breakout Detection Pipeline

## Overview

The breakout CNN is a hybrid **EfficientNetV2-S + tabular** model that classifies
Ruby-style chart snapshots as "good breakout" (high follow-through probability)
or "bad breakout" (likely to chop/fail).  It runs as an advisory or hard gate
on ORB signals during the active trading session (03:00–12:00 ET).

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CNN Pipeline Lifecycle                          │
│                                                                     │
│  OFF-HOURS (12:00–00:00 ET)          PRE-MARKET (00:00–03:00 ET)   │
│  ┌──────────────┐                    ┌─────────────────────┐       │
│  │ 1. Generate   │  historical bars  │ 4. Morning briefing │       │
│  │    dataset    │──────────────────▶│    includes model   │       │
│  │    images     │  render charts    │    health check     │       │
│  └──────┬───────┘                    └─────────────────────┘       │
│         │                                                           │
│  ┌──────▼───────┐                    ACTIVE (03:00–12:00 ET)       │
│  │ 2. Train on  │  GPU (RTX 2070)   ┌─────────────────────┐       │
│  │    GPU with   │──────────────────▶│ 5. Live inference   │       │
│  │    val gate   │  promote if pass  │    predict_breakout │       │
│  └──────┬───────┘                    │    on each ORB      │       │
│         │                            │    signal            │       │
│  ┌──────▼───────┐                    └─────────────────────┘       │
│  │ 3. Promote   │                                                   │
│  │    champion   │  breakout_cnn_best.pt                            │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Image Generation Philosophy

**Chart images are ephemeral, generated artifacts — not static assets.**

They are rendered on-demand from historical bar data by the dataset generator
(`lib/analysis/dataset_generator.py`) and should never be committed to git.
The reasoning:

| Concern | Approach |
|---|---|
| **Reproducibility** | Given the same bars + config, the renderer produces identical PNGs. Store the *recipe* (code + bar data), not the output. |
| **Freshness** | Each retraining cycle should include the *latest* market data. Stale images from weeks ago hurt model generalization. |
| **Disk/Git bloat** | 3,866 images = 650 MB. Every retrain cycle generates new ones. Committing them would add hundreds of MB per week to git history. |
| **Train/Val/Live parity** | All images — training, validation, and live inference — go through the same `render_snapshot_for_inference()` pipeline, ensuring consistency. |

### How images flow through the system

```
Historical bars (Redis/Postgres/CSV)
        │
        ▼
┌───────────────────┐     Off-hours: dataset_generator.py
│  generate_dataset │     Renders chart snapshots for every ORB
│  for_symbol()     │     simulation window across 90 days of bars
└───────┬───────────┘
        │
        ▼
  dataset/images/*.png    ← .gitignored, lives only on disk
  dataset/labels.csv      ← .gitignored, regenerated each cycle
        │
        ▼
┌───────────────────┐
│  split_dataset()  │     Stratified 85/15 train/val split
└───────┬───────────┘
        │
        ├──▶ dataset/train.csv
        └──▶ dataset/val.csv
                │
                ▼
┌───────────────────┐     GPU training (train_gpu.py)
│  BreakoutDataset  │     Loads images + tabular features
│  + DataLoader     │     Applies augmentations (train only)
└───────┬───────────┘
        │
        ▼
  models/breakout_cnn_best.pt   ← promoted champion (Git LFS)
```

### Live inference images

During the active session, when an ORB breakout is detected, the engine:

1. Renders a chart snapshot from the *current* live bars using the same
   `render_snapshot_for_inference()` function
2. Feeds it to `predict_breakout()` alongside the tabular features
3. Gets a probability score (0.0–1.0)
4. Applies the confidence threshold (default 0.82) to gate the signal

This means train/val/live images are all produced by the **exact same renderer**,
eliminating train-serve skew.

---

## Git & LFS Setup

### What's tracked in git

| Path | Tracked? | Method | Why |
|---|---|---|---|
| `models/breakout_cnn_best.pt` | ✅ | Git LFS | The live champion model — needed after clone |
| `models/breakout_cnn_best_meta.json` | ✅ | Regular git | Tiny JSON with promotion metadata |
| `models/*.pt` (all others) | ❌ | .gitignored | Training checkpoints — ephemeral |
| `models/archive/` | ❌ | .gitignored | Archived previous champions — local only |
| `models/training_history.csv` | ❌ | .gitignored | Per-epoch metrics — regenerated each run |
| `models/retrain_audit.jsonl` | ❌ | .gitignored | Audit log — local operational data |
| `dataset/images/**` | ❌ | .gitignored | Generated chart PNGs — regenerated from bars |
| `dataset/labels.csv` | ❌ | .gitignored | Generated manifest — regenerated each cycle |
| `dataset/train.csv` | ❌ | .gitignored | Split output — regenerated each cycle |
| `dataset/val.csv` | ❌ | .gitignored | Split output — regenerated each cycle |
| `dataset/.gitkeep` | ✅ | Regular git | Ensures directory exists after clone |
| `models/.gitkeep` | ✅ | Regular git | Ensures directory exists after clone |

### .gitattributes (LFS tracking)

```
*.pt filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.torchscript filter=lfs diff=lfs merge=lfs -text
```

### Initial setup (one time, after cloning)

```bash
# 1. Install Git LFS (if not already installed)
#    Windows: winget install GitHub.GitLFS
#    macOS:   brew install git-lfs
#    Linux:   sudo apt install git-lfs

# 2. Pull LFS objects (downloads breakout_cnn_best.pt)
git lfs pull

# 3. Verify
git lfs ls-files
# Should show: breakout_cnn_best.pt
```

### Migration from the old setup

If your repo already has images and checkpoints committed directly to git,
run the migration script to clean up:

```bash
cd futures
bash scripts/migrate_git_lfs.sh
```

This will:
1. Remove `dataset/images/` and dated `models/*.pt` from the git index
2. Re-add only `breakout_cnn_best.pt` under Git LFS
3. Optionally rewrite history to purge old blobs (saves ~3 GB)

After migration, push with:
```bash
git push --force-with-lease origin main
```

---

## Retraining Workflow

### Automatic (engine scheduler)

The engine's `ScheduleManager` runs two actions during off-hours:

1. **`GENERATE_CHART_DATASET`** (priority 4) — Calls `dataset_generator.generate_dataset()`
   to render new chart images from the last 90 days of bars
2. **`TRAIN_BREAKOUT_CNN`** (priority 5) — Calls the overnight retraining pipeline
   which handles training, validation, and promotion

These run automatically once per off-hours session (12:00–00:00 ET). The
training action only fires after dataset generation completes.

### Manual (CLI)

#### Quick retrain on existing data

```bash
cd futures
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py \
    --skip-dataset --immediate
```

#### Full pipeline (generate + train + validate + promote)

```bash
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py --immediate
```

#### Dry run (train + validate, but don't promote)

```bash
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py \
    --immediate --dry-run
```

#### Custom parameters

```bash
PYTHONPATH=src .venv/bin/python scripts/retrain_overnight.py \
    --immediate \
    --epochs 30 \
    --batch-size 64 \
    --lr 2e-4 \
    --patience 10 \
    --min-accuracy 82.0 \
    --min-precision 78.0 \
    --symbols MGC,MES,MNQ,M2K
```

### Pipeline stages in detail

```
Stage 1: Dataset Refresh
  └─ generate_dataset() renders chart images from historical bars
  └─ Appends new images to dataset/images/, updates labels.csv
  └─ Skips images that already exist on disk (incremental)

Stage 2: Train/Val Split
  └─ split_dataset() does stratified 85/15 split on labels.csv
  └─ Writes dataset/train.csv and dataset/val.csv
  └─ Maintains label distribution (good_long, good_short, bad_long, bad_short)

Stage 3: GPU Training
  └─ EfficientNetV2-S backbone + tabular head
  └─ Mixed precision (AMP) on RTX 2070 SUPER (~54s/epoch)
  └─ Class-weighted loss, cosine schedule, warmup, early stopping
  └─ Produces candidate model: models/breakout_cnn_YYYYMMDD_acc*.pt

Stage 4: Validation Gate
  └─ Evaluates candidate on val set (accuracy, precision, recall)
  └─ Evaluates current champion for comparison
  └─ Gate thresholds (configurable via env vars):
       CNN_RETRAIN_MIN_ACC=80.0        (minimum accuracy)
       CNN_RETRAIN_MIN_PRECISION=75.0  (minimum precision)
       CNN_RETRAIN_MIN_RECALL=70.0     (minimum recall)
       CNN_RETRAIN_IMPROVEMENT=0.0     (improvement over champion)
  └─ Candidate MUST pass all thresholds to be promoted

Stage 5: Model Promotion
  └─ Archives current champion to models/archive/
  └─ Copies candidate → models/breakout_cnn_best.pt (atomic)
  └─ Writes models/breakout_cnn_best_meta.json with metrics
  └─ Invalidates cached model in inference module

Stage 6: Cleanup
  └─ Prunes archived champions (keeps 10 most recent)
  └─ Prunes old training checkpoints (keeps 6 most recent)
```

### Validation gate rationale

The gate prevents model regressions. A new model is only promoted if it:

- Meets **absolute accuracy** ≥ 80% — ensures the model is useful
- Meets **precision** ≥ 75% — few false positives (don't approve bad trades)
- Meets **recall** ≥ 70% — catches most good setups
- Beats the champion accuracy by the configured improvement margin

If the gate rejects a candidate, the champion remains untouched and the
rejection is logged in `models/retrain_audit.jsonl` for review.

---

## Fresh Clone Setup

After cloning the repo on a new machine:

```bash
# 1. Clone and enter
git clone <repo-url> futures
cd futures

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install CUDA PyTorch (for GPU training)
pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 5. Pull LFS objects (downloads the champion model)
git lfs pull

# 6. Verify champion model exists
ls -la models/breakout_cnn_best.pt

# 7. (Optional) Generate dataset for local training
PYTHONPATH=src python -m lib.analysis.dataset_generator \
    --symbols MGC,MES,MNQ --days-back 90

# 8. (Optional) Run a training cycle
PYTHONPATH=src python scripts/retrain_overnight.py \
    --skip-dataset --immediate
```

At this point:
- **Inference works** immediately — the champion model was pulled via LFS
- **Training works** after step 7 generates local images from your bar data
- **Images are never committed** — each machine generates its own from bars

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CNN_RETRAIN_SYMBOLS` | `MGC,MES,MNQ` | Symbols for dataset generation |
| `CNN_RETRAIN_DAYS_BACK` | `90` | Days of history to process |
| `CNN_RETRAIN_EPOCHS` | `25` | Training epochs |
| `CNN_RETRAIN_BATCH_SIZE` | `64` | Batch size (64 fits 8 GB VRAM) |
| `CNN_RETRAIN_LR` | `2e-4` | Peak learning rate |
| `CNN_RETRAIN_PATIENCE` | `8` | Early stopping patience |
| `CNN_RETRAIN_MIN_ACC` | `80.0` | Minimum val accuracy to promote |
| `CNN_RETRAIN_MIN_PRECISION` | `75.0` | Minimum precision to promote |
| `CNN_RETRAIN_MIN_RECALL` | `70.0` | Minimum recall to promote |
| `CNN_RETRAIN_IMPROVEMENT` | `0.0` | Required accuracy gain over champion |
| `ORB_CNN_GATE` | `0` | `0` = advisory mode, `1` = hard gate |

---

## File Reference

| File | Purpose |
|---|---|
| `src/lib/analysis/breakout_cnn.py` | Model definition, training, inference API |
| `src/lib/analysis/dataset_generator.py` | Chart image generation from historical bars |
| `src/lib/analysis/chart_renderer.py` | Ruby-style chart rendering engine |
| `src/lib/services/engine/scheduler.py` | Session-aware scheduling (off-hours triggers) |
| `src/lib/services/engine/main.py` | Engine action handlers (dataset gen + training) |
| `scripts/train_gpu.py` | GPU-optimised standalone training script |
| `scripts/retrain_overnight.py` | Full retraining pipeline orchestrator |
| `scripts/migrate_git_lfs.sh` | One-time migration to Git LFS |
| `models/breakout_cnn_best.pt` | Live champion model (Git LFS) |
| `models/breakout_cnn_best_meta.json` | Promotion metadata |

---

## Troubleshooting

### "No champion model found" after clone

```bash
git lfs pull
# If that fails, check: git lfs install
```

### Images not generating

```bash
# Check bar data is available
PYTHONPATH=src python -c "
from lib.analysis.dataset_generator import load_bars
df = load_bars('MES', source='cache', days=5)
print(f'Bars: {len(df) if df is not None else 0}')
"
```

If no bars, the dataset generator has nothing to render from. Ensure your
data source (Redis cache, Postgres, or CSV files in `data/bars/`) is populated.

### Training runs but accuracy is low

- Check dataset balance: `wc -l dataset/train.csv` (need 500+ rows minimum)
- Check label distribution: `cut -d, -f2 dataset/labels.csv | sort | uniq -c`
- Try more epochs: `--epochs 40 --patience 12`
- Check for corrupted images: the training script logs NaN/Inf loss warnings

### Model promoted but inference seems wrong

```bash
# Check which model is loaded
PYTHONPATH=src python -c "
from lib.analysis.breakout_cnn import model_info
print(model_info('models/breakout_cnn_best.pt'))
"
```

Look at the metadata in `models/breakout_cnn_best_meta.json` to see when it
was promoted and what its validation metrics were.