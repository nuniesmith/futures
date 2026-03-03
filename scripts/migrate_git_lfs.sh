#!/usr/bin/env bash
# =============================================================================
# migrate_git_lfs.sh — One-Time Git LFS Migration
# =============================================================================
#
# Removes bloated binary files (3,866 PNGs + 40 model checkpoints ≈ 3.7 GB)
# from Git history and re-adds only the promoted champion model under Git LFS.
#
# After this migration:
#   TRACKED (LFS):  models/breakout_cnn_best.pt
#                    models/breakout_cnn_best_meta.json
#   IGNORED:         dataset/images/**  (regenerated from bar data)
#                    models/breakout_cnn_*.pt  (training checkpoints)
#                    models/archive/  models/training_history.csv  etc.
#
# Prerequisites:
#   - git lfs installed (git lfs version)
#   - .gitignore and .gitattributes already updated (the companion commit)
#   - Working tree is clean (commit or stash pending work first)
#   - BACK UP your repo before running (git clone --mirror as safety net)
#
# Usage:
#   cd futures
#   bash scripts/migrate_git_lfs.sh
#
# What this script does (step by step):
#   1. Validates prerequisites (git lfs, clean tree, correct directory)
#   2. Initialises Git LFS in the repo
#   3. Removes dataset/images/ and dated model checkpoints from the index
#   4. Commits the removal
#   5. Re-adds ONLY the champion model (breakout_cnn_best.pt) via LFS
#   6. Commits the LFS-tracked champion
#   7. Uses git-filter-repo (or BFG) to purge the old blobs from history
#   8. Runs git gc to reclaim disk space
#   9. Prints before/after repo size
#
# ⚠️  WARNING: This rewrites Git history. Coordinate with any collaborators.
#     After running, everyone must re-clone or run:
#       git fetch origin && git reset --hard origin/main
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No colour

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
step()  { printf "\n${BOLD}━━━ Step %s ━━━${NC}\n" "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
step "0/8: Pre-flight checks"

# Must be run from project root (where .git lives)
if [ ! -d ".git" ]; then
    err "Run this script from the project root (where .git/ lives)"
    err "  cd futures && bash scripts/migrate_git_lfs.sh"
    exit 1
fi

# Git LFS must be installed
if ! command -v git-lfs &>/dev/null && ! git lfs version &>/dev/null; then
    err "git-lfs is not installed. Install it first:"
    err "  Windows: winget install GitHub.GitLFS"
    err "  macOS:   brew install git-lfs"
    err "  Linux:   sudo apt install git-lfs"
    exit 1
fi
ok "git-lfs found: $(git lfs version 2>/dev/null | head -1)"

# Working tree should be clean (aside from the .gitignore/.gitattributes changes
# we're about to use)
DIRTY=$(git status --porcelain -- ':!.gitignore' ':!.gitattributes' ':!scripts/migrate_git_lfs.sh' ':!dataset/.gitkeep' ':!models/.gitkeep' ':!scripts/retrain_overnight.py' ':!src/lib/services/engine/main.py' 2>/dev/null | head -5)
if [ -n "$DIRTY" ]; then
    warn "Working tree has uncommitted changes (beyond expected migration files):"
    echo "$DIRTY"
    printf "\n${YELLOW}Continue anyway? (y/N): ${NC}"
    read -r answer
    if [ "$answer" != "y" ] && [ "$answer" != "Y" ]; then
        info "Aborted. Commit or stash your changes first."
        exit 1
    fi
fi

# Confirm the user wants to proceed (history rewrite is destructive)
printf "\n"
warn "This script will:"
warn "  1. Remove ~3,866 images + ~40 model checkpoints from git tracking"
warn "  2. Re-add only breakout_cnn_best.pt under Git LFS"
warn "  3. Optionally rewrite git history to purge old blobs (~3.4 GB)"
warn ""
warn "History rewrite requires all collaborators to re-clone."
printf "\n${YELLOW}Proceed with migration? (y/N): ${NC}"
read -r confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    info "Aborted."
    exit 0
fi

# Record starting size
STARTING_SIZE=$(du -sh .git/ 2>/dev/null | cut -f1)
info "Current .git/ size: ${STARTING_SIZE}"

# ---------------------------------------------------------------------------
# Step 1: Initialise Git LFS
# ---------------------------------------------------------------------------
step "1/8: Initialise Git LFS"

git lfs install --local
ok "Git LFS initialised for this repository"

# Verify .gitattributes has the LFS track rules
if grep -q "filter=lfs" .gitattributes 2>/dev/null; then
    ok ".gitattributes already has LFS tracking rules"
else
    warn ".gitattributes missing LFS rules — adding *.pt tracking"
    echo '*.pt filter=lfs diff=lfs merge=lfs -text' >> .gitattributes
fi

# ---------------------------------------------------------------------------
# Step 2: Back up the champion model before we touch anything
# ---------------------------------------------------------------------------
step "2/8: Back up champion model"

CHAMPION="models/breakout_cnn_best.pt"
CHAMPION_META="models/breakout_cnn_best_meta.json"
BACKUP_DIR=".migration_backup"

mkdir -p "$BACKUP_DIR"

if [ -f "$CHAMPION" ]; then
    cp "$CHAMPION" "$BACKUP_DIR/breakout_cnn_best.pt"
    ok "Champion model backed up to $BACKUP_DIR/"
    CHAMPION_SIZE=$(du -sh "$CHAMPION" | cut -f1)
    info "Champion size: $CHAMPION_SIZE"
else
    warn "No champion model found at $CHAMPION — will skip LFS re-add"
fi

if [ -f "$CHAMPION_META" ]; then
    cp "$CHAMPION_META" "$BACKUP_DIR/breakout_cnn_best_meta.json"
    ok "Champion metadata backed up"
fi

# ---------------------------------------------------------------------------
# Step 3: Remove images from git index (keep files on disk)
# ---------------------------------------------------------------------------
step "3/8: Remove dataset images from git tracking"

IMAGE_COUNT=$(git ls-files dataset/images/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$IMAGE_COUNT" -gt 0 ]; then
    info "Removing $IMAGE_COUNT tracked images from git index..."
    git rm -r --cached --quiet dataset/images/ 2>/dev/null || true
    ok "Removed $IMAGE_COUNT images from index (files still on disk)"
else
    info "No images tracked in git — nothing to remove"
fi

# Also remove dataset CSVs (they're regenerated)
for csv_file in dataset/labels.csv dataset/train.csv dataset/val.csv dataset/dataset_stats.json; do
    if git ls-files --error-unmatch "$csv_file" &>/dev/null 2>&1; then
        git rm --cached --quiet "$csv_file" 2>/dev/null || true
        info "Removed $csv_file from index"
    fi
done

# ---------------------------------------------------------------------------
# Step 4: Remove all model checkpoints from git index
# ---------------------------------------------------------------------------
step "4/8: Remove model checkpoints from git tracking"

MODEL_COUNT=$(git ls-files models/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -gt 0 ]; then
    info "Removing $MODEL_COUNT tracked model files from git index..."
    git rm -r --cached --quiet models/ 2>/dev/null || true
    ok "Removed $MODEL_COUNT model files from index (files still on disk)"
else
    info "No model files tracked in git — nothing to remove"
fi

# ---------------------------------------------------------------------------
# Step 5: Commit the removal
# ---------------------------------------------------------------------------
step "5/8: Commit removal of binary blobs"

# Stage the updated .gitignore, .gitattributes, and .gitkeep files
git add .gitignore .gitattributes
git add dataset/.gitkeep 2>/dev/null || true
git add models/.gitkeep 2>/dev/null || true

# Check if there's anything to commit
if git diff --cached --quiet 2>/dev/null; then
    info "Nothing staged — skipping commit"
else
    git commit -m "chore: remove dataset images + model checkpoints from git tracking

Remove ~3,866 PNG chart images (650 MB) and ~40 model checkpoints (3.1 GB)
from git.  These are ephemeral artifacts:

  - Images are regenerated from historical bar data by the dataset generator
  - Model checkpoints are produced each training run; only the validated
    champion (breakout_cnn_best.pt) needs to be versioned

Updated .gitignore to exclude:
  - dataset/images/** (all generated chart PNGs)
  - dataset/*.csv and dataset_stats.json (regenerated each cycle)
  - models/* except breakout_cnn_best.pt and its metadata

Added .gitattributes with Git LFS tracking for:
  - *.pt (PyTorch model weights)
  - *.onnx, *.torchscript (future model formats)

Files remain on disk — only removed from git tracking."
    ok "Committed removal"
fi

# ---------------------------------------------------------------------------
# Step 6: Re-add champion model under LFS
# ---------------------------------------------------------------------------
step "6/8: Re-add champion model under Git LFS"

# Restore champion from backup
if [ -f "$BACKUP_DIR/breakout_cnn_best.pt" ]; then
    # Make sure the file is in place
    cp "$BACKUP_DIR/breakout_cnn_best.pt" "$CHAMPION"

    # LFS track should already be in .gitattributes, but verify
    if ! git lfs track --list 2>/dev/null | grep -q '\.pt'; then
        git lfs track "*.pt"
        git add .gitattributes
    fi

    # Add the champion — LFS filter will handle it
    git add --force "$CHAMPION"

    # Also add metadata if it exists
    if [ -f "$BACKUP_DIR/breakout_cnn_best_meta.json" ]; then
        cp "$BACKUP_DIR/breakout_cnn_best_meta.json" "$CHAMPION_META"
        git add --force "$CHAMPION_META"
    fi

    git commit -m "chore: add champion model to Git LFS

Track breakout_cnn_best.pt (85% val accuracy) via Git LFS.
This is the only model file committed to the repo — all training
checkpoints are .gitignored and managed locally.

The overnight retraining pipeline (retrain_overnight.py) handles:
  - Generating fresh training data from historical bars
  - GPU training with validation gate
  - Atomic promotion of new champion (only if it beats current)
  - Archival of previous champions locally"
    ok "Champion model committed via LFS"

    # Verify LFS is working
    info "Verifying LFS tracking..."
    LFS_FILES=$(git lfs ls-files 2>/dev/null | head -5)
    if [ -n "$LFS_FILES" ]; then
        ok "LFS files:"
        echo "$LFS_FILES"
    else
        warn "No LFS files detected — check .gitattributes configuration"
    fi
else
    warn "No champion model to re-add (skipped)"
fi

# ---------------------------------------------------------------------------
# Step 7: Purge old blobs from history (optional, biggest size savings)
# ---------------------------------------------------------------------------
step "7/8: Purge old blobs from git history (optional)"

printf "\n"
warn "This step rewrites git history to remove the old binary blobs."
warn "It will save ~3+ GB but requires all collaborators to re-clone."
warn ""
warn "You can skip this now and do it later with:"
warn "  git filter-repo --path dataset/images/ --path models/ --invert-paths"
warn "  OR"
warn "  java -jar bfg.jar --delete-folders images --delete-files '*.pt' --no-blob-protection"
printf "\n${YELLOW}Rewrite history now? (y/N): ${NC}"
read -r rewrite

if [ "$rewrite" = "y" ] || [ "$rewrite" = "Y" ]; then
    # Try git-filter-repo first (preferred), then BFG, then filter-branch
    if command -v git-filter-repo &>/dev/null; then
        info "Using git-filter-repo (recommended)..."

        # git-filter-repo requires a fresh clone or --force
        # We selectively remove the paths that had big binaries
        git filter-repo \
            --path dataset/images/ \
            --path models/breakout_cnn_20260228_005653_acc69.pt \
            --path models/breakout_cnn_20260301_100228_acc35.pt \
            --path models/breakout_cnn_20260301_103529_acc66.pt \
            --path models/breakout_cnn_20260301_103625_acc67.pt \
            --path models/breakout_cnn_20260301_103719_acc68.pt \
            --path models/breakout_cnn_20260301_104535_acc65.pt \
            --path models/breakout_cnn_20260301_104724_acc68.pt \
            --path models/breakout_cnn_20260301_105253_acc68.pt \
            --path models/breakout_cnn_20260301_105857_acc73.pt \
            --path models/breakout_cnn_20260301_110607_acc74.pt \
            --path models/breakout_cnn_20260301_111714_acc45.pt \
            --path models/breakout_cnn_20260301_111714_bestloss.pt \
            --path models/breakout_cnn_20260301_111808_acc64.pt \
            --path models/breakout_cnn_20260301_111808_bestloss.pt \
            --path models/breakout_cnn_20260301_111902_bestloss.pt \
            --path models/breakout_cnn_20260301_111958_bestloss.pt \
            --path models/breakout_cnn_20260301_112052_bestloss.pt \
            --path models/breakout_cnn_20260301_112148_acc68.pt \
            --path models/breakout_cnn_20260301_112148_bestloss.pt \
            --path models/breakout_cnn_20260301_112243_acc69.pt \
            --path models/breakout_cnn_20260301_112243_bestloss.pt \
            --path models/breakout_cnn_20260301_112338_acc73.pt \
            --path models/breakout_cnn_20260301_112338_bestloss.pt \
            --path models/breakout_cnn_20260301_112432_acc78.pt \
            --path models/breakout_cnn_20260301_112432_bestloss.pt \
            --path models/breakout_cnn_20260301_112715_acc82.pt \
            --path models/breakout_cnn_20260301_112715_bestloss.pt \
            --path models/breakout_cnn_20260301_112810_bestloss.pt \
            --path models/breakout_cnn_20260301_112904_bestloss.pt \
            --path models/breakout_cnn_20260301_112958_bestloss.pt \
            --path models/breakout_cnn_20260301_113051_acc83.pt \
            --path models/breakout_cnn_20260301_113052_bestloss.pt \
            --path models/breakout_cnn_20260301_113240_acc84.pt \
            --path models/breakout_cnn_20260301_113240_bestloss.pt \
            --path models/breakout_cnn_20260301_113334_bestloss.pt \
            --path models/breakout_cnn_20260301_113521_acc85.pt \
            --path models/breakout_cnn_20260301_113521_bestloss.pt \
            --path models/breakout_cnn_20260301_113804_bestloss.pt \
            --path models/breakout_cnn_20260301_113858_final.pt \
            --path models/training_history.csv \
            --path dataset/labels.csv \
            --path dataset/train.csv \
            --path dataset/val.csv \
            --path dataset/dataset_stats.json \
            --invert-paths \
            --force

        ok "History rewritten with git-filter-repo"

    elif command -v bfg &>/dev/null || [ -f "bfg.jar" ]; then
        info "Using BFG Repo-Cleaner..."
        BFG_CMD="bfg"
        if [ -f "bfg.jar" ]; then
            BFG_CMD="java -jar bfg.jar"
        fi

        # BFG removes blobs by pattern — protects the HEAD commit
        $BFG_CMD --delete-folders images --no-blob-protection .
        $BFG_CMD --delete-files '*.pt' --no-blob-protection .

        git reflog expire --expire=now --all
        git gc --prune=now --aggressive

        ok "History rewritten with BFG"

    else
        warn "Neither git-filter-repo nor BFG found."
        warn ""
        warn "Install one of them to purge history:"
        warn "  pip install git-filter-repo"
        warn "  OR download BFG: https://rtyley.github.io/bfg-repo-cleaner/"
        warn ""
        warn "Then run manually:"
        warn "  git filter-repo --path dataset/images/ --invert-paths --force"
        warn ""
        warn "Skipping history rewrite for now."
        warn "Your .git/ will remain large until you do this step."
    fi
else
    info "Skipping history rewrite."
    info ""
    info "You can do this later with:"
    info "  pip install git-filter-repo"
    info "  git filter-repo --path dataset/images/ --invert-paths --force"
    info ""
    info "The .gitignore changes still prevent NEW images/checkpoints from"
    info "being committed, so the repo won't grow further."
fi

# ---------------------------------------------------------------------------
# Step 8: Garbage collection + summary
# ---------------------------------------------------------------------------
step "8/8: Garbage collection & summary"

info "Running git gc..."
git reflog expire --expire=now --all 2>/dev/null || true
git gc --prune=now 2>/dev/null || true

ENDING_SIZE=$(du -sh .git/ 2>/dev/null | cut -f1)

# Clean up migration backup
if [ -d "$BACKUP_DIR" ]; then
    rm -rf "$BACKUP_DIR"
    info "Cleaned up migration backup"
fi

printf "\n"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Git LFS Migration Complete                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  .git/ before:  %-42s ║\n" "$STARTING_SIZE"
printf "║  .git/ after:   %-42s ║\n" "$ENDING_SIZE"
echo "║                                                            ║"
echo "║  What changed:                                             ║"
echo "║    ✓ dataset/images/ removed from tracking (regenerated)   ║"
echo "║    ✓ Training checkpoints removed from tracking            ║"
echo "║    ✓ breakout_cnn_best.pt tracked via Git LFS              ║"
echo "║    ✓ .gitignore updated to prevent re-adding               ║"
echo "║    ✓ .gitattributes configured for LFS                     ║"
echo "║                                                            ║"
echo "║  After a fresh clone, run:                                 ║"
echo "║    git lfs pull              (downloads the champion .pt)  ║"
echo "║    PYTHONPATH=src python scripts/retrain_overnight.py \    ║"
echo "║      --skip-dataset --immediate  (regenerate from bars)    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Next: push with force to update origin:                   ║"
echo "║    git push --force-with-lease origin main                 ║"
echo "║                                                            ║"
echo "║  Collaborators must re-clone or:                           ║"
echo "║    git fetch origin && git reset --hard origin/main        ║"
echo "║                                                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
printf "\n"

# Show what's tracked now
info "Currently tracked model files:"
git ls-files models/ 2>/dev/null || echo "  (none)"
printf "\n"
info "LFS-tracked files:"
git lfs ls-files 2>/dev/null || echo "  (none)"
printf "\n"
info "Dataset files tracked:"
git ls-files dataset/ 2>/dev/null || echo "  (none — images are regenerated)"
printf "\n"
ok "Migration complete!"
