#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dependencies from requirements.txt
pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt"

# Install linter
pip install ruff
