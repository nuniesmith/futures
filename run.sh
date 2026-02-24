#!/bin/bash
set -euo pipefail

# Generate .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "No .env file found. Generating with placeholders..."
    cat > .env <<'EOF'
# xAI Grok API key for the AI Analyst tab
# Get yours at https://console.x.ai
XAI_API_KEY=your_xai_api_key_here

# Redis URL (set automatically by docker-compose, only override for custom setups)
# REDIS_URL=redis://redis:6379/0

# SQLite journal path (inside the container)
# DB_PATH=/app/data/futures_journal.db
EOF
    echo ".env file created. Edit it to add your XAI_API_KEY, then re-run this script."
    exit 0
fi

# Check if XAI_API_KEY is still the placeholder
if grep -q "your_xai_api_key_here" .env; then
    echo "WARNING: XAI_API_KEY is still set to placeholder in .env"
    echo "  The Grok AI Analyst tab will not work until you update it."
    echo "  Edit .env and replace 'your_xai_api_key_here' with your actual key."
    echo ""
fi

echo "Starting Futures Dashboard..."
docker compose up --build -d

echo ""
echo "Dashboard is running:"
echo "  Streamlit:  http://localhost:8501"
echo "  Trade API:  http://localhost:8000"
echo "  Redis:      localhost:6379"
echo ""
echo "Logs: docker compose logs -f app"
echo "Stop: docker compose down"
