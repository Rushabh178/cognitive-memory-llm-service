#!/bin/bash
# ============================================================
# setup.sh — One-shot environment setup for Linux / macOS
#
# Run this once before starting the server for the first time:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -e  # exit immediately if any command fails

echo "----------------------------------------------------"
echo " AI Cognitive Memory System — Environment Setup"
echo "----------------------------------------------------"

# Step 1: Create a virtual environment in ./venv
# A venv keeps all project dependencies isolated from your
# system Python so different projects can use different versions.
echo "[1/4] Creating virtual environment..."
python3 -m venv venv

# Step 2: Activate the virtual environment.
# After this, 'python' and 'pip' refer to the venv copies,
# not the system-wide ones.
echo "[2/4] Activating virtual environment..."
source venv/bin/activate

# Step 3: Install all dependencies from requirements.txt.
# --upgrade pip first to avoid old-pip warnings.
echo "[3/4] Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

# Step 4: Create .env from .env.example if .env doesn't exist yet.
# We never overwrite an existing .env to avoid wiping real API keys.
echo "[4/4] Checking .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  --> .env created from .env.example"
    echo "  --> IMPORTANT: open .env and fill in ANTHROPIC_API_KEY and API_BEARER_TOKEN"
else
    echo "  --> .env already exists, skipping copy"
fi

echo ""
echo "----------------------------------------------------"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Fill in your API keys in .env"
echo "   2. Activate the venv:  source venv/bin/activate"
echo "   3. Start the server:   uvicorn main:app --reload"
echo "----------------------------------------------------"
