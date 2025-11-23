#!/bin/bash

echo "=========================================="
echo "      GDELT PROJECT AUTO-LAUNCHER"
echo "=========================================="

# 0. Set Context to Script Directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || { echo "Failed to enter script directory"; exit 1; }

echo "[INFO] Working directory set to: $SCRIPT_DIR"

# 1. Check VENV Creation
if [ ! -d ".venv" ]; then
    echo "[INFO] Virtual environment not found. Creating one..."
    python3 -m venv .venv
else
    echo "[INFO] Using existing virtual environment."
fi

# 2. Activate & Install Requirements (ALWAYS RUNS)
source .venv/bin/activate

echo "[INFO] Checking/Installing requirements..."
.venv/bin/pip install -r requirements.txt

# 3. Menu
echo ""
echo "Select an option:"
echo "1. Run FULL Pipeline (Ingest Data + LLM + Viz)"
echo "2. Run LLM + Viz (Use existing data)"
echo "3. Run Viz ONLY (Use existing LLM results)"
echo "4. Exit"
read -p "Enter choice: " choice

if [ "$choice" == "1" ]; then
    echo "Running Step 1: Data Ingestion..."
    python step1_ingest_data.py
    echo "Running Step 2: LLM + Viz..."
    python step2_run_model.py
elif [ "$choice" == "2" ]; then
    echo "Running Step 2: LLM + Viz..."
    python step2_run_model.py
elif [ "$choice" == "3" ]; then
    echo "Running Step 3: Visualization Only..."
    python step3_visualize.py
fi

echo "Done."