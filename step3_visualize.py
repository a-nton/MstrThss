import pandas as pd
import os
import glob
from config import RESULTS_DIR
from utils import get_todays_data_dir
from modules.viz import generate_bokeh_dashboard

def get_latest_run_dir(base_dir):
    """
    Finds the most recent run directory for today.
    Returns path like: results/2025-11-23/run_003/
    """
    from datetime import datetime
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_path = os.path.join(base_dir, date_str)

    if not os.path.exists(date_path):
        return None

    runs = sorted(glob.glob(os.path.join(date_path, "run_*")))
    if not runs:
        return None

    return runs[-1]  # Most recent run

def main():
    print("=== STEP 3: VISUALIZATION ONLY ===")

    # Find the latest run with results
    run_dir = get_latest_run_dir(RESULTS_DIR)

    if not run_dir:
        print("No existing run directories found for today.")
        print("Please run the LLM step first (option 2).")
        return

    csv_path = os.path.join(run_dir, "run_results.csv")

    if not os.path.exists(csv_path):
        print(f"No results CSV found at {csv_path}")
        print("Please run the LLM step first (option 2).")
        return

    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows")

    print("Generating Dashboard...")
    dash_path = generate_bokeh_dashboard(df, run_dir)
    print(f"Dashboard ready: {dash_path}")

if __name__ == "__main__":
    main()
