import os
import glob
from datetime import datetime

def get_todays_data_dir(base_dir):
    """
    Returns a path like: data_warehouse/2025-11-23/
    Creates it if it doesn't exist.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(base_dir, date_str)
    os.makedirs(path, exist_ok=True)
    return path

def create_new_run_dir(base_dir):
    """
    Returns a path like: results/2025-11-23/run_001/
    Auto-increments the run number based on existing folders.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_path = os.path.join(base_dir, date_str)
    os.makedirs(date_path, exist_ok=True)
    
    # Count existing run folders to determine the next number
    existing_runs = glob.glob(os.path.join(date_path, "run_*"))
    run_num = len(existing_runs) + 1
    
    new_run_path = os.path.join(date_path, f"run_{run_num:03d}")
    os.makedirs(new_run_path, exist_ok=True)
    
    print(f"📁 Created output directory: {new_run_path}")
    return new_run_path