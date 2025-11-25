#!/usr/bin/env python3
"""
GDELT Pipeline Launcher with Plugin Selection

This launcher provides an interactive menu for selecting:
- Data source plugins (for fetching news headlines)
- LLM config plugins (for different models/prompts)

Then runs the appropriate pipeline steps with selected plugins.
"""

import sys
import subprocess
import os
from datetime import datetime
from plugin_loader import discover_plugins, select_plugin

class TeeOutput:
    """Captures terminal output to both console and log file"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def run_command_realtime(command):
    """Runs a command and streams output to sys.stdout (TeeOutput) in real-time"""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, # Merge stderr into stdout
        text=True,
        bufsize=1, # Line buffering
        universal_newlines=True
    )
    
    # Read output line by line as it is generated
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush() # Ensure it hits the terminal/log immediately
        
    return process.wait() == 0

def get_log_path():
    """Get path for the current run's log file"""
    from utils import create_new_run_dir
    from config import RESULTS_DIR
    run_dir = create_new_run_dir(RESULTS_DIR)
    log_path = os.path.join(run_dir, "pipeline_log.txt")
    return log_path, run_dir

def run_step1(data_source_module, run_dir):
    """Run data ingestion with selected data source plugin"""
    print("\n" + "="*50)
    print("Running Step 1: Data Ingestion...")
    print("="*50)
    sys.stdout.flush()
    
    # Added "-u" for unbuffered output
    cmd = [sys.executable, "-u", "step1_ingest_data.py", "--data-source", data_source_module, "--run-dir", run_dir]
    return run_command_realtime(cmd)

def run_step2(llm_config_module, run_dir):
    """Run LLM processing with selected LLM config plugin"""
    print("\n" + "="*50)
    print("Running Step 2: LLM Processing...")
    print("="*50)
    sys.stdout.flush()
    
    # Added "-u" for unbuffered output
    cmd = [sys.executable, "-u", "step2_run_model.py", "--llm-config", llm_config_module, "--run-dir", run_dir]
    return run_command_realtime(cmd)

def run_step3():
    """Run visualization (no plugin needed)"""
    print("\n" + "="*50)
    print("Running Step 3: Visualization...")
    print("="*50)
    sys.stdout.flush()
    
    # Added "-u" for unbuffered output
    cmd = [sys.executable, "-u", "step3_visualize.py"]
    return run_command_realtime(cmd)

def main():
    print("="*50)
    print("      GDELT PIPELINE LAUNCHER")
    print("="*50)
    print("\nSelect pipeline mode:")
    print("1. FULL Pipeline (Ingest → LLM → Viz)")
    print("2. LLM ONLY (Use existing data)")
    print("3. Viz ONLY (Use existing results)")
    print("4. Exit")

    choice = input("\nEnter choice (1-4): ").strip()

    # Setup logging for runs that need it (1 and 2)
    tee = None
    log_path = None

    try:
        if choice == "1":
            # Full pipeline - need both plugins
            print("\n--- Step 1: Select Data Source ---")
            data_sources = discover_plugins("modules/data_sources")
            data_source_module = select_plugin(data_sources, "data source")

            if not data_source_module:
                print("❌ Data source selection cancelled")
                return

            print("\n--- Step 2: Select LLM Configuration ---")
            llm_configs = discover_plugins("modules/llm_configs")
            llm_config_module = select_plugin(llm_configs, "LLM config")

            if not llm_config_module:
                print("❌ LLM config selection cancelled")
                return

            # Create log file
            log_path, run_dir = get_log_path()
            tee = TeeOutput(log_path)
            original_stdout = sys.stdout
            sys.stdout = tee

            print(f"\n📝 Logging output to: {log_path}")
            print(f"⏱️  Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            sys.stdout.flush()

            # Run all steps
            if not run_step1(data_source_module, run_dir):
                print("❌ Step 1 failed")
                return

            if not run_step2(llm_config_module, run_dir):
                print("❌ Step 2 failed")
                return

            run_step3()

            print(f"\n⏱️  Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Results and log saved to: {run_dir}")
            sys.stdout.flush()

        elif choice == "2":
            # LLM only - need LLM config plugin
            print("\n--- Select LLM Configuration ---")
            llm_configs = discover_plugins("modules/llm_configs")
            llm_config_module = select_plugin(llm_configs, "LLM config")

            if not llm_config_module:
                print("❌ LLM config selection cancelled")
                return

            # Create log file
            log_path, run_dir = get_log_path()
            tee = TeeOutput(log_path)
            original_stdout = sys.stdout
            sys.stdout = tee

            print(f"\n📝 Logging output to: {log_path}")
            print(f"⏱️  Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            sys.stdout.flush()

            run_step2(llm_config_module, run_dir)

            print(f"\n⏱️  Run completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📁 Results and log saved to: {run_dir}")
            sys.stdout.flush()

        elif choice == "3":
            # Viz only - no plugins needed
            run_step3()

        elif choice == "4":
            print("Goodbye!")
            return

        else:
            print("Invalid choice")
            return

    finally:
        # Restore stdout and close log
        if tee:
            sys.stdout = original_stdout
            tee.close()
            print(f"\n✓ Log saved to: {log_path}")

if __name__ == "__main__":
    main()
