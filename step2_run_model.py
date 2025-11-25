import pandas as pd
import os
import sys
from config import ASSETS, DATA_DIR, RESULTS_DIR
from utils import get_todays_data_dir, create_new_run_dir
from modules.alignment import align_news
from modules.llm import run_llm
from plugin_loader import load_plugin

def main(llm_config_module=None, run_dir=None):
    print("=== STEP 2: LLM PROCESSING ===")

    # 1. Load LLM config plugin
    if not llm_config_module:
        print("❌ No LLM config plugin specified")
        sys.exit(1)

    try:
        llm_plugin = load_plugin("modules/llm_configs", llm_config_module)
        print(f"✓ Using LLM config: {llm_plugin.PLUGIN_NAME}")
        print(f"  {llm_plugin.PLUGIN_DESCRIPTION}")
    except Exception as e:
        print(f"❌ Failed to load LLM config plugin '{llm_config_module}': {e}")
        sys.exit(1)

    # 2. Locate Input Data
    data_source_dir = get_todays_data_dir(DATA_DIR)
    if not os.path.exists(data_source_dir):
        print(f"❌ Error: No data found at {data_source_dir}")
        print("   Please run step1_ingest_data.py first!")
        return

    # 3. Use provided run_dir or create new one
    if not run_dir:
        run_dir = create_new_run_dir(RESULTS_DIR)
    else:
        os.makedirs(run_dir, exist_ok=True)

    all_panels = []

    # 4. Process Each Asset
    for asset in ASSETS:
        ticker = asset["symbol"]
        name = asset["name"]

        print(f"\n--- Analyzing {ticker} ---")

        price_path = os.path.join(data_source_dir, f"{ticker}_prices.csv")
        news_path = os.path.join(data_source_dir, f"{ticker}_news.csv")

        if not os.path.exists(price_path):
            print(f"   Skipping {ticker} (No price data)")
            continue

        df_p = pd.read_csv(price_path)
        df_n = pd.read_csv(news_path) if os.path.exists(news_path) else pd.DataFrame()

        # Align
        panel = align_news(df_p, df_n)

        # Predict (using plugin)
        panel_llm = run_llm(panel, name, llm_plugin)
        all_panels.append(panel_llm)

    # 5. Save Results
    if all_panels:
        final_df = pd.concat(all_panels, ignore_index=True)
        csv_path = os.path.join(run_dir, "run_results.csv")
        final_df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved results to {csv_path}")
        print(f"📁 Results saved to: {run_dir}")
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-config", required=True, help="LLM config plugin module name")
    parser.add_argument("--run-dir", help="Run directory for output")
    args = parser.parse_args()
    main(llm_config_module=args.llm_config, run_dir=args.run_dir)