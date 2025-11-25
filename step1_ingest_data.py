import pandas as pd
import os
import sys
from config import ASSETS, DATA_DIR
from utils import get_todays_data_dir
from modules.ingestion import load_price_data
from plugin_loader import load_plugin

def main(data_source_module=None, run_dir=None):
    print("=== STEP 1: DATA INGESTION ===")

    # 1. Load data source plugin
    if not data_source_module:
        print("❌ No data source plugin specified")
        sys.exit(1)

    try:
        plugin = load_plugin("modules/data_sources", data_source_module)
        print(f"✓ Using data source: {plugin.PLUGIN_NAME}")
        print(f"  {plugin.PLUGIN_DESCRIPTION}")
    except Exception as e:
        print(f"❌ Failed to load data source plugin '{data_source_module}': {e}")
        sys.exit(1)

    # 2. Prepare Directory
    save_dir = get_todays_data_dir(DATA_DIR)
    print(f"💾 Saving raw data to: {save_dir}")

    # Store run_dir for later steps if provided
    if run_dir:
        import json
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, ".run_info.json"), "w") as f:
            json.dump({"data_dir": save_dir}, f)

    # 3. Fetch Data
    for asset in ASSETS:
        ticker = asset["symbol"]
        name = asset["name"]

        print(f"\n--- Processing {ticker} ---")

        # Prices (still using existing ingestion module)
        df_prices = load_price_data(ticker)
        price_path = os.path.join(save_dir, f"{ticker}_prices.csv")
        df_prices.to_csv(price_path, index=False)
        print(f"   Saved prices to {price_path}")

        # News (using plugin)
        df_news = plugin.fetch_news(ticker, name, start_date=None, end_date=None)
        news_path = os.path.join(save_dir, f"{ticker}_news.csv")
        if not df_news.empty:
            df_news.to_csv(news_path, index=False)
            print(f"   Saved news to {news_path}")
        else:
            print("   No news found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", required=True, help="Data source plugin module name")
    parser.add_argument("--run-dir", help="Run directory for output")
    args = parser.parse_args()
    main(data_source_module=args.data_source, run_dir=args.run_dir)