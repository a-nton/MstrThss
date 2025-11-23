import pandas as pd
import os
from config import ASSETS, DATA_DIR, RESULTS_DIR
from utils import get_todays_data_dir, create_new_run_dir
from modules.alignment import align_news
from modules.llm import run_llm
from modules.viz import generate_bokeh_dashboard

def main():
    print("=== STEP 2: MODEL & VIZ ===")
    
    # 1. Locate Input Data
    data_source_dir = get_todays_data_dir(DATA_DIR)
    if not os.path.exists(data_source_dir):
        print(f"❌ Error: No data found at {data_source_dir}")
        print("   Please run step1_ingest_data.py first!")
        return

    # 2. Create Output Run Folder
    run_dir = create_new_run_dir(RESULTS_DIR)
    
    all_panels = []

    # 3. Process Each Asset
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
        
        # Predict
        panel_llm = run_llm(panel, name)
        all_panels.append(panel_llm)

    # 4. Save & Visualize
    if all_panels:
        final_df = pd.concat(all_panels, ignore_index=True)
        csv_path = os.path.join(run_dir, "run_results.csv")
        final_df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved results to {csv_path}")
        
        print("📊 Generating Dashboard...")
        dash_path = generate_bokeh_dashboard(final_df, run_dir)
        print(f"✅ Dashboard ready: {dash_path}")
    else:
        print("❌ No results generated.")

if __name__ == "__main__":
    main()