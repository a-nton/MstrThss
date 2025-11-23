import pandas as pd
import os
from config import ASSETS, DATA_DIR
from utils import get_todays_data_dir
from modules.ingestion import load_price_data, fetch_gdelt_daily

def main():
    print("=== STEP 1: DATA INGESTION ===")
    
    # 1. Prepare Directory
    save_dir = get_todays_data_dir(DATA_DIR)
    print(f"💾 Saving raw data to: {save_dir}")

    # 2. Fetch Data
    for asset in ASSETS:
        ticker = asset["symbol"]
        name = asset["name"]
        
        print(f"\n--- Processing {ticker} ---")
        
        # Prices
        df_prices = load_price_data(ticker)
        price_path = os.path.join(save_dir, f"{ticker}_prices.csv")
        df_prices.to_csv(price_path, index=False)
        print(f"   Saved prices to {price_path}")
        
        # News
        df_news = fetch_gdelt_daily(ticker, name)
        news_path = os.path.join(save_dir, f"{ticker}_news.csv")
        if not df_news.empty:
            df_news.to_csv(news_path, index=False)
            print(f"   Saved news to {news_path}")
        else:
            print("   No news found.")

if __name__ == "__main__":
    main()