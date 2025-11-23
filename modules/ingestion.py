import requests
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from urllib.parse import urlencode
from config import DOC_BASE_URL

def load_price_data(ticker):
    """
    Fetches daily price data from YFinance.
    Hardcoded range matches your original test: Sept-Nov 2025
    """
    # Adjust these dates if you want to run for "today"
    start = date(2025, 9, 1)
    end = date(2025, 11, 1)

    print(f"   Downloading prices for {ticker} ({start} to {end})...")
    
    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # Clean up MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        close_col = (ticker, "Close")
        df = data[close_col].rename("close").to_frame()
    else:
        close_col = "Close"
        if close_col not in data.columns:
            # Fallback if auto_adjust changes column names
            close_col = "Adj Close" if "Adj Close" in data.columns else data.columns[0]
        df = data[close_col].rename("close").to_frame()

    df["date"] = df.index
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.reset_index(drop=True)

    # Calculate lagged returns
    df["ret_1d"] = df["close"].pct_change()
    df["ret_2d"] = df["close"].pct_change(periods=2)
    df["ret_3d"] = df["close"].pct_change(periods=3)

    # Calculate Forward returns
    df["ret_fwd_1d"] = df["close"].shift(-1) / df["close"] - 1
    df["ret_fwd_1w"] = df["close"].shift(-5) / df["close"] - 1
    df["ret_fwd_1m"] = df["close"].shift(-21) / df["close"] - 1

    df["ticker"] = ticker
    return df

def fetch_gdelt_daily(ticker, company_name, per_day=200):
    """
    Fetches GDELT articles.
    Hardcoded range matches your original test: Sept 2025
    """
    rows = []
    # Adjust dates here for live runs
    day = date(2025, 9, 1)
    end = date(2025, 9, 30)

    query = f'({ticker} OR {company_name}) sourcecountry:us sourcelang:english'

    print(f"   Querying GDELT for {ticker}...")
    
    while day <= end:
        start_dt = f"{day:%Y%m%d}000000"
        end_dt = f"{day:%Y%m%d}235959"

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "STARTDATETIME": start_dt,
            "ENDDATETIME": end_dt,
            "maxrecords": per_day,
            "sort": "DateDesc",
        }

        url = f"{DOC_BASE_URL}?{urlencode(params)}"
        
        try:
            r = requests.get(url, timeout=40)
            r.raise_for_status()
            data = r.json()
            
            a = data.get("articles", [])
            if a:
                df = pd.DataFrame(a)
                df["calendar_date"] = day
                df["ticker"] = ticker

                # Filter by title relevance
                if "title" in df.columns:
                    mask = df["title"].str.contains(ticker, case=False, na=False) | \
                           df["title"].str.contains(company_name, case=False, na=False)
                    df_filtered = df[mask]
                    if not df_filtered.empty:
                        rows.append(df_filtered)
                else:
                    rows.append(df)
                    
        except Exception as e:
            print(f"   ! Error fetching {day}: {e}")
            
        day += timedelta(days=1)

    if not rows:
        return pd.DataFrame()

    df_final = pd.concat(rows, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["url", "title"])
    print(f"   -> Found {len(df_final)} relevant articles.")
    return df_final