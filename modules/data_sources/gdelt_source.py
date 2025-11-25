"""
GDELT News Data Source Plugin

Required interface for all data source plugins:
- PLUGIN_NAME: str - Display name for the menu
- PLUGIN_DESCRIPTION: str - Brief description
- fetch_news(ticker, company_name, start_date, end_date) -> pd.DataFrame
  Returns DataFrame with columns: url, title, calendar_date, ticker
"""

import requests
import pandas as pd
import sys
import time
from datetime import date, timedelta
from urllib.parse import urlencode

PLUGIN_NAME = "GDELT News API"
PLUGIN_DESCRIPTION = "US English news from GDELT project (up to 200 articles/day)"

DOC_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_news(ticker, company_name, start_date=None, end_date=None, per_day=200):
    """
    Fetch news from GDELT API.

    Args:
        ticker: Stock symbol (e.g., "META")
        company_name: Company name (e.g., "Meta")
        start_date: datetime.date object (defaults to Sept 1, 2025)
        end_date: datetime.date object (defaults to Sept 30, 2025)
        per_day: Max articles per day

    Returns:
        pd.DataFrame with columns: url, title, calendar_date, ticker
        (and optionally other metadata columns)
    """
    # Default date range (matching original hardcoded range)
    if start_date is None:
        start_date = date(2025, 9, 1)
    if end_date is None:
        end_date = date(2025, 9, 30)

    rows = []
    query = f'({ticker} OR {company_name}) sourcecountry:us sourcelang:english'

    # Calculate total days for progress counter
    total_days = (end_date - start_date).days + 1
    start_time = time.time()

    day = start_date
    day_counter = 0

    while day <= end_date:
        day_counter += 1

        # Display progress (self-replacing)
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r   Querying GDELT [{day_counter}/{total_days}] {day}...")
        sys.stdout.flush()

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

            articles = data.get("articles", [])
            if articles:
                df = pd.DataFrame(articles)
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
            sys.stdout.write("\n")
            print(f"   ! Error fetching {day}: {e}")

        day += timedelta(days=1)

    # Final summary with total time
    elapsed_total = time.time() - start_time
    sys.stdout.write(f"\r   Querying GDELT [{total_days}/{total_days}] - Completed in {elapsed_total:.1f}s\n")
    sys.stdout.flush()

    if not rows:
        return pd.DataFrame()

    df_final = pd.concat(rows, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["url", "title"])
    print(f"   -> Found {len(df_final)} relevant articles.")

    return df_final
