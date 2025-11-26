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
import re
from datetime import date, timedelta
from urllib.parse import urlencode

PLUGIN_NAME = "GDELT News API"
PLUGIN_DESCRIPTION = "US English news from GDELT project (up to 200 articles/day)"

DOC_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def clean_company_name_for_query(name):
    """
    Clean company name for GDELT API query.
    Removes text in parentheses and corporate suffixes.
    """
    # Remove content in parentheses "Alphabet (Google)" -> "Alphabet"
    name = re.sub(r'\(.*?\)', '', name)
    
    # Remove common suffixes
    suffixes = [r'\bInc\.?\b', r'\bCorp\.?\b', r'\bLtd\.?\b', r'\bGroup\b']
    for s in suffixes:
        name = re.sub(s, '', name, flags=re.IGNORECASE)
        
    return name.strip()

def fetch_news(ticker, company_name, start_date=None, end_date=None, per_day=200):
    """
    Fetch news from GDELT API.
    """
    # Default date range
    if start_date is None:
        start_date = date(2025, 9, 1)
    if end_date is None:
        end_date = date(2025, 9, 30)

    rows = []
    
    # Clean name for query construction
    # "Alphabet (Google)" -> "Alphabet" (or "Google" if that was main)
    # Actually, for GDELT API we want the most specific term.
    # If user provided "Alphabet (Google)", simple cleaning leaves "Alphabet".
    # But our new config_manager uses "Google", so this cleaning is a safety net.
    clean_name = clean_company_name_for_query(company_name)
    
    # Quote the name if it has spaces (e.g. "Meta Platforms")
    if " " in clean_name:
        clean_name = f'"{clean_name}"'
        
    # Construct query: (GOOGL OR "Alphabet")
    query = f'({ticker} OR {clean_name}) sourcecountry:us sourcelang:english'

    total_days = (end_date - start_date).days + 1
    start_time = time.time()

    day = start_date
    day_counter = 0

    print(f"   Query: {query}")

    while day <= end_date:
        day_counter += 1

        # Display progress
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
                # We use the ORIGINAL company_name here for broader matching in Python
                # But we split it to be smart
                keywords = re.split(r'[\s\(\)]', company_name)
                keywords = [k.strip() for k in keywords if len(k.strip()) > 2]
                keywords.append(ticker)
                
                if "title" in df.columns:
                    # Match ANY keyword in title
                    mask = pd.Series([False] * len(df), index=df.index)
                    for k in keywords:
                        mask |= df["title"].str.contains(re.escape(k), case=False, na=False)
                    
                    df_filtered = df[mask]
                    if not df_filtered.empty:
                        rows.append(df_filtered)
                else:
                    rows.append(df)

        except Exception as e:
            sys.stdout.write("\n")
            print(f"   ! Error fetching {day}: {e}")

        day += timedelta(days=1)

    elapsed_total = time.time() - start_time
    sys.stdout.write(f"\r   Querying GDELT [{total_days}/{total_days}] - Completed in {elapsed_total:.1f}s\n")
    sys.stdout.flush()

    if not rows:
        return pd.DataFrame()

    df_final = pd.concat(rows, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["url", "title"])
    print(f"   -> Found {len(df_final)} relevant articles.")

    return df_final