"""
GDELT BigQuery Data Source Plugin

Required interface for all data source plugins:
- PLUGIN_NAME: str - Display name for the menu
- PLUGIN_DESCRIPTION: str - Brief description
- fetch_news(ticker, company_name, start_date, end_date) -> pd.DataFrame
  Returns DataFrame with columns: url, title, calendar_date, ticker

Requires:
- Google Cloud project with BigQuery API enabled
- GOOGLE_APPLICATION_CREDENTIALS environment variable set to path of service account JSON
- google-cloud-bigquery library installed
"""

import pandas as pd
import sys
import time
import os
from datetime import date, timedelta
from config import DATA_START_DATE, NEWS_PREDICTION_END_DATE 

try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None 

PLUGIN_NAME = "GDELT BigQuery (Historical)"
PLUGIN_DESCRIPTION = "Full GDELT archive via BigQuery (2015-present, requires GCP credentials)"


def check_bigquery_available():
    """Check if BigQuery is properly configured"""
    if bigquery is None:
        return False, "google-cloud-bigquery not installed (run: pip install google-cloud-bigquery)"
    
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"

    try:
        client = bigquery.Client()
        return True, None
    except Exception as e:
        return False, f"Failed to create BigQuery client: {str(e)}"


def fetch_news(ticker, company_name, start_date=None, end_date=None, max_per_day=1000):
    """
    Fetch news from GDELT BigQuery using "Smart Regex" matching.
    
    Cost Note: Making the filter stricter DOES NOT increase query cost. 
    It creates a higher quality result set, ensuring the LIMIT 1000 captures 
    real news instead of noise.
    """

    # Check BigQuery availability
    available, error_msg = check_bigquery_available()
    if not available:
        print(f"❌ BigQuery not available: {error_msg}")
        print("   Setup instructions in BIGQUERY_SETUP.md")
        return pd.DataFrame()

    if start_date is None:
        start_date = DATA_START_DATE
    if end_date is None:
        end_date = NEWS_PREDICTION_END_DATE

    rows = []
    total_days = (end_date - start_date).days + 1
    start_time = time.time()
    
    total_bytes_processed = 0
    total_bytes_billed = 0
    
    day = start_date
    day_counter = 0

    client = bigquery.Client()
    
    # --- SMART REGEX PATTERNS ---
    # (?i) = Case insensitive
    # (?:^|[^a-z]) = Start of string OR any character that is NOT a letter
    # This allows 'META' to match 'ORG_META_PLATFORMS' (underscore is non-letter in this regex context)
    # But rejects 'METAL' because 'L' is a letter.
    
    # Pattern for Ticker (e.g., META)
    bq_pattern_ticker = fr'(?i)(?:^|[^a-z]){ticker.lower()}(?:$|[^a-z])'
    
    # Pattern for Name (e.g., Meta)
    bq_pattern_name = fr'(?i)(?:^|[^a-z]){company_name.lower()}(?:$|[^a-z])'

    print(f"   Querying GDELT BigQuery day-by-day from {start_date} to {end_date}...")
    
    while day <= end_date:
        day_counter += 1
        
        sys.stdout.write(f"\r   Executing BigQuery [{day_counter}/{total_days}] {day}...")
        sys.stdout.flush()
        
        date_str = day.strftime('%Y-%m-%d')
        
        # Build query with Smart Regex
        query = f"""
        SELECT
            DocumentIdentifier as url,
            DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as calendar_date,
            V2Tone as Tone,
            REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS page_title
        FROM
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            _PARTITIONDATE = '{date_str}'
            AND DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) = '{date_str}'
            AND Extras LIKE '%<PAGE_TITLE>%'
            AND GKGRECORDID NOT LIKE '%-T%'
            
            -- SMART SEARCH: Strict enough to reject 'Metal', loose enough for 'Meta_Platforms'
            AND (
                REGEXP_CONTAINS(Themes, r'{bq_pattern_ticker}')
                OR REGEXP_CONTAINS(Themes, r'{bq_pattern_name}')
                OR REGEXP_CONTAINS(Persons, r'{bq_pattern_name}')
                OR REGEXP_CONTAINS(Organizations, r'{bq_pattern_name}')
            )
            AND SourceCommonName IS NOT NULL
        LIMIT {max_per_day}
        """

        try:
            query_job = client.query(query)
            results = query_job.result()
            
            total_bytes_processed += query_job.total_bytes_processed
            total_bytes_billed += query_job.total_bytes_billed

            df = results.to_dataframe()
            
            # Debug count
            raw_count = len(df)

            if not df.empty:
                df['title'] = df['page_title'] 
                df['ticker'] = ticker
                
                # Basic cleanup: remove empty titles
                df = df[df['title'].astype(str).str.strip() != '']
                
                # Redundant safety filter (Python side)
                # Just in case the regex missed an edge case, we do a final strict check on the TITLE.
                strict_ticker = rf'\b{ticker}\b'
                strict_name = rf'\b{company_name}\b'
                
                mask = df["title"].str.contains(strict_ticker, case=False, regex=True) | \
                       df["title"].str.contains(strict_name, case=False, regex=True)
                
                df = df[mask]
                
                # Final deduplication
                df = df[['url', 'title', 'calendar_date', 'ticker']]
                df = df.drop_duplicates(subset=['url'])
                
                rows.append(df)
            
            final_count = len(df)
            sys.stdout.write(f"\r   Executing BigQuery [{day_counter}/{total_days}] {day} - Fetched {raw_count} -> Kept {final_count}\n")
            sys.stdout.flush()

        except Exception as e:
            sys.stdout.write("\n")
            print(f"❌ BigQuery error for {day}: {e}")
            sys.stdout.flush()

        day += timedelta(days=1)

    elapsed_total = time.time() - start_time
    total_articles = sum(len(df) for df in rows)
    sys.stdout.write(f"\r   BigQuery query completed in {elapsed_total:.1f}s - Found {total_articles} total articles\n")
    sys.stdout.flush()
    
    print(f"   📊 Total Bytes processed: {total_bytes_processed / 1e9:.2f} GB")
    print(f"   📊 Total Bytes billed: {total_bytes_billed / 1e9:.2f} GB")
    
    if not rows:
        return pd.DataFrame()
    
    df_final = pd.concat(rows, ignore_index=True)
    return df_final