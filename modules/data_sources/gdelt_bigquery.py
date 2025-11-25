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
# Import BigQuery here to ensure check_bigquery_available works
try:
    from google.cloud import bigquery
except ImportError:
    bigquery = None # Handle missing dependency gracefully

PLUGIN_NAME = "GDELT BigQuery (Historical)"
PLUGIN_DESCRIPTION = "Full GDELT archive via BigQuery (2015-present, requires GCP credentials)"


def check_bigquery_available():
    """Check if BigQuery is properly configured"""
    if bigquery is None:
        return False, "google-cloud-bigquery not installed (run: pip install google-cloud-bigquery)"
    
    # Check for credentials
    if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"

    # Try to create client
    try:
        client = bigquery.Client()
        return True, None
    except Exception as e:
        return False, f"Failed to create BigQuery client: {str(e)}"

# --- REMOVED extract_title_from_url function ---


def fetch_news(ticker, company_name, start_date=None, end_date=None, max_per_day=1000):
    """
    Fetch news from GDELT BigQuery day-by-day.

    Args:
        ticker: Stock symbol (e.g., "META")
        company_name: Company name (e.g., "Meta")
        start_date: datetime.date object (defaults to Sept 1, 2025)
        end_date: datetime.date object (defaults to Sept 30, 2025)
        max_per_day: Maximum number of articles to retrieve for EACH day.

    Returns:
        pd.DataFrame with columns: url, title, calendar_date, ticker
    """

    # Check BigQuery availability
    available, error_msg = check_bigquery_available()
    if not available:
        print(f"❌ BigQuery not available: {error_msg}")
        print("   Setup instructions in BIGQUERY_SETUP.md")
        return pd.DataFrame()

    # Default date range
    if start_date is None:
        start_date = date(2025, 9, 1)
    if end_date is None:
        end_date = date(2025, 9, 30)

    rows = []
    total_days = (end_date - start_date).days + 1
    start_time = time.time()
    
    day = start_date
    day_counter = 0

    client = bigquery.Client()

    print(f"   Querying GDELT BigQuery day-by-day from {start_date} to {end_date}...")
    
    while day <= end_date:
        day_counter += 1
        
        # Display progress
        sys.stdout.write(f"\r   Executing BigQuery [{day_counter}/{total_days}] {day}...")
        sys.stdout.flush()
        
        # Prepare filters for the current day
        date_str = day.strftime('%Y-%m-%d')
        
        # Build query: Uses REGEXP_EXTRACT on the Extras field for the actual headline
        query = f"""
        SELECT
            DocumentIdentifier as url,
            DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as calendar_date,
            V2Tone as Tone,
            REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS page_title -- TRUE HEADLINE FIX
        FROM
            `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE
            -- CRITICAL: Use partition filter for cost reduction
            _PARTITIONDATE = '{date_str}'
            -- Secondary date filter for robustness
            AND DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) = '{date_str}'
            -- Filter out records that don't have a title field, which fixes the garbage output
            AND Extras LIKE '%<PAGE_TITLE>%'
            AND (
                LOWER(Themes) LIKE '%{ticker.lower()}%'
                OR LOWER(Themes) LIKE '%{company_name.lower()}%'
                OR LOWER(Persons) LIKE '%{company_name.lower()}%'
                OR LOWER(Organizations) LIKE '%{company_name.lower()}%'
            )
            AND SourceCommonName IS NOT NULL
        LIMIT {max_per_day}
        """

        try:
            query_job = client.query(query)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()

            # --- Post-Processing ---
            if not df.empty:
                # Use the extracted page_title as the 'title' column
                df['title'] = df['page_title'] 
                df['ticker'] = ticker
                df = df[['url', 'title', 'calendar_date', 'ticker']]
                df = df.drop_duplicates(subset=['url'])
                rows.append(df)
            
            # Show daily status
            sys.stdout.write(f"\r   Executing BigQuery [{day_counter}/{total_days}] {day} - Found {len(df)} articles\n")
            sys.stdout.flush()

        except Exception as e:
            sys.stdout.write("\n")
            print(f"❌ BigQuery error for {day}: {e}")
            sys.stdout.flush()

        day += timedelta(days=1)

    # Final summary with total time and total articles
    elapsed_total = time.time() - start_time
    total_articles = sum(len(df) for df in rows)
    sys.stdout.write(f"\r   BigQuery query completed in {elapsed_total:.1f}s - Found {total_articles} total articles\n")
    sys.stdout.flush()

    if not rows:
        return pd.DataFrame()
    
    df_final = pd.concat(rows, ignore_index=True)
    return df_final