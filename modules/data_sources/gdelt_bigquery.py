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

PLUGIN_NAME = "GDELT BigQuery (Historical)"
PLUGIN_DESCRIPTION = "Full GDELT archive via BigQuery (2015-present, requires GCP credentials)"

def check_bigquery_available():
    """Check if BigQuery is properly configured"""
    try:
        from google.cloud import bigquery

        # Check for credentials
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            return False, "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"

        # Try to create client
        try:
            client = bigquery.Client()
            return True, None
        except Exception as e:
            return False, f"Failed to create BigQuery client: {str(e)}"

    except ImportError:
        return False, "google-cloud-bigquery not installed (run: pip install google-cloud-bigquery)"

def fetch_news(ticker, company_name, start_date=None, end_date=None, max_results=10000):
    """
    Fetch news from GDELT BigQuery.

    Args:
        ticker: Stock symbol (e.g., "META")
        company_name: Company name (e.g., "Meta")
        start_date: datetime.date object (defaults to Sept 1, 2025)
        end_date: datetime.date object (defaults to Sept 30, 2025)
        max_results: Maximum number of results to return

    Returns:
        pd.DataFrame with columns: url, title, calendar_date, ticker

    Note: Requires BigQuery setup and credentials.
          Free tier: 1TB queries/month, but GDELT queries can be large.
    """
    from google.cloud import bigquery

    # Check BigQuery availability
    available, error_msg = check_bigquery_available()
    if not available:
        print(f"❌ BigQuery not available: {error_msg}")
        print("   Setup instructions:")
        print("   1. Create Google Cloud project: https://console.cloud.google.com")
        print("   2. Enable BigQuery API")
        print("   3. Create service account and download JSON key")
        print("   4. Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
        print("   5. Install library: pip install google-cloud-bigquery")
        return pd.DataFrame()

    # Default date range
    if start_date is None:
        start_date = date(2025, 9, 1)
    if end_date is None:
        end_date = date(2025, 9, 30)

    print(f"   Querying GDELT BigQuery from {start_date} to {end_date}...")
    start_time = time.time()

    # Initialize BigQuery client
    client = bigquery.Client()

    # Build query
    # GDELT 2.0 GKG (Global Knowledge Graph) table
    # Contains article metadata including URLs and titles
    query = f"""
    SELECT
        DocumentIdentifier as url,
        DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as calendar_date,
        Themes,
        Persons,
        Organizations,
        Locations,
        Tone
    FROM
        `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)))
            BETWEEN '{start_date.strftime('%Y-%m-%d')}'
            AND '{end_date.strftime('%Y-%m-%d')}'
        AND (
            LOWER(Themes) LIKE '%{ticker.lower()}%'
            OR LOWER(Themes) LIKE '%{company_name.lower()}%'
            OR LOWER(Persons) LIKE '%{company_name.lower()}%'
            OR LOWER(Organizations) LIKE '%{company_name.lower()}%'
        )
        AND SourceCommonName IS NOT NULL
        AND DocumentIdentifier IS NOT NULL
    LIMIT {max_results}
    """

    try:
        # Run query
        sys.stdout.write(f"\r   Executing BigQuery (this may take a moment)...")
        sys.stdout.flush()

        query_job = client.query(query)
        results = query_job.result()

        # Convert to DataFrame
        df = results.to_dataframe()

        if df.empty:
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r   BigQuery completed in {elapsed:.1f}s - No results\n")
            sys.stdout.flush()
            return pd.DataFrame()

        # Extract title from URL (BigQuery GKG doesn't store titles directly)
        # We'll need to fetch titles separately or use URL as title
        df['title'] = df['url'].apply(lambda x: extract_title_from_url(x))
        df['ticker'] = ticker

        # Select only needed columns
        df = df[['url', 'title', 'calendar_date', 'ticker']]

        # Deduplicate
        df = df.drop_duplicates(subset=['url'])

        elapsed = time.time() - start_time
        sys.stdout.write(f"\r   BigQuery completed in {elapsed:.1f}s - Found {len(df)} articles\n")
        sys.stdout.flush()

        # Show query stats
        print(f"   📊 Bytes processed: {query_job.total_bytes_processed / 1e9:.2f} GB")
        print(f"   📊 Bytes billed: {query_job.total_bytes_billed / 1e9:.2f} GB")

        return df

    except Exception as e:
        print(f"\n❌ BigQuery error: {e}")
        return pd.DataFrame()

def extract_title_from_url(url):
    """
    Extract a readable title from URL.
    This is a fallback since BigQuery GKG doesn't store article titles.
    For production, you might want to scrape the actual titles.
    """
    if not url:
        return "Unknown"

    # Remove protocol
    url = url.replace('http://', '').replace('https://', '')

    # Take domain + path
    parts = url.split('/')
    if len(parts) > 1:
        # Try to get article slug
        slug = parts[-1].replace('-', ' ').replace('_', ' ')
        # Remove file extensions
        slug = slug.split('.')[0]
        return slug[:100]  # Truncate

    return url[:100]
