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
import re
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


def clean_company_name(name):
    """
    Extracts meaningful search phrases from a company name.
    1. Splits by parentheses (aliases).
    2. Removes corporate suffixes (Inc, Corp, Ltd, etc).
    3. Preserves multi-word phrases (e.g., 'General Motors').
    """
    # 1. Split by parens to handle "Alphabet (Google)" -> ["Alphabet", "Google"]
    raw_parts = re.split(r'[\(\)]', name)
    
    # 2. Define corporate stopwords to strip (case insensitive)
    # Note: We use \b to ensure whole word matching
    stopwords = [
        r"\bInc\.?\b", r"\bCorp\.?\b", r"\bCorporation\b", r"\bLtd\.?\b", 
        r"\bLimited\b", r"\bCo\.?\b", r"\bCompany\b", r"\bPLC\b", 
        r"\bGroup\b", r"\bHoldings\b", r"\bSA\b", r"\bAG\b", r"\bNV\b"
    ]
    
    cleaned_phrases = []
    
    for part in raw_parts:
        clean = part.strip()
        if not clean:
            continue
            
        # Remove stopwords
        for stop in stopwords:
            clean = re.sub(stop, "", clean, flags=re.IGNORECASE)
            
        # Clean up extra spaces left behind
        clean = " ".join(clean.split())
        
        # Only keep if significant length (> 2 chars)
        if len(clean) > 2:
            cleaned_phrases.append(clean)
            
    return list(set(cleaned_phrases))


def fetch_news(ticker, company_name, start_date=None, end_date=None, max_per_day=1000):
    """
    Fetch news from GDELT BigQuery using "Smart Regex" matching.
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
    
    # --- 1. PREPARE SEARCH TERMS ---
    # Extract phrases like ["Alphabet", "Google"] or ["General Motors"]
    search_phrases = clean_company_name(company_name)
    
    # Always include Ticker
    if ticker not in search_phrases:
        search_phrases.append(ticker)
        
    print(f"   Querying GDELT BigQuery for: {search_phrases}")
    
    # --- 2. BUILD BIGQUERY REGEX ---
    # We want to match whole words/phrases.
    # "General Motors" should match "General Motors" OR "General_Motors" (GDELT format)
    # Logic: (?i)(?:^|[^a-z])(Phrase One|Phrase Two|Ticker)(?:$|[^a-z])
    
    regex_parts = []
    for phrase in search_phrases:
        # Escape regex chars, then allow spaces to match literal space OR underscore
        # e.g. "General Motors" -> "General[_\s]Motors"
        escaped = re.escape(phrase)
        flexible_space = escaped.replace("\ ", r"[_\s]")
        regex_parts.append(flexible_space)
        
    joined_regex = "|".join(regex_parts)
    bq_smart_regex = fr'(?i)(?:^|[^a-z])({joined_regex})(?:$|[^a-z])'
    
    while day <= end_date:
        day_counter += 1
        
        sys.stdout.write(f"\r   Executing BigQuery [{day_counter}/{total_days}] {day}...")
        sys.stdout.flush()
        
        date_str = day.strftime('%Y-%m-%d')
        
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
            
            -- SMART SEARCH: Match any of our valid phrases in Themes, Persons, or Orgs
            AND (
                REGEXP_CONTAINS(Themes, r'{bq_smart_regex}')
                OR REGEXP_CONTAINS(Persons, r'{bq_smart_regex}')
                OR REGEXP_CONTAINS(Organizations, r'{bq_smart_regex}')
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
                
                # --- 3. PYTHON-SIDE VALIDATION FILTER ---
                # Re-verify title contains at least one of our phrases
                # This filters out articles where the company was only mentioned in the footer/tags
                
                mask = pd.Series([False] * len(df), index=df.index)
                
                for phrase in search_phrases:
                    # Strict word boundary check for Ticker, looser for multi-word phrases
                    if phrase == ticker:
                        pattern = rf'\b{re.escape(phrase)}\b'
                    else:
                        pattern = re.escape(phrase)
                        
                    mask |= df["title"].str.contains(pattern, case=False, regex=True, na=False)
                
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