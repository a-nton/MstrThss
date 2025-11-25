"""
Google News RSS Data Source Plugin

Required interface for all data source plugins:
- PLUGIN_NAME: str - Display name for the menu
- PLUGIN_DESCRIPTION: str - Brief description
- fetch_news(ticker, company_name, start_date, end_date) -> pd.DataFrame
  Returns DataFrame with columns: url, title, calendar_date, ticker
"""

import feedparser
import requests
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from urllib.parse import quote
from bs4 import BeautifulSoup

PLUGIN_NAME = "Google News RSS"
PLUGIN_DESCRIPTION = "Google News RSS feed with approximate date filtering (after:/before: operators)"

def fetch_article_content(url, timeout=5):
    """
    Attempts to fetch article content from URL.
    Returns content text or empty string on failure.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content.strip()
    except Exception:
        return ""

def fetch_news(ticker, company_name, start_date=None, end_date=None, num_articles=100):
    """
    Fetch news from Google News RSS feed.

    Args:
        ticker: Stock symbol (e.g., "META")
        company_name: Company name (e.g., "Meta")
        start_date: datetime.date object (uses 'after:' operator in query)
        end_date: datetime.date object (uses 'before:' operator in query)
        num_articles: Max articles to fetch (default 100)

    Returns:
        pd.DataFrame with columns: url, title, calendar_date, ticker
        Note: calendar_date is parsed from RSS 'published' field

    Note: Google News RSS date filtering is approximate and may not be exact.
          Results are also filtered client-side by publication date.
    """
    # Build date filter string for query
    date_filter = ""
    if start_date and end_date:
        date_filter = f" after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
    elif start_date:
        date_filter = f" after:{start_date.strftime('%Y-%m-%d')}"
    elif end_date:
        date_filter = f" before:{end_date.strftime('%Y-%m-%d')}"

    # Build search queries with date filter
    queries = [
        f"{ticker} stock{date_filter}",
        f"{company_name} stock{date_filter}",
        f"{ticker} {company_name}{date_filter}",
        f"{company_name} news{date_filter}"
    ]

    all_articles = []
    start_time = time.time()

    for query_idx, query in enumerate(queries, 1):
        # Display progress
        sys.stdout.write(f"\r   Querying Google News RSS [{query_idx}/{len(queries)}] '{query}'...")
        sys.stdout.flush()

        rss_url = f"https://news.google.com/rss/search?q={quote(query)}"

        try:
            feed = feedparser.parse(rss_url)
            articles_from_feed = feed.entries[:num_articles // len(queries)]

            for item in articles_from_feed:
                # Parse publication date
                try:
                    pub_date = datetime(*item.published_parsed[:6]).date()
                except Exception:
                    pub_date = datetime.now().date()

                # Filter by date range if provided
                if start_date and pub_date < start_date:
                    continue
                if end_date and pub_date > end_date:
                    continue

                article = {
                    "url": item.link,
                    "title": item.title,
                    "calendar_date": pub_date,
                    "ticker": ticker,
                    "published": item.published if hasattr(item, 'published') else str(pub_date)
                }

                all_articles.append(article)

        except Exception as e:
            sys.stdout.write("\n")
            print(f"   ! Error fetching query '{query}': {e}")

    # Final summary
    elapsed_total = time.time() - start_time
    sys.stdout.write(f"\r   Querying Google News RSS [{len(queries)}/{len(queries)}] - Completed in {elapsed_total:.1f}s\n")
    sys.stdout.flush()

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # Deduplicate by URL and title (case-insensitive title comparison)
    df = df.drop_duplicates(subset=["url"])
    df["title_lower"] = df["title"].str.lower()
    df = df.drop_duplicates(subset=["title_lower"])
    df = df.drop(columns=["title_lower"])

    # Filter by title relevance (must contain ticker or company name)
    mask = df["title"].str.contains(ticker, case=False, na=False) | \
           df["title"].str.contains(company_name, case=False, na=False)
    df = df[mask]

    print(f"   -> Found {len(df)} relevant articles.")

    return df
