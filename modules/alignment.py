import pandas as pd
from datetime import timedelta
from difflib import SequenceMatcher

def fuzzy_dedupe_headlines(titles, threshold=0.95):
    """
    Remove near-duplicate headlines using fuzzy string matching.
    Keeps the first occurrence of similar headlines.
    """
    if not titles:
        return []

    unique = []
    for title in titles:
        title_str = str(title).strip()
        if not title_str:
            continue

        # Check if this title is too similar to any we've already kept
        is_duplicate = False
        for kept in unique:
            similarity = SequenceMatcher(None, title_str.lower(), kept.lower()).ratio()
            if similarity >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(title_str)

    return unique

def align_news(prices, news):
    """
    Merges daily news into the price dataframe.
    """
    if news.empty:
        prices["headline_text"] = ""
        prices["n_headlines"] = 0
        return prices

    # Ensure format
    if "seendate" in news.columns:
        news["seendate"] = pd.to_datetime(news["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    
    news["news_date"] = news["calendar_date"]

    # Aggregate headlines by date with fuzzy deduplication
    def aggregate_headlines(titles):
        unique_titles = fuzzy_dedupe_headlines(list(titles))
        return " || ".join(unique_titles)

    agg = (
        news.groupby(["ticker", "news_date"])
            .agg(
                headline_text=("title", aggregate_headlines),
                n_headlines=("title", lambda x: len(fuzzy_dedupe_headlines(list(x)))),
            )
            .reset_index()
    )

    # Merge
    out = prices.merge(
        agg,
        left_on=["ticker", "date"],
        right_on=["ticker", "news_date"],
        how="left"
    )

    out["headline_text"] = out["headline_text"].fillna("")
    out["n_headlines"] = out["n_headlines"].fillna(0)
    
    return out