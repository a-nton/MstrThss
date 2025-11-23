import json
import time
import numpy as np
from openai import OpenAI, RateLimitError
from config import OPENAI_API_KEY, OPENAI_MODEL, HORIZONS

client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(row, ticker, company_name):
    headlines = row.get("headline_text") or "No news available."
    
    # Add recent price context
    recent_context = ""
    returns_info = []
    if pd.notna(row.get("ret_1d")):
        returns_info.append(f"Yesterday to today: {row['ret_1d']*100:+.2f}%")
    
    if returns_info:
        recent_context = f"\nRecent price action:\n- " + "\n- ".join(returns_info)

    return f"""
You are a financial analyst. Using the news headlines and recent price context provided below, predict the price movement of {ticker} ({company_name}) for three horizons:
- 1d (Next trading day)
- 1w (Next week)
- 1m (Next month)

{recent_context}

Headlines:
{headlines}

Return ONLY valid JSON in this exact format:
{{
  "horizons": {{
    "1d": {{ "prob_up": 0.X, "expected_move_pct": X.X }},
    "1w": {{ "prob_up": 0.X, "expected_move_pct": X.X }},
    "1m": {{ "prob_up": 0.X, "expected_move_pct": X.X }}
  }},
  "justification": "2-3 sentences explaining the key factors driving your prediction."
}}
""".strip()

def ask_llm(prompt, retries=3):
    backoff = 2
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            # Basic JSON extraction
            js_str = text[text.find("{"): text.rfind("}") + 1]
            js = json.loads(js_str)
            
            horizons = js.get("horizons", {})
            out = {}
            for h in HORIZONS.keys():
                hdata = horizons.get(h, {})
                out[f"prob_up_{h}"] = float(hdata.get("prob_up", np.nan))
                out[f"exp_move_pct_{h}"] = abs(float(hdata.get("expected_move_pct", np.nan)))
            out["justification"] = js.get("justification", "")
            return out
            
        except RateLimitError:
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            print(f"   LLM Error: {e}")
            break
            
    # Return NaNs on failure
    result = {}
    for h in HORIZONS.keys():
        result[f"prob_up_{h}"] = np.nan
        result[f"exp_move_pct_{h}"] = np.nan
    result["justification"] = ""
    return result

import pandas as pd
def run_llm(panel, company_name):
    """
    Iterates over the dataframe and calls LLM for days with news.
    """
    results = {f"prob_up_{h}": [] for h in HORIZONS.keys()}
    results.update({f"exp_move_pct_{h}": [] for h in HORIZONS.keys()})
    results["justification"] = []

    # Count only rows with headlines for accurate progress
    rows_with_news = panel[panel["n_headlines"] > 0]
    total_with_news = len(rows_with_news)
    news_counter = 0

    for i, row in panel.iterrows():
        ticker = row["ticker"]

        # Only predict if there are headlines
        if row["n_headlines"] > 0:
            news_counter += 1
            print(f"   [{news_counter}/{total_with_news}] Prompting for {ticker} on {row['date']}...")
            prompt = build_prompt(row, ticker, company_name)
            out = ask_llm(prompt)
            for k in results.keys():
                results[k].append(out.get(k, np.nan if k != "justification" else ""))
        else:
            for k in results.keys():
                results[k].append(np.nan if k != "justification" else "")

    # Attach results to dataframe
    for k, v in results.items():
        panel[k] = v

    return panel