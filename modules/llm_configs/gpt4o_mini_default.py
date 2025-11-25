"""
GPT-4o-mini Default Configuration

Required interface for all LLM config plugins:
- PLUGIN_NAME: str - Display name for the menu
- PLUGIN_DESCRIPTION: str - Brief description
- build_prompt(row, ticker, company_name, horizons) -> str
- parse_response(response_text, horizons) -> dict
- get_model_name() -> str
- get_temperature() -> float
"""

import json
import numpy as np
import pandas as pd

PLUGIN_NAME = "GPT-4o-mini (Financial Analyst)"
PLUGIN_DESCRIPTION = "Standard financial analyst prompt with justification"

def get_model_name():
    """Return the OpenAI model identifier"""
    return "gpt-4o-mini"

def get_temperature():
    """Return the temperature for generation (0.0 = deterministic)"""
    return 0.0

def build_prompt(row, ticker, company_name, horizons):
    """
    Build the prompt for the LLM.

    Args:
        row: DataFrame row with headline_text, ret_1d, etc.
        ticker: Stock symbol
        company_name: Company name
        horizons: dict like {"1d": 1, "1w": 5, "1m": 21}

    Returns:
        str: The complete prompt
    """
    headlines = row.get("headline_text") or "No news available."

    # Add recent price context
    recent_context = ""
    returns_info = []
    if pd.notna(row.get("ret_1d")):
        returns_info.append(f"Yesterday to today: {row['ret_1d']*100:+.2f}%")

    if returns_info:
        recent_context = f"\nRecent price action:\n- " + "\n- ".join(returns_info)

    # Build horizon descriptions
    horizon_list = "\n".join([f"- {h} (Next {horizons[h]} trading day{'s' if horizons[h] > 1 else ''})"
                               for h in horizons.keys()])

    return f"""
You are a financial analyst. Using the news headlines and recent price context provided below, predict the price movement of {ticker} ({company_name}) for three horizons:
{horizon_list}

{recent_context}

Headlines:
{headlines}

Return ONLY valid JSON in this exact format:
{{
  "horizons": {{
    {', '.join([f'"{h}": {{"prob_up": 0.X, "expected_move_pct": X.X}}' for h in horizons.keys()])}
  }},
  "justification": "2-3 sentences explaining the key factors driving your prediction."
}}
""".strip()

def parse_response(response_text, horizons):
    """
    Parse the LLM response into a standardized dict.

    Args:
        response_text: Raw text from LLM
        horizons: dict of horizon keys

    Returns:
        dict with keys: prob_up_{h}, exp_move_pct_{h}, justification
    """
    try:
        # Extract JSON
        js_str = response_text[response_text.find("{"): response_text.rfind("}") + 1]
        js = json.loads(js_str)

        horizons_data = js.get("horizons", {})
        result = {}

        for h in horizons.keys():
            hdata = horizons_data.get(h, {})
            result[f"prob_up_{h}"] = float(hdata.get("prob_up", np.nan))
            result[f"exp_move_pct_{h}"] = abs(float(hdata.get("expected_move_pct", np.nan)))

        result["justification"] = js.get("justification", "")
        return result

    except Exception as e:
        print(f"   Parse error: {e}")
        # Return NaNs on failure
        result = {}
        for h in horizons.keys():
            result[f"prob_up_{h}"] = np.nan
            result[f"exp_move_pct_{h}"] = np.nan
        result["justification"] = ""
        return result
