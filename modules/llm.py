import time
import sys
import numpy as np
from openai import OpenAI, RateLimitError
from config import OPENAI_API_KEY, HORIZONS

def run_llm(panel, company_name, llm_config_plugin):
    """
    Iterates over the dataframe and calls LLM for days with news.

    Args:
        panel: DataFrame with aligned news and price data
        company_name: Company name string
        llm_config_plugin: Loaded LLM config plugin module

    Returns:
        DataFrame with predictions added
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = llm_config_plugin.get_model_name()
    temperature = llm_config_plugin.get_temperature()

    results = {f"prob_up_{h}": [] for h in HORIZONS.keys()}
    results.update({f"exp_move_pct_{h}": [] for h in HORIZONS.keys()})
    results["justification"] = []

    # Count only rows with headlines for accurate progress
    rows_with_news = panel[panel["n_headlines"] > 0]
    total_with_news = len(rows_with_news)
    news_counter = 0
    start_time = time.time()

    for _, row in panel.iterrows():
        ticker = row["ticker"]

        # Only predict if there are headlines
        if row["n_headlines"] > 0:
            news_counter += 1

            # Display self-replacing progress
            sys.stdout.write(f"\r   Prompting LLM [{news_counter}/{total_with_news}] {ticker} on {row['date']}...")
            sys.stdout.flush()

            # Build prompt using plugin
            prompt = llm_config_plugin.build_prompt(row, ticker, company_name, HORIZONS)

            # Call LLM with retry logic
            out = ask_llm_with_retry(client, model_name, temperature, prompt, llm_config_plugin)

            for k in results.keys():
                results[k].append(out.get(k, np.nan if k != "justification" else ""))
        else:
            for k in results.keys():
                results[k].append(np.nan if k != "justification" else "")

    # Final summary with total time
    if total_with_news > 0:
        elapsed_total = time.time() - start_time
        sys.stdout.write(f"\r   Prompting LLM [{total_with_news}/{total_with_news}] - Completed in {elapsed_total:.1f}s\n")
        sys.stdout.flush()

    # Attach results to dataframe
    for k, v in results.items():
        panel[k] = v

    return panel

def ask_llm_with_retry(client, model_name, temperature, prompt, llm_config_plugin, retries=3):
    """
    Calls OpenAI API with retry logic and uses plugin to parse response.

    Args:
        client: OpenAI client instance
        model_name: Model identifier (e.g., "gpt-4o-mini")
        temperature: Temperature for generation
        prompt: Prompt string
        llm_config_plugin: Plugin module with parse_response method
        retries: Number of retry attempts

    Returns:
        dict: Parsed response with prob_up_{h}, exp_move_pct_{h}, justification
    """
    backoff = 2

    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = resp.choices[0].message.content

            # Parse using plugin
            out = llm_config_plugin.parse_response(text, HORIZONS)
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