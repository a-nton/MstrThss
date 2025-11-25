import time
import sys
import pandas as pd
import numpy as np
from openai import OpenAI, RateLimitError
from config import OPENAI_API_KEY, HORIZONS

def run_llm(panel, company_name, llm_config_plugin):
    """
    Iterates over the dataframe and calls LLM.
    Supports both standard plugins and MAKER async plugins.

    Args:
        panel: DataFrame with aligned news and price data
        company_name: Company name string
        llm_config_plugin: Loaded LLM config plugin module

    Returns:
        DataFrame with predictions added (concatenated with original data)
    """
    # Initialize Standard Client (for sync plugins)
    client = OpenAI(api_key=OPENAI_API_KEY)
    model_name = llm_config_plugin.get_model_name()
    
    # We will collect a list of dicts, then create a DataFrame
    # This allows dynamic columns from the MAKER plugin (e.g. agent_risk, agent_tech)
    results_list = []

    # Filter for rows with news to count progress correctly
    rows_with_news = panel[panel["n_headlines"] > 0]
    total_with_news = len(rows_with_news)
    news_counter = 0
    start_time = time.time()

    for index, row in panel.iterrows():
        row_result = {}
        
        # Only predict if there are headlines
        if row["n_headlines"] > 0:
            news_counter += 1
            
            # Display self-replacing progress
            sys.stdout.write(f"\r   Prompting LLM [{news_counter}/{total_with_news}] {row['ticker']} on {row['date']}...")
            sys.stdout.flush()

            # Check for Async/MAKER capability in the plugin
            if hasattr(llm_config_plugin, 'run_analysis_async'):
                # --- MAKER PATH (Async Agents) ---
                try:
                    # The plugin handles the async event loop internally via a wrapper
                    out = llm_config_plugin.run_analysis_async(row, model_name, HORIZONS)
                    row_result = out
                except Exception as e:
                    print(f" Error in Async Agent: {e}")
                    # Fallback to NaNs
                    row_result = {f"prob_up_{h}": np.nan for h in HORIZONS.keys()}
                    row_result["justification"] = f"Agent Error: {str(e)}"
            else:
                # --- STANDARD PATH (Sync Single Prompt) ---
                temperature = llm_config_plugin.get_temperature()
                prompt = llm_config_plugin.build_prompt(row, row['ticker'], company_name, HORIZONS)
                out = ask_llm_with_retry(client, model_name, temperature, prompt, llm_config_plugin)
                row_result = out
        else:
            # No news filler
            row_result = {f"prob_up_{h}": np.nan for h in HORIZONS.keys()}
            row_result.update({f"exp_move_pct_{h}": np.nan for h in HORIZONS.keys()})
            row_result["justification"] = "No news available."

        results_list.append(row_result)

    # Final summary
    if total_with_news > 0:
        elapsed = time.time() - start_time
        sys.stdout.write(f"\r   Prompting LLM [{total_with_news}/{total_with_news}] - Completed in {elapsed:.1f}s\n")
    
    # Create DataFrame from results list
    df_results = pd.DataFrame(results_list)
    
    # Reset index of panel to match simple concat (safety measure)
    panel = panel.reset_index(drop=True)
    
    # Concatenate original panel with new results
    # We use concat to preserve all new columns (agent_fundamental, etc.)
    final_df = pd.concat([panel, df_results], axis=1)

    return final_df

def ask_llm_with_retry(client, model_name, temperature, prompt, llm_config_plugin, retries=3):
    """
    Calls OpenAI API with retry logic (Standard Path).
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
    result["justification"] = "API Error or Parse Failure"
    return result