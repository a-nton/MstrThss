"""
MAKER Framework: Council of Agents (Decomposition + Consensus)

Agents:
1. Bull Case Analyst (Seeks positive catalysts and growth drivers)
2. Bear Case Analyst (Seeks risks, headwinds, and negative catalysts)
3. Technician (Price Trend Analysis)
4. Psychologist (Market Sentiment & Behavioral Analysis)

Strategy:
- Asynchronous parallel execution
- Weighted ensemble averaging (all voices matter)
- Bull/Bear analysts create natural tension and balanced analysis
"""

import asyncio
import json
import os
import numpy as np
from openai import AsyncOpenAI
from config import OPENAI_API_KEY
from modules.volatility_calibration import get_historical_volatility, calibrate_magnitude
from modules.technical_indicators import get_technical_summary

PLUGIN_NAME = "MAKER Consensus (Bull vs Bear)"
PLUGIN_DESCRIPTION = "4-Agent Ensemble (Bull, Bear, Technical, Sentiment) with Balanced Analysis"

def get_model_name():
    return "gpt-4o-mini"

def get_temperature():
    return 0.0  # Deterministic for agents

# --- PROMPTS ---

def prompt_bull_case(row):
    headlines = row.get("headline_text", "No news.")
    return f"""
Role: Financial Analyst
Task: Analyze the following headlines and assess their impact on stock price direction and magnitude for three time horizons: 1 day, 1 week, and 1 month.

In your analysis, consider potential positive developments and growth opportunities alongside any risks. Evaluate the strength of catalysts and competitive positioning.

Consider how the impact may differ across time horizons:
- Short-term (1d): Immediate market reaction
- Medium-term (1w): How the news develops over a week
- Long-term (1m): Sustained impact and fundamental changes

Headlines:
{headlines}

Return valid JSON:
{{
  "1d": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1w": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1m": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "reason": "Your assessment explaining differences across horizons if any."
}}

For expected_move_pct, estimate the magnitude in percentage terms based on the news significance and historical context.
"""

def prompt_bear_case(row):
    headlines = row.get("headline_text", "No news.")
    return f"""
Role: Financial Analyst
Task: Analyze the following headlines and assess their impact on stock price direction and magnitude for three time horizons: 1 day, 1 week, and 1 month.

In your analysis, consider potential risks and headwinds alongside any opportunities. Evaluate execution challenges and competitive threats.

Consider how the impact may differ across time horizons:
- Short-term (1d): Immediate market reaction
- Medium-term (1w): How risks develop over a week
- Long-term (1m): Sustained impact and fundamental deterioration

Headlines:
{headlines}

Return valid JSON:
{{
  "1d": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1w": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1m": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "reason": "Your assessment explaining differences across horizons if any."
}}

For expected_move_pct, estimate the magnitude in percentage terms based on the news significance and historical context.
"""

def prompt_technical(row):
    r1 = row.get('ret_1d', 0)
    r2 = row.get('ret_2d', 0)
    r3 = row.get('ret_3d', 0)
    ticker = row.get('ticker', 'UNKNOWN')

    # Fetch technical indicators
    tech_summary = get_technical_summary(ticker)

    context = f"""Recent Returns:
1-Day: {r1:.2%}
2-Day: {r2:.2%}
3-Day: {r3:.2%}

{tech_summary}"""

    return f"""
Role: Financial Analyst
Task: Analyze the technical indicators and price momentum to assess directional signals for three time horizons: 1 day, 1 week, and 1 month.

{context}

Consider how price action, momentum indicators, and volatility patterns inform the current market structure and potential price movements across different timeframes.

Consider how technical signals may differ across time horizons:
- Short-term (1d): Oversold/overbought conditions, immediate momentum
- Medium-term (1w): Trend strength, moving average crossovers
- Long-term (1m): Major trend direction, support/resistance levels

Return valid JSON:
{{
  "1d": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1w": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1m": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "reason": "Your analysis explaining technical signals across horizons."
}}

For expected_move_pct, estimate the magnitude in percentage terms based on the strength of technical signals and typical volatility
"""

def prompt_sentiment(row):
    headlines = row.get("headline_text", "No news.")
    return f"""
Role: Financial Analyst
Task: Analyze how these headlines will affect market psychology and investor behavior across three time horizons: 1 day, 1 week, and 1 month.

Consider both immediate emotional reactions and potential contrarian signals. Evaluate headline volume, tone, and whether sentiment appears balanced or overdone.

Consider how sentiment impact may differ across time horizons:
- Short-term (1d): Initial emotional reaction, headline-driven moves
- Medium-term (1w): Sentiment persistence vs. fade, narrative development
- Long-term (1m): Fundamental reassessment, lasting perception changes

Headlines:
{headlines}

Return valid JSON:
{{
  "1d": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1w": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "1m": {{"signal": "BULLISH|BEARISH|NEUTRAL", "confidence": 0.0-1.0, "expected_move_pct": <number>}},
  "reason": "Your assessment of sentiment evolution across horizons."
}}

For expected_move_pct, estimate the magnitude in percentage terms based on the sentiment intensity and how it may evolve.
"""

# --- ASYNC AGENT HANDLER ---

async def call_agent(client, model, prompt, agent_name):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return agent_name, json.loads(content)
    except Exception as e:
        # Return neutral fallback on error
        fallback = {"signal": "NEUTRAL", "confidence": 0, "expected_move_pct": 0, "reason": f"Error: {str(e)}"}
        return agent_name, fallback

async def run_council(row, model_name, horizons):
    """
    Orchestrates the parallel execution of 4 agents: Bull, Bear, Technical, Sentiment
    """
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Fetch historical volatility for calibration
    ticker = row.get('ticker', 'UNKNOWN')
    vol_daily, vol_annual = get_historical_volatility(ticker)

    # 1. Parallel Execution (Bull vs Bear + Technical + Sentiment)
    results = await asyncio.gather(
        call_agent(client, model_name, prompt_bull_case(row), "bull"),
        call_agent(client, model_name, prompt_bear_case(row), "bear"),
        call_agent(client, model_name, prompt_technical(row), "technical"),
        call_agent(client, model_name, prompt_sentiment(row), "sentiment")
    )

    data = {name: res for name, res in results}

    bull = data['bull']
    bear = data['bear']
    tech = data['technical']
    sent = data['sentiment']

    # --- HORIZON-SPECIFIC CONSENSUS LOGIC (4-Agent Weighted Ensemble) ---

    # Convert each agent's horizon-specific signal + confidence into a directional score
    # Score range: -1 (bearish) to +1 (bullish)
    def agent_to_score(agent_horizon_data):
        signal = agent_horizon_data.get("signal", "NEUTRAL").upper()
        confidence = float(agent_horizon_data.get("confidence", 0.5))

        if signal == "BULLISH":
            return confidence  # 0.0 to 1.0
        elif signal == "BEARISH":
            return -confidence  # 0.0 to -1.0
        else:
            return 0.0  # Neutral = no opinion

    # Agent weights (equal by default)
    weights = {"bull": 1.0, "bear": 1.0, "technical": 1.0, "sentiment": 1.0}
    total_weight = sum(weights.values())

    # 2. Calculate ensemble for EACH horizon
    output = {}

    for h_name, h_days in horizons.items():
        # Get horizon-specific data from each agent
        bull_h = bull.get(h_name, {})
        bear_h = bear.get(h_name, {})
        tech_h = tech.get(h_name, {})
        sent_h = sent.get(h_name, {})

        # Calculate scores for this horizon
        bull_score = agent_to_score(bull_h)
        bear_score = agent_to_score(bear_h)
        tech_score = agent_to_score(tech_h)
        sent_score = agent_to_score(sent_h)

        # Weighted ensemble for this horizon
        ensemble_score = (
            weights["bull"] * bull_score +
            weights["bear"] * bear_score +
            weights["technical"] * tech_score +
            weights["sentiment"] * sent_score
        ) / total_weight

        # Map to probability: -1 → 0%, 0 → 50%, +1 → 100%
        prob = 0.5 + (0.5 * ensemble_score)
        prob = max(0.0, min(1.0, prob))

        output[f"prob_up_{h_name}"] = prob

        # 3. Determine Magnitude (simple average, directionless)
        # Collect all magnitude predictions from agents
        moves = []

        # Include all agent predictions (magnitude is directionless)
        bull_move = float(bull_h.get("expected_move_pct", 0))
        bear_move = float(bear_h.get("expected_move_pct", 0))
        tech_move = float(tech_h.get("expected_move_pct", 0))
        sent_move = float(sent_h.get("expected_move_pct", 0))

        if bull_move > 0:
            moves.append(bull_move)
        if bear_move > 0:
            moves.append(bear_move)
        if tech_move > 0:
            moves.append(tech_move)
        if sent_move > 0:
            moves.append(sent_move)

        # Simple average (no weighting by confidence)
        if moves:
            avg_move_raw = sum(moves) / len(moves)
        else:
            avg_move_raw = 0.0

        # No dampening applied - use raw prediction
        final_magnitude = calibrate_magnitude(
            avg_move_raw,
            vol_daily,
            horizon_days=h_days,
            dampening_factor=0.35
        )
        output[f"exp_move_pct_{h_name}"] = final_magnitude

    # 4. Export agent details (use 1d horizon for display)
    bull_1d = bull.get("1d", {})
    bear_1d = bear.get("1d", {})
    tech_1d = tech.get("1d", {})
    sent_1d = sent.get("1d", {})

    output["justification"] = f"Multi-horizon ensemble. Bull: {bull.get('reason', 'N/A')} | Bear: {bear.get('reason', 'N/A')}"
    output["agent_bull"] = f"{bull_1d.get('signal', 'NEUTRAL')} (conf:{bull_1d.get('confidence', 0):.2f}) - {bull.get('reason', 'N/A')}"
    output["agent_bear"] = f"{bear_1d.get('signal', 'NEUTRAL')} (conf:{bear_1d.get('confidence', 0):.2f}) - {bear.get('reason', 'N/A')}"
    output["agent_technical"] = f"{tech_1d.get('signal', 'NEUTRAL')} (conf:{tech_1d.get('confidence', 0):.2f}) - {tech.get('reason', 'N/A')}"
    output["agent_sentiment"] = f"{sent_1d.get('signal', 'NEUTRAL')} (conf:{sent_1d.get('confidence', 0):.2f}) - {sent.get('reason', 'N/A')}"

    # Keep old column names for backward compatibility with existing viz
    output["agent_fundamental"] = output["agent_bull"]  # Map Bull to old "fundamental" column
    output["agent_risk"] = output["agent_sentiment"]    # Map Sentiment to old "risk" column

    await client.close()
    return output

def run_analysis_async(row, model_name, horizons):
    return asyncio.run(run_council(row, model_name, horizons))