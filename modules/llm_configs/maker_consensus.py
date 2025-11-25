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
Role: Bull Case Analyst (Growth & Opportunity Focused)
Task: Build the strongest possible BULLISH case based on these headlines.

Look for:
- Revenue growth signals and market expansion
- Positive earnings surprises or guidance raises
- Strategic wins (contracts, partnerships, M&A)
- Innovation and competitive advantages
- Market share gains and pricing power
- Strong management execution
- Regulatory tailwinds or resolved uncertainties

Your job is to identify genuine positive catalysts and growth drivers. Be optimistic but not delusional - only highlight signals with substance.

Headlines:
{headlines}

Return valid JSON:
{{
  "signal": "BULLISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "expected_move_pct": 0.0-3.0,
  "reason": "Strongest bull case based on evidence."
}}

Note on magnitude:
- 0.3-0.8% = Minor positive news
- 0.8-1.5% = Moderate catalyst
- 1.5-2.5% = Major catalyst (earnings beat, M&A)
- 2.5-3.0% = Extraordinary event only

Note: You can only return BULLISH or NEUTRAL. If you see no positive catalysts, return NEUTRAL with low confidence.
"""

def prompt_bear_case(row):
    headlines = row.get("headline_text", "No news.")
    return f"""
Role: Bear Case Analyst (Risk & Headwind Focused)
Task: Build the strongest possible BEARISH case based on these headlines.

Look for:
- Revenue misses, slowing growth, or guidance cuts
- Regulatory risks, lawsuits, or compliance issues
- Competitive threats and market share losses
- Execution failures or management turnover
- Rising costs, margin compression, or pricing pressure
- Macroeconomic headwinds affecting the sector
- Overvaluation signals or deteriorating fundamentals

Your job is to identify genuine risks and negative catalysts. Be critical but not paranoid - only highlight substantive concerns.

Headlines:
{headlines}

Return valid JSON:
{{
  "signal": "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "expected_move_pct": 0.0-3.0,
  "reason": "Strongest bear case based on evidence."
}}

Note on magnitude:
- 0.3-0.8% = Minor negative news
- 0.8-1.5% = Moderate risk
- 1.5-2.5% = Major risk (revenue miss, lawsuit)
- 2.5-3.0% = Extraordinary event only

Note: You can only return BEARISH or NEUTRAL. If you see no significant risks, return NEUTRAL with low confidence.
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
Role: Technical Analyst (Price Action & Indicators Specialist)
Task: Analyze technical indicators and price momentum to identify directional signals.

{context}

Guidelines:
- RSI <30 = Oversold (potential bounce), >70 = Overbought (potential pullback)
- MACD > Signal = Bullish momentum, MACD < Signal = Bearish momentum
- Price above MAs = Uptrend, below MAs = Downtrend
- High volatility = Lower confidence (less predictable)
- Only predict moves >1.5% when multiple indicators strongly align

Return valid JSON:
{{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "expected_move_pct": 0.0-2.0,
  "reason": "Which indicators align and what they suggest."
}}

Note: Base magnitude on volatility. If 20d volatility is 1.5%, a normal move is ±1.5%.
"""

def prompt_sentiment(row):
    headlines = row.get("headline_text", "No news.")
    return f"""
Role: Market Psychologist (Sentiment & Behavioral Analyst)
Task: Analyze how these headlines will affect market psychology and investor behavior.

Consider both perspectives:

Momentum view:
- Positive news flow can drive buying pressure
- Negative news creates fear and selling
- Volume and intensity of coverage matters

Contrarian view:
- Excessive optimism may signal overbought conditions
- Extreme pessimism may signal oversold conditions
- "Everyone knows" news is often priced in
- Hype and FOMO can precede reversals

Weigh the immediate emotional reaction against contrarian signals. Consider headline volume, tone, and whether sentiment appears overdone in either direction.

Headlines:
{headlines}

Return valid JSON:
{{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "expected_move_pct": 0.0-3.0,
  "reason": "Justification based on market psychology."
}}

Note: Sentiment-driven moves are typically <2%. Only predict >2% for extreme sentiment shifts.
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

    # --- REFINED CONSENSUS LOGIC (4-Agent Weighted Ensemble) ---

    # Convert each agent's signal + confidence into a directional score
    # Score range: -1 (bearish) to +1 (bullish)
    def agent_to_score(agent):
        signal = agent.get("signal", "NEUTRAL").upper()
        confidence = float(agent.get("confidence", 0.5))

        if signal == "BULLISH":
            return confidence  # 0.0 to 1.0
        elif signal == "BEARISH":
            return -confidence  # 0.0 to -1.0
        else:
            return 0.0  # Neutral = no opinion

    bull_score = agent_to_score(bull)
    bear_score = agent_to_score(bear)
    tech_score = agent_to_score(tech)
    sent_score = agent_to_score(sent)

    # Weighted average of all four agents
    # Equal weights by default (can be tuned later)
    weights = {"bull": 1.0, "bear": 1.0, "technical": 1.0, "sentiment": 1.0}
    total_weight = sum(weights.values())

    ensemble_score = (
        weights["bull"] * bull_score +
        weights["bear"] * bear_score +
        weights["technical"] * tech_score +
        weights["sentiment"] * sent_score
    ) / total_weight

    # Ensemble score is now in range [-1, +1]
    # Map to probability: -1 → 0%, 0 → 50%, +1 → 100%
    prob = 0.5 + (0.5 * ensemble_score)

    # Clamp to [0, 1] for safety
    prob = max(0.0, min(1.0, prob))

    # Determine consensus description
    votes_list = [
        bull.get("signal", "NEUTRAL").upper(),
        bear.get("signal", "NEUTRAL").upper(),
        tech.get("signal", "NEUTRAL").upper(),
        sent.get("signal", "NEUTRAL").upper()
    ]

    vote_counts = {"BULLISH": votes_list.count("BULLISH"),
                   "BEARISH": votes_list.count("BEARISH"),
                   "NEUTRAL": votes_list.count("NEUTRAL")}

    if ensemble_score > 0.15:
        final_reason = f"Ensemble Bullish (score: {ensemble_score:.2f}). Bull: {bull_score:.2f}, Bear: {bear_score:.2f}, Tech: {tech_score:.2f}, Sent: {sent_score:.2f}"
    elif ensemble_score < -0.15:
        final_reason = f"Ensemble Bearish (score: {ensemble_score:.2f}). Bull: {bull_score:.2f}, Bear: {bear_score:.2f}, Tech: {tech_score:.2f}, Sent: {sent_score:.2f}"
    else:
        final_reason = f"Ensemble Neutral (score: {ensemble_score:.2f}). Bull: {bull_score:.2f}, Bear: {bear_score:.2f}, Tech: {tech_score:.2f}, Sent: {sent_score:.2f}"

    # 3. Determine Magnitude (Average of ALL agents, weighted by abs(score))
    moves = []
    weights_mag = []

    if abs(bull_score) > 0.01:
        moves.append(float(bull.get("expected_move_pct", 0)))
        weights_mag.append(abs(bull_score))

    if abs(bear_score) > 0.01:
        moves.append(float(bear.get("expected_move_pct", 0)))
        weights_mag.append(abs(bear_score))

    if abs(tech_score) > 0.01:
        moves.append(float(tech.get("expected_move_pct", 0)))
        weights_mag.append(abs(tech_score))

    if abs(sent_score) > 0.01:
        moves.append(float(sent.get("expected_move_pct", 0)))
        weights_mag.append(abs(sent_score))

    if moves and weights_mag:
        avg_move_raw = sum(m * w for m, w in zip(moves, weights_mag)) / sum(weights_mag)
    else:
        avg_move_raw = 0.0

    # 4. Format Output with VOLATILITY CALIBRATION
    output = {}

    for h_name, h_days in horizons.items():
        output[f"prob_up_{h_name}"] = prob

        # Apply volatility-based calibration to magnitude
        # This fixes the 3.38x overestimation problem
        calibrated_magnitude = calibrate_magnitude(
            avg_move_raw,
            vol_daily,
            horizon_days=h_days,
            dampening_factor=0.35  # Empirical: reduces 5.5% → 1.9% (closer to 1.6% actual)
        )
        output[f"exp_move_pct_{h_name}"] = calibrated_magnitude

    # Export details for visualization (4 agents now)
    output["justification"] = final_reason
    output["agent_bull"] = f"{votes_list[0]} (conf:{bull.get('confidence', 0):.2f}) - {bull.get('reason', 'N/A')}"
    output["agent_bear"] = f"{votes_list[1]} (conf:{bear.get('confidence', 0):.2f}) - {bear.get('reason', 'N/A')}"
    output["agent_technical"] = f"{votes_list[2]} (conf:{tech.get('confidence', 0):.2f}) - {tech.get('reason', 'N/A')}"
    output["agent_sentiment"] = f"{votes_list[3]} (conf:{sent.get('confidence', 0):.2f}) - {sent.get('reason', 'N/A')}"

    # Keep old column names for backward compatibility with existing viz
    output["agent_fundamental"] = output["agent_bull"]  # Map Bull to old "fundamental" column
    output["agent_risk"] = output["agent_sentiment"]    # Map Sentiment to old "risk" column

    await client.close()
    return output

def run_analysis_async(row, model_name, horizons):
    return asyncio.run(run_council(row, model_name, horizons))