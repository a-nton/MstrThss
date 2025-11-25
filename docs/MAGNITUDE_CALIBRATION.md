# Magnitude Overestimation: Diagnosis & Solution

## Problem Statement

**Current Issue:**
- **Predicted magnitude (average)**: 5.46%
- **Actual magnitude (average)**: 1.61%
- **Overestimation ratio**: **3.38x**

Agents are consistently predicting moves 3-4x larger than reality.

## Root Causes

### 1. LLM Anchoring Bias
LLMs (especially GPT-4o-mini) tend to anchor on the upper bounds of provided ranges:
- Prompt says: `"expected_move_pct": 0.0-10.0`
- Agents think: "Big news = big move" → return 5-8%
- Reality: Most daily moves are <2%

### 2. No Historical Context
Agents don't know that:
- Average 1-day stock move: **~1.5%**
- Median 1-day move: **~1.0%**
- 95th percentile move: **~3.5%**

### 3. News Impact Overestimation
Headlines sound important to LLMs, but markets often price them in quickly:
- "$7.5B investment announcement" → Predicted: 5.9%, Actual: 0.3%
- Markets are semi-efficient, most news has limited impact

### 4. Missing Volatility Calibration
Unlike the LSTM predictor, current system doesn't:
- Calculate historical volatility (20-day std)
- Adjust magnitude based on volatility regime
- Use realized volatility to constrain predictions

---

## Solution: Multi-Layered Calibration

### **Approach 1: Adjust Prompt Ranges (Quick Fix)**

**Update agent prompts to realistic ranges:**

```python
# Current (unrealistic)
"expected_move_pct": 0.0-10.0

# New (calibrated to reality)
"expected_move_pct": 0.0-3.0  # 1-day horizon
"expected_move_pct": 0.0-5.0  # 1-week horizon
"expected_move_pct": 0.0-8.0  # 1-month horizon
```

**Why this works:**
- 3% is ~95th percentile for 1-day moves
- Forces agents to be realistic
- Still allows for extreme events (earnings, black swans)

---

### **Approach 2: Historical Volatility Scaling (Recommended)**

**Calculate realized volatility and scale predictions:**

```python
def calculate_historical_volatility(ticker_prices, window=20):
    """
    Calculate annualized volatility based on recent returns.
    """
    returns = ticker_prices.pct_change().dropna()
    volatility_daily = returns.rolling(window).std().iloc[-1]
    volatility_annualized = volatility_daily * np.sqrt(252)
    return volatility_daily, volatility_annualized

def scale_magnitude_by_volatility(predicted_move, volatility_daily, horizon_days=1):
    """
    Scale LLM prediction to realistic magnitude using historical volatility.

    Args:
        predicted_move: LLM prediction (e.g., 5.5%)
        volatility_daily: Daily volatility (e.g., 0.015 = 1.5%)
        horizon_days: Prediction horizon (1, 5, 21)

    Returns:
        Calibrated magnitude
    """
    # Expected move = volatility * sqrt(horizon) * confidence_multiplier
    # confidence_multiplier ~ 1.0 for typical news, up to 2.0 for major events

    # Normalize LLM prediction to confidence multiplier
    # If LLM says 5%, and range is 0-10%, that's 50% confidence → 1.0x multiplier
    max_predicted = 10.0  # Current prompt max
    confidence_multiplier = (predicted_move / max_predicted) * 2.0

    # Calculate realistic magnitude
    expected_move = volatility_daily * np.sqrt(horizon_days) * confidence_multiplier

    return expected_move * 100  # Return as percentage
```

**Example:**
- LLM predicts: 5.5% move
- Historical volatility: 1.5% daily
- Horizon: 1 day
- Confidence multiplier: (5.5 / 10.0) * 2.0 = 1.1x
- **Calibrated magnitude: 1.5% * √1 * 1.1 = 1.65%**

---

### **Approach 3: LSTM-Style Magnitude Model (Advanced)**

**Learn magnitude from historical data:**

```python
def learn_magnitude_scaling(historical_predictions, historical_actuals):
    """
    Learn the relationship between predicted and actual magnitudes.
    Returns a scaling factor.
    """
    pred_mag = np.abs(historical_predictions)
    actual_mag = np.abs(historical_actuals)

    # Linear regression: actual = alpha * predicted
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(pred_mag.reshape(-1, 1), actual_mag)

    scaling_factor = model.coef_[0]
    return scaling_factor

# Apply to new predictions
scaling_factor = 0.30  # Example: actual = 0.30 * predicted (roughly 1.6 / 5.5)
calibrated_move = predicted_move * scaling_factor
```

**Current data suggests:**
- `scaling_factor ≈ 1.61 / 5.46 = 0.295`
- Apply 0.30x multiplier to all agent predictions

---

## Recommended Implementation

**Hybrid approach combining all three:**

### Step 1: Update Prompt Ranges (Immediate)

```python
def prompt_bull_case(row):
    return f"""
    ...
    Return valid JSON:
    {{
      "signal": "BULLISH" | "NEUTRAL",
      "confidence": 0.0-1.0,
      "expected_move_pct": 0.0-3.0,  # ← Changed from 0.0-10.0
      "reason": "..."
    }}

    Note on magnitude:
    - 0.5-1.0% = Minor positive news
    - 1.0-2.0% = Moderate catalyst
    - 2.0-3.0% = Major catalyst (earnings beat, M&A, etc.)
    - Only predict >3% for truly extraordinary events
    """
```

### Step 2: Add Historical Volatility Context

**Modify `maker_consensus.py` to include volatility:**

```python
def run_council(row, model_name, horizons):
    # ... existing code ...

    # Calculate historical volatility
    ticker = row['ticker']
    volatility_daily = calculate_ticker_volatility(ticker)  # Fetch from price data

    # ... run agents ...

    # After ensemble calculation:
    for h_name, h_days in horizons.items():
        # Scale magnitude by volatility
        raw_magnitude = avg_move  # From agent averaging

        # Calibrated magnitude: scale by sqrt(horizon) and volatility
        volatility_horizon = volatility_daily * np.sqrt(h_days) * 100

        # If agents predict 5% but volatility suggests 1.5%, cap at 2x volatility
        max_realistic = volatility_horizon * 2.0
        calibrated_magnitude = min(raw_magnitude, max_realistic)

        # Additional dampening: agents overestimate by ~3x historically
        calibrated_magnitude *= 0.35  # Empirical scaling factor

        output[f"exp_move_pct_{h_name}"] = calibrated_magnitude
```

### Step 3: Add Guidance in Agent Prompts

**Technical Analyst gets volatility context:**

```python
def prompt_technical(row):
    r1 = row.get('ret_1d', 0)
    vol_20d = row.get('volatility_20d', 0.015)  # Historical volatility

    context = f"""
1-Day Return: {r1:.2%}
Recent 20-day volatility: {vol_20d * 100:.2f}%

Typical 1-day move for this stock: {vol_20d * 100:.2f}%
Large move (2σ): {vol_20d * 2 * 100:.2f}%
Extreme move (3σ): {vol_20d * 3 * 100:.2f}%
"""

    return f"""
Role: Technical Analyst
...
Price Context:
{context}

When estimating expected_move_pct:
- Use the typical daily move as your baseline
- Only predict moves >2x volatility for strong momentum
- Extreme predictions (>3x volatility) require exceptional circumstances

Return valid JSON:
{{
  "signal": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": 0.0-1.0,
  "expected_move_pct": 0.0-{vol_20d * 3 * 100:.1f},  # Dynamic range based on volatility
  "reason": "..."
}}
"""
```

---

## Implementation Code

**Create `modules/volatility_calibration.py`:**

```python
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def get_historical_volatility(ticker, window=20):
    """
    Fetch historical price data and calculate realized volatility.

    Returns:
        volatility_daily: Daily volatility (std of returns)
        volatility_annualized: Annualized volatility
    """
    try:
        # Fetch last 90 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            # Default to market average if no data
            return 0.015, 0.24  # 1.5% daily, 24% annual

        # Calculate returns
        returns = df['Close'].pct_change().dropna()

        # Rolling volatility
        vol_daily = returns.rolling(window).std().iloc[-1]
        vol_annual = vol_daily * np.sqrt(252)

        return vol_daily, vol_annual

    except Exception as e:
        print(f"Warning: Could not fetch volatility for {ticker}: {e}")
        return 0.015, 0.24  # Default fallback

def calibrate_magnitude(
    predicted_move,
    volatility_daily,
    horizon_days=1,
    dampening_factor=0.35
):
    """
    Calibrate LLM-predicted magnitude using historical volatility.

    Args:
        predicted_move: Raw LLM prediction (0-10%)
        volatility_daily: Historical daily volatility (e.g., 0.015)
        horizon_days: Prediction horizon (1, 5, 21)
        dampening_factor: Empirical scaling (0.35 based on 3.38x overestimation)

    Returns:
        Calibrated magnitude in percentage
    """
    # Expected move based on volatility (1σ move)
    vol_horizon = volatility_daily * np.sqrt(horizon_days) * 100

    # LLM confidence (normalized to 0-2 range)
    confidence_mult = (predicted_move / 10.0) * 2.0

    # Realistic maximum (2σ move with confidence multiplier)
    max_realistic = vol_horizon * 2.0 * confidence_mult

    # Apply cap and dampening
    capped_move = min(predicted_move, max_realistic)
    calibrated_move = capped_move * dampening_factor

    return calibrated_move

# Example usage:
# vol_daily, vol_annual = get_historical_volatility("AMZN")
# calibrated = calibrate_magnitude(5.5, vol_daily, horizon_days=1)
# print(f"LLM: 5.5%, Calibrated: {calibrated:.2f}%")
```

---

## Updated `maker_consensus.py` Integration

**Add to imports:**
```python
from modules.volatility_calibration import get_historical_volatility, calibrate_magnitude
```

**Modify `run_council()` function:**

```python
async def run_council(row, model_name, horizons):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    ticker = row.get('ticker', 'UNKNOWN')

    # Fetch historical volatility
    vol_daily, vol_annual = get_historical_volatility(ticker)

    # Pass volatility to technical analyst prompt
    row['volatility_20d'] = vol_daily

    # Run agents (unchanged)
    results = await asyncio.gather(
        call_agent(client, model_name, prompt_bull_case(row), "bull"),
        call_agent(client, model_name, prompt_bear_case(row), "bear"),
        call_agent(client, model_name, prompt_technical(row), "technical"),
        call_agent(client, model_name, prompt_sentiment(row), "sentiment")
    )

    data = {name: res for name, res in results}

    # ... ensemble logic (unchanged) ...

    # Calculate magnitude with CALIBRATION
    moves = []
    weights_mag = []

    for agent_name, score in [("bull", bull_score), ("bear", bear_score),
                               ("technical", tech_score), ("sentiment", sent_score)]:
        if abs(score) > 0.01:
            raw_move = float(data[agent_name].get("expected_move_pct", 0))
            moves.append(raw_move)
            weights_mag.append(abs(score))

    if moves and weights_mag:
        avg_move_raw = sum(m * w for m, w in zip(moves, weights_mag)) / sum(weights_mag)
    else:
        avg_move_raw = 0.0

    # Apply calibration per horizon
    output = {}
    for h_name, h_days in horizons.items():
        output[f"prob_up_{h_name}"] = prob

        # CALIBRATED MAGNITUDE
        calibrated_mag = calibrate_magnitude(
            avg_move_raw,
            vol_daily,
            horizon_days=h_days,
            dampening_factor=0.35  # Tunable parameter
        )
        output[f"exp_move_pct_{h_name}"] = calibrated_mag

    # ... rest of output (unchanged) ...

    await client.close()
    return output
```

---

## Expected Results

### Before Calibration:
- Predicted: 5.46% average
- Actual: 1.61% average
- Overestimation: 3.38x

### After Calibration (with 0.35x dampening):
- Predicted: 5.46% * 0.35 = **1.91% average**
- Actual: 1.61% average
- Error: **19% overestimation** (down from 238%)

### Further Tuning:
If 1.91% is still too high, adjust `dampening_factor`:
- Current: 0.35
- More conservative: 0.30 → 1.64% predicted
- Match exactly: 0.295 → 1.61% predicted

---

## Validation

**After implementing, validate with:**

```python
# In step3_visualize.py or viz.py
def validate_magnitude_calibration(df):
    """
    Compare predicted vs actual magnitudes and print metrics.
    """
    df['pred_mag'] = df['exp_move_pct_1d'] / 100.0
    df['actual_mag'] = np.abs(df['ret_fwd_1d'])

    valid = df[['pred_mag', 'actual_mag']].dropna()

    mae = np.abs(valid['pred_mag'] - valid['actual_mag']).mean()
    mse = ((valid['pred_mag'] - valid['actual_mag']) ** 2).mean()
    ratio = valid['pred_mag'].mean() / valid['actual_mag'].mean()

    print("=== MAGNITUDE CALIBRATION METRICS ===")
    print(f"Predicted magnitude (mean): {valid['pred_mag'].mean() * 100:.2f}%")
    print(f"Actual magnitude (mean): {valid['actual_mag'].mean() * 100:.2f}%")
    print(f"MAE: {mae * 100:.2f}%")
    print(f"RMSE: {np.sqrt(mse) * 100:.2f}%")
    print(f"Overestimation ratio: {ratio:.2f}x")
    print()

    if ratio < 1.2:
        print("✅ GOOD: Magnitude well-calibrated (< 20% overestimation)")
    elif ratio < 1.5:
        print("⚠️  OK: Moderate overestimation (20-50%)")
    else:
        print("❌ BAD: Significant overestimation (> 50%)")
```

---

## Recommended Next Steps

1. **Phase 1 (Quick Fix)**: Update prompt ranges from 0-10% to 0-3% for 1d horizon
2. **Phase 2 (Core Fix)**: Implement volatility_calibration.py and integrate into maker_consensus.py
3. **Phase 3 (Refinement)**: Tune dampening_factor based on backtest results
4. **Phase 4 (Advanced)**: Learn scaling factors dynamically from historical performance

**Priority: High** - This directly impacts prediction accuracy and user trust in the system.

---

## References

- LSTM Predictor: Uses volatility-based magnitude estimation
- Current overestimation: 3.38x (5.46% → 1.61%)
- Target: <1.2x (within 20% of actual)
