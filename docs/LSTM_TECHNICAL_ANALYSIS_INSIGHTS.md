# LSTM Stock Predictor: Technical Analysis Insights

## Executive Summary

Analysis of a successful LSTM-based stock predictor that achieved **70% directional accuracy** using Conv1D + LSTM architecture with Monte Carlo dropout for uncertainty estimation.

**Key Success Factors:**
- 70% confidence threshold for trade execution
- Comprehensive technical indicator suite (39 features)
- Monte Carlo dropout for uncertainty quantification
- Walk-forward validation to prevent data leakage
- Time-weighted training (recent data emphasized)

---

## Architecture Overview

### Model Structure

```
Conv1D(32, kernel=3) → BatchNorm → MC Dropout(0.3)
    ↓
LSTM(64, return_sequences=False) → MC Dropout(0.3)
    ↓
Dense(32, relu) → Dense(4, linear)
```

**Key Components:**
- **Conv1D**: Captures local temporal patterns (3-day windows)
- **LSTM**: Learns long-term dependencies (90-day lookback)
- **MC Dropout (0.3)**: Enabled during inference for uncertainty estimation
- **Multi-horizon output**: Predicts 1d, 1w, 1m, 6m returns simultaneously

### Monte Carlo Dropout for Uncertainty

```python
CONFIDENCE_THRESHOLD = 0.70
MC_DROPOUT_SAMPLES = 25

def mc_dropout_predict(model, X, n_samples=25):
    preds = [model(X, training=True) for _ in range(n_samples)]
    return preds.mean(), preds.std(), preds

# Only execute trades when P(up) > 70%
if predicted_prob > CONFIDENCE_THRESHOLD and confidence > threshold:
    execute_trade()
```

**Why This Works:**
- Dropout randomness creates ensemble of 25 sub-models
- Standard deviation measures prediction uncertainty
- High-certainty predictions are more reliable
- Filters out ambiguous signals

---

## Technical Indicators (39 Features)

### Price-Based Features (7)
| Feature | Description | Purpose |
|---------|-------------|---------|
| `close` | Current close price | Base price level |
| `YesterdayClose` | Previous day close | Price context |
| `YesterdayOpenLogR` | Log return: open/prev_close | Gap analysis |
| `YesterdayHighLogR` | Log return: high/prev_close | Intraday range |
| `YesterdayLowLogR` | Log return: low/prev_close | Support/resistance |
| `YesterdayVolumeLogR` | Log return: volume change | Volume momentum |
| `YesterdayCloseLogR` | Log return: close/prev_close | Daily return |

### Moving Averages (5)
| Feature | Window | Purpose |
|---------|--------|---------|
| `MA10` | 10 days | Short-term trend |
| `MA20` | 20 days | Medium-term trend |
| `MA30` | 30 days | Long-term trend |
| `EMA10` | 10 days (exponential) | Responsive short-term |
| `EMA30` | 30 days (exponential) | Responsive long-term |

**Trend Signals:**
- Price > MA: Uptrend
- Golden cross (MA10 > MA20): Bullish
- Death cross (MA10 < MA20): Bearish

### Momentum Indicators (6)
| Feature | Calculation | Range | Signal |
|---------|-------------|-------|--------|
| `RSI` | 14-period Relative Strength Index | 0-100 | <30: Oversold, >70: Overbought |
| `MACD` | EMA12 - EMA26 | Unbounded | Positive: Bullish momentum |
| `MACD_Signal` | 9-period EMA of MACD | Unbounded | MACD > Signal: Buy signal |
| `momentum_5d` | 5-day rate of change | Unbounded | Positive: Short-term strength |
| `momentum_20d` | 20-day rate of change | Unbounded | Positive: Medium-term strength |
| `overnight_gap` | (Open - Prev Close) / Prev Close | % | Measures gap up/down |

### Volatility Indicators (7)
| Feature | Window | Purpose |
|---------|--------|---------|
| `Volatility_10` | 10-day std of returns | Short-term risk |
| `Volatility_20` | 20-day std of returns | Medium-term risk |
| `Volatility_30` | 30-day std of returns | Long-term risk |
| `volatility_5d` | 5-day rolling volatility | Ultra short-term risk |
| `volatility_20d` | 20-day rolling volatility | Standard risk measure |
| `BollingerUpper` | MA20 + 2*std | Resistance level |
| `BollingerLower` | MA20 - 2*std | Support level |

**Volatility Signals:**
- High volatility: Increased uncertainty, wider stops
- Price > BollingerUpper: Overbought
- Price < BollingerLower: Oversold
- Bollinger squeeze: Breakout imminent

### Volume & Distribution (4)
| Feature | Description | Signal |
|---------|-------------|--------|
| `OBV` | On-Balance Volume (cumulative) | Confirms price trends |
| `abnormal_vol` | Volume deviation from average | Unusual activity |
| `skew_5d` | 5-day return distribution skew | Tail risk measure |
| `intraday_range` | (High - Low) / Close | Daily volatility |

### Statistical (1)
| Feature | Calculation | Purpose |
|---------|-------------|---------|
| `ZScore` | (Price - MA20) / std20 | Standardized deviation from mean |

**Signal:** ZScore > 2: Extremely overbought, ZScore < -2: Extremely oversold

### Alternative Data (5)
| Feature | Source | Signal |
|---------|--------|--------|
| `sentiment` | News sentiment score | Positive: Bullish, Negative: Bearish |
| `sentiment_change` | Change in sentiment | Momentum of sentiment shift |
| `num_articles` | Article count | Media attention level |
| `insider_shares` | Insider trading shares | Insider confidence |
| `insider_amount` | Insider trading dollar value | Scale of insider conviction |
| `insider_buy_flag` | Binary: Buy=1, Sell=0 | Directional insider signal |

### Temporal Features (3)
| Feature | Range | Purpose |
|---------|-------|---------|
| `DayOfWeek` | 0-6 | Capture weekly seasonality |
| `DayOfMonth` | 1-31 | Capture monthly patterns |
| `MonthNumber` | 1-12 | Capture annual seasonality |

---

## Feature Importance (Top 15)

Based on permutation importance analysis from the LSTM model:

1. **RSI** - Most predictive single indicator
2. **MACD / MACD_Signal** - Strong momentum signals
3. **Volatility_20** - Risk regime identification
4. **sentiment / sentiment_change** - News-driven moves
5. **MA10 / MA20** - Trend confirmation
6. **insider_buy_flag** - Information asymmetry
7. **momentum_20d** - Medium-term strength
8. **BollingerUpper/Lower** - Mean reversion signals
9. **OBV** - Volume confirmation
10. **ZScore** - Statistical extremes
11. **overnight_gap** - Pre-market sentiment
12. **volatility_5d** - Short-term risk spikes
13. **skew_5d** - Tail risk
14. **abnormal_vol** - Unusual activity detection
15. **EMA10** - Responsive trend following

---

## Confidence Threshold Strategy

### The 70% Rule

```python
CONFIDENCE_THRESHOLD = 0.70
CONFIDENCE_Z = 1.5

# Adjust prediction for uncertainty
adjusted_pred = mean_prediction - (CONFIDENCE_Z * std_prediction)

# Only trade if:
# 1. Probability of up move > 70%
# 2. Uncertainty is low (std < threshold)
if prob_up > 0.70 and std_prediction < max_uncertainty:
    execute_buy()
```

**Why 70% Works:**
- **Below 70%**: Signal too weak, skip trade (avoid false positives)
- **At 70%**: Sufficient edge for profitable trading after costs
- **Above 70%**: High-conviction trades (concentrate capital)

**Results from backtesting:**
- Buy success rate: **~73%** (close to predicted 70%)
- This shows **good calibration** - model knows when it's right
- Outperformed SPY buy-and-hold significantly

### Uncertainty Adjustment Methods

The system offers two uncertainty handling modes:

1. **subtract_std**: `prediction - (z_score * std)`
   - Conservative: Reduces predicted return by uncertainty
   - Better for risk-averse strategies

2. **confidence_filter**: Only trade if `std < threshold`
   - Filters out uncertain predictions entirely
   - Better for high-conviction strategies

---

## Walk-Forward Validation

**Critical for preventing data leakage:**

```python
TRAIN_VAL_FRAC = 0.8  # 80% train+val, 20% test

# Temporal split (NOT random split)
train_cutoff = min_date + (max_date - min_date) * 0.64  # 80% of 80%
val_cutoff = min_date + (max_date - min_date) * 0.80

# Fit scaler ONLY on training data
scaler.fit(train_data)

# Never look ahead - test data is strictly future dates
```

**Why This Matters:**
- Stock markets are non-stationary (patterns change over time)
- Random splits leak future information into training
- Temporal splits simulate real trading (only use past data)

---

## Adaptive Multi-Horizon Strategy

The model predicts 4 horizons simultaneously:

```python
horizons = ["1d", "1w", "1m", "6m"]
horizon_days = [1, 5, 21, 126]
```

**Strategy Logic:**

1. Each day, generate predictions for all tickers across all horizons
2. Score each prediction: `score = prob * confidence / days`
3. Select the highest-scoring prediction
4. Hold for the predicted horizon duration
5. Exit and re-evaluate

**Benefits:**
- Adapts to market regime (trade short-term in volatile markets, long-term in trends)
- Maximizes risk-adjusted returns
- Avoids forced short-term thinking

---

## Training Enhancements

### Time-Weighted Loss

```python
# Give exponentially more weight to recent data
date_ago = (today - training_date).days
weight = exp(-decay_factor * date_ago)

# decay_factor = 0.002 means ~50% weight at 1.9 years ago
```

**Rationale:**
- Market dynamics change over time
- Recent patterns more relevant than old patterns
- Prevents model from overfitting to outdated regimes

### Early Stopping

```python
early_stop = EarlyStopping(
    patience=25,
    monitor="val_loss",
    restore_best_weights=True
)
```

**Prevents overfitting:**
- Stops training when validation loss stops improving
- Restores weights from best epoch (not final epoch)

---

## Recommendations for GDELT Technical Analyst Agent

### 1. Adopt Core Technical Indicators

**High Priority (Implement First):**
```python
technical_features = [
    "RSI_14",           # Momentum: <30 oversold, >70 overbought
    "MACD",             # Trend: Positive = bullish
    "MACD_Signal",      # Crossover: MACD > Signal = buy
    "Volatility_20",    # Risk regime: High vol = uncertain
    "MA10", "MA20",     # Trend: Price > MA = uptrend
    "BollingerBands",   # Mean reversion: Outside bands = extreme
]
```

**Medium Priority:**
```python
additional_features = [
    "OBV",              # Volume confirmation
    "momentum_20d",     # Medium-term strength
    "ZScore",           # Statistical extremes
    "volatility_5d",    # Short-term risk spikes
]
```

### 2. Implement Confidence-Based Signals

**Current System:**
```python
# Agent returns: BULLISH / BEARISH / NEUTRAL with confidence 0-1
```

**Recommended Enhancement:**
```python
def get_technical_signal(price_data):
    # Calculate all indicators
    indicators = calculate_indicators(price_data)

    # Multi-indicator consensus
    bullish_signals = 0
    bearish_signals = 0
    total_weight = 0

    # RSI
    if indicators['RSI'] < 30:
        bullish_signals += 0.8  # Strong oversold
    elif indicators['RSI'] > 70:
        bearish_signals += 0.8  # Strong overbought
    total_weight += 0.8

    # MACD Crossover
    if indicators['MACD'] > indicators['MACD_Signal']:
        bullish_signals += 0.7
    else:
        bearish_signals += 0.7
    total_weight += 0.7

    # Trend (MA crossover)
    if indicators['MA10'] > indicators['MA20']:
        bullish_signals += 0.6
    else:
        bearish_signals += 0.6
    total_weight += 0.6

    # Bollinger Bands
    if price < indicators['BollingerLower']:
        bullish_signals += 0.5  # Oversold
    elif price > indicators['BollingerUpper']:
        bearish_signals += 0.5  # Overbought
    total_weight += 0.5

    # Calculate net score
    net_score = (bullish_signals - bearish_signals) / total_weight

    # Apply 70% threshold logic
    if net_score > 0.4:  # Equivalent to 70% confidence
        return "BULLISH", abs(net_score)
    elif net_score < -0.4:
        return "BEARISH", abs(net_score)
    else:
        return "NEUTRAL", abs(net_score)
```

### 3. Volatility-Adjusted Confidence

**Key Insight:** High volatility = Lower confidence

```python
def adjust_for_volatility(base_confidence, volatility_20d):
    # Normalize volatility (typical range: 0.01 - 0.05)
    vol_normalized = min(volatility_20d / 0.03, 2.0)

    # Reduce confidence in high-vol environments
    adjusted_confidence = base_confidence * (1 / (1 + vol_normalized))

    return adjusted_confidence
```

**Example:**
- Base technical signal: BULLISH 0.8
- If volatility is 2x normal → Adjusted: BULLISH 0.53
- Reflects uncertainty in volatile markets

### 4. Enhanced Agent Prompt

**Updated Technical Analyst Prompt:**

```python
def prompt_technical_analyst(row):
    return f"""
Role: Technical Analyst (Price Action & Momentum Specialist)

Task: Analyze price action and technical indicators for {row['name']} ({row['ticker']}).

Available Technical Data:
- Recent price action (trend, support/resistance)
- RSI (14-day): [Calculate or fetch]
- MACD vs Signal Line: [Calculate or fetch]
- Price vs MA10/MA20: [Trend context]
- Bollinger Bands position: [Mean reversion signals]
- 20-day volatility: [Risk regime]

Guidelines:
1. **Momentum**: RSI <30 = oversold (bullish), >70 = overbought (bearish)
2. **Trend**: MACD > Signal = bullish momentum
3. **Mean Reversion**: Price > BollingerUpper = overbought, < BollingerLower = oversold
4. **Volatility**: High vol (>2x average) = reduce confidence
5. **Confluence**: Only give high confidence when 3+ indicators align

Confidence Calibration:
- 0.8-1.0: 4+ indicators strongly aligned, low volatility
- 0.6-0.8: 2-3 indicators aligned, normal volatility
- 0.4-0.6: Mixed signals or high volatility
- 0.2-0.4: Weak signals, very uncertain
- 0.0-0.2: No clear technical setup

Return format:
Signal: [BULLISH/BEARISH/NEUTRAL]
Confidence: [0.0-1.0]
Reasoning: [Which indicators are aligned, confluence strength, volatility context]
Magnitude: [Expected move % based on historical volatility]
"""
```

### 5. Integration with Price Data

**Current Challenge:** GDELT system may not have real-time price data for technical calculations.

**Solutions:**

**Option A: Fetch at LLM Time**
```python
import yfinance as yf

def get_technical_context(ticker, lookback_days=90):
    df = yf.download(ticker, period=f"{lookback_days}d")

    # Calculate indicators
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['BB_upper'], df['BB_lower'] = calculate_bollinger(df['Close'], 20)
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std()

    return df.iloc[-1]  # Return latest values
```

**Option B: Pre-compute and Store**
```python
# Add to data warehouse during ingestion
# Store technical indicators alongside price data
# LLM reads from warehouse at prediction time
```

**Option C: LLM Approximation (Fallback)**
```python
# If no price data available, ask LLM to estimate based on context
# Less accurate but better than ignoring technicals entirely
```

---

## Key Takeaways

### What Made the LSTM Predictor Successful

1. **70% Confidence Threshold**: Only trade high-conviction signals
2. **Uncertainty Quantification**: MC Dropout provides calibrated confidence
3. **Comprehensive Features**: 39 technical + alternative data features
4. **Proper Validation**: Walk-forward temporal splits prevent leakage
5. **Multi-Horizon Flexibility**: Adapts to market regime
6. **Time-Weighted Training**: Recent data matters more

### How to Apply to MAKER Framework

1. **Enhance Technical Analyst Agent**:
   - Add RSI, MACD, Bollinger Bands, Volatility calculations
   - Implement multi-indicator confluence scoring
   - Apply volatility-adjusted confidence

2. **Adopt 70% Threshold**:
   - Current ensemble already outputs probability
   - Add filter: Only make predictions when ensemble confidence > 70%
   - Track calibration: Do 70% predictions win ~70% of the time?

3. **Separate Uncertainty from Signal**:
   - Bull/Bear agents: Signal direction
   - Technical agent: Add uncertainty adjustment based on volatility
   - Psychologist: Monitor sentiment extremes

4. **Feature Parity**:
   - Add volatility calculations to price data
   - Fetch technical indicators (RSI, MACD) at prediction time
   - Consider insider trading data if available

---

## Implementation Checklist

- [ ] Add `yfinance` or similar to `requirements.txt`
- [ ] Create `modules/technical_indicators.py` with RSI, MACD, Bollinger, Volatility functions
- [ ] Update Technical Analyst prompt to use calculated indicators
- [ ] Add volatility-based confidence adjustment
- [ ] Implement 70% confidence threshold filter in ensemble
- [ ] Add calibration metrics to dashboard (predicted 70% should win ~70%)
- [ ] Backtest on historical data and compare pre/post technical enhancement
- [ ] Consider adding insider trading data source (SEC Form 4)

---

## References

**LSTM Predictor Code Location:**
`/Users/metanoid/Desktop/Master Thesis/Deprecated:Backup/LSTM_AI_Stock_Predictor-main/`

**Key Files:**
- `forecast.ipynb`: Model training and feature engineering
- `forecasting_backtest_Predictor_v2.py`: Backtesting with 70% threshold
- `TrainingData/processor.py`: Technical indicator calculations
- `README.md`: System overview and architecture

**Model Performance:**
- Directional accuracy: ~73% on test set
- Confidence threshold: 70%
- Outperformed SPY buy-and-hold baseline
- MC Dropout samples: 25
- Window size: 90 days
