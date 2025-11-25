# Bull vs Bear: 4-Agent MAKER System

## Motivation
The original 3-agent system had all agents analyzing the same headlines without opposing perspectives. This led to:
- Groupthink and consensus bias
- Insufficient exploration of downside risks
- Probabilities clustering around optimistic values (75-82%)

## Solution: Adversarial Agent Design

### New 4-Agent Architecture

1. **Bull Case Analyst** (Growth & Opportunity Focused)
   - Actively seeks positive catalysts
   - Only returns: BULLISH or NEUTRAL
   - Looks for: revenue growth, strategic wins, innovation, market expansion

2. **Bear Case Analyst** (Risk & Headwind Focused)
   - Actively seeks risks and negative catalysts
   - Only returns: BEARISH or NEUTRAL
   - Looks for: revenue misses, regulatory risks, competitive threats, cost pressures

3. **Technical Analyst** (Price Action Specialist)
   - Analyzes momentum and mean reversion
   - Returns: BULLISH, BEARISH, or NEUTRAL
   - Unchanged from before

4. **Psychologist** (Sentiment & Behavioral Analyst)
   - Weighs momentum vs contrarian signals
   - Returns: BULLISH, BEARISH, or NEUTRAL
   - Unchanged from before

## Key Innovation: Forced Adversarial Perspectives

**Before:**
- Fundamentalist analyzes headlines → sees mostly positive news → votes BULLISH
- No agent systematically searches for downside

**After:**
- Bull analyst **must** find the best bullish case (or admit there is none → NEUTRAL)
- Bear analyst **must** find the best bearish case (or admit there is none → NEUTRAL)
- Natural tension creates balanced analysis

## Ensemble Calculation

All 4 agents contribute equally to the final probability:

```python
# Convert to directional scores (-1 to +1)
bull_score = +0.8 if BULLISH(conf=0.8), 0 if NEUTRAL
bear_score = -0.7 if BEARISH(conf=0.7), 0 if NEUTRAL
tech_score = +0.5 if BULLISH(conf=0.5)
sent_score = 0 if NEUTRAL

# Equal weighted average
ensemble_score = (0.8 + (-0.7) + 0.5 + 0) / 4 = 0.15

# Map to probability
prob = 50% + (50% × 0.15) = 57.5%
```

**Result:** Bull and Bear opinions partially cancel out, creating more balanced predictions.

## Example Scenarios

### Scenario 1: Mixed News (Investment Announcement + Lawsuit)

| Agent | Signal | Confidence | Score | Reasoning |
|-------|--------|-----------|-------|-----------|
| Bull  | BULLISH | 0.7 | +0.7 | "$5B investment shows strong growth commitment" |
| Bear  | BEARISH | 0.6 | -0.6 | "Major lawsuit creates regulatory uncertainty" |
| Tech  | NEUTRAL | - | 0.0 | "Price flat, no clear momentum" |
| Sent  | NEUTRAL | - | 0.0 | "Conflicting headlines, no strong sentiment" |

**Ensemble:** (0.7 - 0.6 + 0 + 0) / 4 = **0.025** → **51.25% probability**

**OLD SYSTEM:** Fundamentalist might vote BULLISH (ignoring lawsuit), leading to ~65% probability

---

### Scenario 2: Uniformly Positive News

| Agent | Signal | Confidence | Score | Reasoning |
|-------|--------|-----------|-------|-----------|
| Bull  | BULLISH | 0.9 | +0.9 | "Earnings beat, guidance raised, new partnerships" |
| Bear  | NEUTRAL | 0.1 | 0.0 | "No significant risks identified" |
| Tech  | BULLISH | 0.7 | +0.7 | "Strong upward momentum" |
| Sent  | BULLISH | 0.6 | +0.6 | "Positive sentiment across all headlines" |

**Ensemble:** (0.9 + 0 + 0.7 + 0.6) / 4 = **0.55** → **77.5% probability**

This is reasonable - strong bull case, weak bear case, supporting technicals and sentiment.

---

### Scenario 3: Uniformly Negative News

| Agent | Signal | Confidence | Score | Reasoning |
|-------|--------|-----------|-------|-----------|
| Bull  | NEUTRAL | 0.1 | 0.0 | "No positive catalysts found" |
| Bear  | BEARISH | 0.8 | -0.8 | "Revenue miss, guidance cut, management exodus" |
| Tech  | BEARISH | 0.6 | -0.6 | "Breakdown below support levels" |
| Sent  | BEARISH | 0.5 | -0.5 | "Panic selling, negative sentiment" |

**Ensemble:** (0 - 0.8 - 0.6 - 0.5) / 4 = **-0.475** → **26.25% probability**

Properly bearish prediction.

## Benefits

1. **Reduced Clustering:** Bull/Bear tension creates natural variance
2. **Better Calibration:** Forced to consider both sides reduces bias
3. **Transparency:** Can see bull case vs bear case in tooltips
4. **Genuine Disagreement:** When bull and bear both have high confidence on opposite sides, ensemble reflects uncertainty

## Backward Compatibility

The visualization automatically detects 4-agent vs 3-agent results:
- 4-agent runs: Shows Bull, Bear, Technician, Psychologist
- 3-agent runs: Shows Fundamentalist, Technician, Psychologist

## Testing Recommendations

1. Run on same historical period as before
2. Compare probability distributions (should be wider, more centered around 50%)
3. Check if extreme probabilities (>80% or <20%) now require stronger evidence
4. Verify bear case is finding genuine risks (not just being contrarian for sake of it)

## Configuration

Agent weights can be tuned in the code:

```python
weights = {
    "bull": 1.0,      # Increase to favor growth stories
    "bear": 1.0,      # Increase to favor defensive/risk-aware analysis
    "technical": 1.0, # Increase to favor price momentum
    "sentiment": 1.0  # Increase to favor crowd psychology
}
```

For a more conservative system, increase bear weight to 1.5.
