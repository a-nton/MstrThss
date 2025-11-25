# MAKER Framework Improvements

## Problem Identified
Probabilities were clustering around 75-82%, showing insufficient variance and potential bullish bias.

## Root Causes
1. **Winner-Takes-All Consensus**: 2/3 majority completely ignored dissenting agent
2. **Optimism Bias**: LLMs (especially GPT-4o-mini) tend toward positive interpretations
3. **Headline Selection Bias**: GDELT news about big tech naturally skews positive during growth periods

## Solutions Implemented

### 1. Refined Agent Prompts

**Fundamentalist (Value Investor)**
- Added skepticism: "Be SKEPTICAL. Most news is noise."
- Strict focus criteria: Only actual earnings, confirmed M&A, regulatory impact
- Explicit ignore list: Speculation, PR fluff, analyst upgrades, vague partnerships

**Psychologist (Contrarian Sentiment)**
- Added contrarian thinking: "Too positive = bearish signal"
- Considers overcrowded trades and sentiment extremes
- Default to NEUTRAL unless clear directional signal
- Balances short-term momentum vs reversal potential

### 2. Ensemble Consensus (Weighted Average)

**OLD SYSTEM:**
```
2 agents: BULLISH (conf: 0.8)
1 agent:  NEUTRAL
→ Result: 82% (ignores neutral agent completely!)
```

**NEW SYSTEM:**
```
Agent Scores:
- Fundamentalist: BULLISH 0.8 → score = +0.8
- Technician:     NEUTRAL     → score = 0.0
- Psychologist:   BULLISH 0.8 → score = +0.8

Ensemble Score = (0.8 + 0.0 + 0.8) / 3 = 0.533
Probability = 50% + (50% × 0.533) = 76.7%
```

**Key Improvement:** Dissenting or neutral opinions now reduce the final probability proportionally.

### 3. New Probability Mapping

**Formula:**
```python
ensemble_score = (fund_score + tech_score + sent_score) / 3
probability = 0.5 + (0.5 * ensemble_score)
```

**Score Range:**
- Agent scores: -1 (bearish) to +1 (bullish)
- Ensemble score: -1 to +1 (average of agents)
- Probability: 0% to 100% (mapped linearly)

**Examples:**

| Scenario | Fund | Tech | Sent | Score | Prob | Old Prob |
|----------|------|------|------|-------|------|----------|
| Unanimous bullish | B(0.8) | B(0.8) | B(0.8) | 0.80 | 90% | 82% |
| 2 bull, 1 neutral | B(0.8) | N | B(0.8) | 0.53 | 77% | 82% |
| 2 bull, 1 bear | B(0.8) | B(0.8) | Be(0.6) | 0.33 | 67% | 82% |
| Split decision | B(0.7) | Be(0.7) | N | 0.00 | 50% | 50% |
| Weak consensus | B(0.3) | B(0.4) | N | 0.23 | 62% | 71% |

### 4. Weighted Magnitude Calculation

Magnitude (expected move %) is now weighted by each agent's confidence:

```python
# Only include non-neutral agents
# Weight by abs(score) to emphasize stronger opinions

moves = [fund_mag, tech_mag, sent_mag]
weights = [abs(fund_score), abs(tech_score), abs(sent_score)]

avg_move = sum(m * w for m, w in zip(moves, weights)) / sum(weights)
```

## Expected Results

### Probability Distribution
- **Before**: Clustered 75-82% (median: 80%)
- **After**: Wider spread 30-90% (median: ~55-65%)

### Sentiment Balance
- More NEUTRAL predictions when news is ambiguous
- Dissenting opinions reduce extreme probabilities
- Skeptical prompts reduce false positives

### Accuracy Improvements
- Fewer overconfident predictions
- Better calibration (predicted 70% should win ~70% of the time)
- Reduced optimism bias

## Configuration Options

**Agent Weights** (can be tuned):
```python
weights = {
    "fundamental": 1.0,  # Increase to 1.5 to favor fundamental analysis
    "technical": 1.0,    # Increase to 1.5 to favor price momentum
    "sentiment": 1.0     # Increase to 1.5 to favor sentiment signals
}
```

**Ensemble Thresholds**:
- Bullish: `ensemble_score > 0.15`
- Bearish: `ensemble_score < -0.15`
- Neutral: `-0.15 ≤ ensemble_score ≤ 0.15`

## Testing Recommendations

1. **Run on historical data** (Sept 2025 period)
2. **Compare probability distributions** before/after
3. **Check calibration**: Group predictions by probability buckets (50-60%, 60-70%, etc.) and measure actual accuracy
4. **Monitor for opposite bias**: If probabilities now cluster too low, adjust agent prompts
