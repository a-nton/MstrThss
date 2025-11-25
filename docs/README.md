# GDELT Stock Prediction System - Documentation

This folder contains all technical documentation for the GDELT-based stock prediction pipeline using the MAKER (Multi-Agent Knowledge Ensemble Reasoning) framework.

## Documentation Index

### System Architecture

- **[MAKER_IMPROVEMENTS.md](MAKER_IMPROVEMENTS.md)** - Evolution from 3-agent winner-takes-all to ensemble averaging
  - Problem: Probability clustering at 75-82%
  - Solution: Weighted ensemble averaging of all agent opinions
  - Refined agent prompts for reduced bias
  - Configuration options and testing recommendations

- **[BULL_VS_BEAR_UPDATE.md](BULL_VS_BEAR_UPDATE.md)** - 4-Agent adversarial system design
  - Split from 3-agent to Bull vs Bear architecture
  - Bull Case Analyst (growth-focused, BULLISH/NEUTRAL only)
  - Bear Case Analyst (risk-focused, BEARISH/NEUTRAL only)
  - Natural adversarial tension reduces groupthink
  - Example scenarios and backward compatibility

### Technical Analysis

- **[LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md](LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md)** - Comprehensive analysis of successful LSTM predictor
  - 70% confidence threshold strategy
  - 39 technical indicators (RSI, MACD, Bollinger Bands, volatility, etc.)
  - Monte Carlo dropout for uncertainty quantification
  - Feature importance rankings
  - Recommendations for enhancing MAKER Technical Analyst agent

### Magnitude Calibration

- **[MAGNITUDE_CALIBRATION.md](MAGNITUDE_CALIBRATION.md)** - **NEW** Fixing magnitude overestimation (3.38x error)
  - Problem diagnosis: LLM anchoring bias and lack of historical context
  - Solution: Volatility-based scaling + prompt range adjustment
  - Implementation guide with code examples
  - Expected results: 5.5% → 1.9% predictions (closer to 1.6% actual)

## Quick Navigation

### For Understanding the System
Start here:
1. BULL_VS_BEAR_UPDATE.md - Current agent architecture
2. MAKER_IMPROVEMENTS.md - Why ensemble averaging works
3. LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md - Technical analysis best practices

### For Implementation
Key sections:
- **Agent Design**: BULL_VS_BEAR_UPDATE.md → "New 4-Agent Architecture"
- **Probability Calculation**: MAKER_IMPROVEMENTS.md → "Ensemble Consensus"
- **Technical Indicators**: LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md → "Technical Indicators (39 Features)"
- **Confidence Thresholds**: LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md → "The 70% Rule"

### For Configuration
- **Agent Weights**: MAKER_IMPROVEMENTS.md → "Configuration Options"
- **Thresholds**: BULL_VS_BEAR_UPDATE.md → "Configuration"
- **Technical Features**: LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md → "Recommendations for GDELT Technical Analyst Agent"

## System Evolution Timeline

1. **Original System**: 3-agent MAKER (Fundamentalist, Technician, Psychologist)
   - Winner-takes-all consensus
   - Problem: Clustering at 75-82% probability

2. **Ensemble Averaging** (MAKER_IMPROVEMENTS.md)
   - Weighted average of all agent opinions
   - Dissenting voices now reduce extreme probabilities
   - Result: Wider probability distribution

3. **Bull vs Bear Split** (BULL_VS_BEAR_UPDATE.md)
   - 4-agent adversarial system
   - Forced opposing perspectives
   - Bull analyst seeks positives, Bear analyst seeks risks
   - Natural tension creates balanced predictions

4. **Technical Analysis Enhancement** (LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md)
   - Insights from 73% accurate LSTM system
   - Comprehensive technical indicator suite
   - Confidence-based trade filtering
   - Volatility-adjusted predictions

## Key Metrics & Thresholds

| Metric | Value | Source |
|--------|-------|--------|
| LSTM Accuracy | 73% | LSTM_TECHNICAL_ANALYSIS_INSIGHTS.md |
| Confidence Threshold | 70% | LSTM predictor |
| MC Dropout Samples | 25 | LSTM predictor |
| Agent Count | 4 | BULL_VS_BEAR_UPDATE.md |
| Ensemble Method | Weighted Average | MAKER_IMPROVEMENTS.md |
| Prediction Horizons | 1d, 1w, 1m | config.py |

## Agent Roles Summary

| Agent | Focus | Allowed Signals | Weight |
|-------|-------|----------------|--------|
| Bull Case | Growth, catalysts, opportunities | BULLISH, NEUTRAL | 1.0 |
| Bear Case | Risks, headwinds, threats | BEARISH, NEUTRAL | 1.0 |
| Technical | Price action, momentum, volatility | BULLISH, BEARISH, NEUTRAL | 1.0 |
| Psychologist | Sentiment, contrarian signals | BULLISH, BEARISH, NEUTRAL | 1.0 |

**Ensemble Formula:**
```
ensemble_score = (bull_score + bear_score + tech_score + sent_score) / 4
probability = 50% + (50% × ensemble_score)
```

## Technical Indicators Priority

### High Priority (Implement First)
- RSI (14-period)
- MACD & Signal Line
- Moving Averages (MA10, MA20)
- Bollinger Bands
- 20-day Volatility

### Medium Priority
- On-Balance Volume (OBV)
- 20-day Momentum
- Z-Score
- 5-day Volatility

### Advanced
- Sentiment data
- Insider trading signals
- Skewness & tail risk
- Multi-horizon adaptive strategy

## Related Files in Codebase

- `modules/llm_configs/maker_consensus.py` - Agent implementation
- `modules/viz.py` - Dashboard with agent tooltips
- `config.py` - System configuration
- `launcher.py` - Pipeline orchestration
- `run_pipeline.sh` - Main launcher script

## References

- LSTM Predictor: `/Users/metanoid/Desktop/Master Thesis/Deprecated:Backup/LSTM_AI_Stock_Predictor-main/`
- GDELT API: https://api.gdeltproject.org/
- OpenAI GPT-4o-mini: Model used for agent reasoning

---

**Last Updated:** November 25, 2025
**System Version:** Bull vs Bear (4-agent) with Ensemble Averaging
