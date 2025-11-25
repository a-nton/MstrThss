"""
Technical Indicators Module

Provides basic technical analysis indicators inspired by the LSTM predictor.
Focuses on the most predictive and easiest to implement indicators:
1. RSI (Relative Strength Index)
2. Moving Averages (MA10, MA20)
3. MACD (Moving Average Convergence Divergence)
4. Volatility (20-day)

These are fetched via yfinance and calculated for use in agent prompts.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Cache to avoid repeated API calls
_indicator_cache = {}


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index.

    Args:
        prices: Series of closing prices
        period: RSI period (default: 14)

    Returns:
        RSI value (0-100)
    """
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices: Series of closing prices
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        tuple: (macd, signal_line, histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    return (
        macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0.0,
        signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0.0,
        histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0.0
    )


def get_technical_indicators(ticker, use_cache=True):
    """
    Fetch price data and calculate key technical indicators.

    Args:
        ticker: Stock ticker symbol (e.g., 'AMZN')
        use_cache: Use cached indicators if available (default: True)

    Returns:
        dict with indicators:
            - rsi: RSI value (0-100)
            - ma10: 10-day moving average
            - ma20: 20-day moving average
            - macd: MACD value
            - macd_signal: MACD signal line
            - macd_histogram: MACD histogram
            - volatility_20d: 20-day volatility (%)
            - current_price: Latest close price
            - price_vs_ma10: % above/below MA10
            - price_vs_ma20: % above/below MA20
    """
    # Check cache
    if use_cache and ticker in _indicator_cache:
        cache_time, indicators = _indicator_cache[ticker]
        # Cache valid for 1 hour
        if (datetime.now() - cache_time).seconds < 3600:
            return indicators

    try:
        # Fetch 90 days of data (sufficient for all indicators)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if len(df) < 30:
            return _default_indicators()

        # Handle MultiIndex columns (yfinance returns multi-level for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close_prices = df['Close']

        # Calculate indicators
        rsi = calculate_rsi(close_prices, period=14)
        macd, macd_signal, macd_hist = calculate_macd(close_prices)

        ma10 = close_prices.rolling(10).mean().iloc[-1]
        ma20 = close_prices.rolling(20).mean().iloc[-1]

        # Volatility (20-day std of returns)
        returns = close_prices.pct_change()
        vol_20d = returns.rolling(20).std().iloc[-1] * 100  # Convert to %

        current_price = close_prices.iloc[-1]

        # Price vs MAs (% difference)
        price_vs_ma10 = ((current_price - ma10) / ma10) * 100 if not pd.isna(ma10) else 0.0
        price_vs_ma20 = ((current_price - ma20) / ma20) * 100 if not pd.isna(ma20) else 0.0

        indicators = {
            'rsi': float(rsi),
            'ma10': float(ma10) if not pd.isna(ma10) else current_price,
            'ma20': float(ma20) if not pd.isna(ma20) else current_price,
            'macd': float(macd),
            'macd_signal': float(macd_signal),
            'macd_histogram': float(macd_hist),
            'volatility_20d': float(vol_20d) if not pd.isna(vol_20d) else 1.5,
            'current_price': float(current_price),
            'price_vs_ma10': float(price_vs_ma10),
            'price_vs_ma20': float(price_vs_ma20)
        }

        # Cache result
        _indicator_cache[ticker] = (datetime.now(), indicators)

        return indicators

    except Exception as e:
        print(f"⚠️  Error fetching indicators for {ticker}: {e}")
        return _default_indicators()


def _default_indicators():
    """
    Return default indicator values when data is unavailable.
    """
    return {
        'rsi': 50.0,  # Neutral
        'ma10': 0.0,
        'ma20': 0.0,
        'macd': 0.0,
        'macd_signal': 0.0,
        'macd_histogram': 0.0,
        'volatility_20d': 1.5,  # 1.5% daily
        'current_price': 0.0,
        'price_vs_ma10': 0.0,
        'price_vs_ma20': 0.0
    }


def interpret_rsi(rsi):
    """
    Interpret RSI value into trading signal.

    Returns: (signal, description)
    """
    if rsi < 30:
        return "OVERSOLD", f"RSI {rsi:.1f} < 30: Strong oversold signal"
    elif rsi < 40:
        return "WEAK", f"RSI {rsi:.1f}: Approaching oversold"
    elif rsi > 70:
        return "OVERBOUGHT", f"RSI {rsi:.1f} > 70: Strong overbought signal"
    elif rsi > 60:
        return "STRONG", f"RSI {rsi:.1f}: Approaching overbought"
    else:
        return "NEUTRAL", f"RSI {rsi:.1f}: Neutral momentum"


def interpret_macd(macd, macd_signal, macd_hist):
    """
    Interpret MACD signals.

    Returns: (signal, description)
    """
    if macd > macd_signal and macd_hist > 0:
        return "BULLISH", f"MACD ({macd:.2f}) > Signal ({macd_signal:.2f}): Bullish crossover"
    elif macd < macd_signal and macd_hist < 0:
        return "BEARISH", f"MACD ({macd:.2f}) < Signal ({macd_signal:.2f}): Bearish crossover"
    else:
        return "NEUTRAL", f"MACD ({macd:.2f}) vs Signal ({macd_signal:.2f}): No clear signal"


def interpret_ma_trend(price_vs_ma10, price_vs_ma20):
    """
    Interpret trend based on moving averages.

    Returns: (signal, description)
    """
    if price_vs_ma10 > 2 and price_vs_ma20 > 2:
        return "UPTREND", f"Price {price_vs_ma10:.1f}% above MA10, {price_vs_ma20:.1f}% above MA20: Strong uptrend"
    elif price_vs_ma10 < -2 and price_vs_ma20 < -2:
        return "DOWNTREND", f"Price {price_vs_ma10:.1f}% below MA10, {price_vs_ma20:.1f}% below MA20: Strong downtrend"
    elif price_vs_ma10 > 0 and price_vs_ma20 > 0:
        return "UPTREND", f"Price above both MAs: Uptrend"
    elif price_vs_ma10 < 0 and price_vs_ma20 < 0:
        return "DOWNTREND", f"Price below both MAs: Downtrend"
    else:
        return "CHOPPY", f"Price between MAs: Choppy/consolidating"


def get_technical_summary(ticker):
    """
    Get a complete technical analysis summary for the agent.

    Returns:
        String formatted for LLM prompt
    """
    indicators = get_technical_indicators(ticker)

    rsi_signal, rsi_desc = interpret_rsi(indicators['rsi'])
    macd_signal, macd_desc = interpret_macd(
        indicators['macd'],
        indicators['macd_signal'],
        indicators['macd_histogram']
    )
    trend_signal, trend_desc = interpret_ma_trend(
        indicators['price_vs_ma10'],
        indicators['price_vs_ma20']
    )

    summary = f"""
Technical Indicators for {ticker}:

Momentum:
  {rsi_desc}

Trend:
  {trend_desc}

MACD:
  {macd_desc}

Volatility:
  20-day volatility: {indicators['volatility_20d']:.2f}%
  Typical daily move: ±{indicators['volatility_20d']:.2f}%
  2σ move (unusual): ±{indicators['volatility_20d'] * 2:.2f}%
"""

    return summary.strip()


# Testing
if __name__ == "__main__":
    print("Testing technical indicators module...\n")

    test_tickers = ["AMZN", "NVDA", "TSLA"]

    for ticker in test_tickers:
        print(f"=== {ticker} ===")
        print(get_technical_summary(ticker))
        print()
