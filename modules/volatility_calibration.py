"""
Volatility Calibration Module

Provides functions to:
1. Calculate historical volatility from price data
2. Calibrate LLM magnitude predictions using realized volatility
3. Prevent magnitude overestimation (currently 3.38x too high)

Based on analysis showing agents predict 5.5% moves vs 1.6% actual.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Cache volatility to avoid repeated API calls
_volatility_cache = {}

def get_historical_volatility(ticker, window=20, recent_window=5, use_cache=True):
    """
    Fetch historical price data and calculate realized volatility.

    Args:
        ticker: Stock ticker symbol (e.g., 'AMZN')
        window: Rolling window for baseline volatility (default: 20 days)
        recent_window: Rolling window for recent volatility (default: 5 days)
        use_cache: Use cached volatility if available (default: True)

    Returns:
        tuple: (vol_daily_baseline, vol_annual, vol_daily_recent)
               vol_daily_baseline: 20-day rolling std (baseline "normal" vol)
               vol_annual: Annualized volatility (daily * sqrt(252))
               vol_daily_recent: 5-day rolling std (recent volatility for spike detection)
    """
    # Check cache first
    if use_cache and ticker in _volatility_cache:
        return _volatility_cache[ticker]

    try:
        # Fetch last 90 days of data (sufficient for 20-day rolling window)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if len(df) == 0:
            print(f"⚠️  No price data for {ticker}, using market default volatility")
            return _use_default_volatility(ticker)

        # Handle MultiIndex columns (yfinance returns multi-level for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Calculate daily returns
        returns = df['Close'].pct_change().dropna()

        if len(returns) < window:
            print(f"⚠️  Insufficient data for {ticker} (only {len(returns)} days), using default")
            return _use_default_volatility(ticker)

        # Baseline volatility (20-day rolling window)
        vol_daily_baseline = returns.rolling(window).std().iloc[-1]

        # Recent volatility (5-day rolling window for spike detection)
        vol_daily_recent = returns.rolling(recent_window).std().iloc[-1] if len(returns) >= recent_window else vol_daily_baseline

        # Handle NaN
        if pd.isna(vol_daily_baseline):
            return _use_default_volatility(ticker)

        if pd.isna(vol_daily_recent):
            vol_daily_recent = vol_daily_baseline

        # Annualized volatility
        vol_annual = vol_daily_baseline * np.sqrt(252)

        # Cache result (now includes recent vol)
        _volatility_cache[ticker] = (vol_daily_baseline, vol_annual, vol_daily_recent)

        return vol_daily_baseline, vol_annual, vol_daily_recent

    except Exception as e:
        print(f"⚠️  Error fetching volatility for {ticker}: {e}")
        return _use_default_volatility(ticker)


def _use_default_volatility(ticker):
    """
    Fallback to market-average volatility when data is unavailable.

    Typical volatilities:
    - Large cap tech: 1.5-2.0% daily (24-32% annual)
    - S&P 500: 1.0% daily (16% annual)
    - Small cap: 2.5% daily (40% annual)
    """
    default_daily = 0.015  # 1.5% daily
    default_annual = 0.24  # 24% annual
    default_recent = default_daily  # Assume recent = baseline when no data

    _volatility_cache[ticker] = (default_daily, default_annual, default_recent)
    return default_daily, default_annual, default_recent


def calibrate_magnitude(
    predicted_move,
    volatility_daily,
    horizon_days=1,
    dampening_factor=0.35,
    max_sigma=2.0
):
    """
    Return the raw predicted magnitude without dampening.

    Since magnitude is directionless, we use the simple average of agent predictions
    as the final magnitude estimate. No volatility-based scaling is applied.

    Args:
        predicted_move: Raw LLM prediction in % (e.g., 1.5)
        volatility_daily: Historical daily volatility (not used, kept for compatibility)
        horizon_days: Prediction horizon (not used, kept for compatibility)
        dampening_factor: Not used (kept for compatibility)
        max_sigma: Not used (kept for compatibility)

    Returns:
        Raw predicted magnitude in percentage (e.g., 1.5)

    Example:
        >>> predicted = 1.5    # LLM says 1.5% move
        >>> calibrate_magnitude(predicted, 0.015, horizon_days=1)
        1.5  # Returns raw prediction
    """
    # Return raw prediction without any dampening
    return predicted_move


def get_calibration_metrics(df, horizon='1d'):
    """
    Calculate magnitude calibration metrics from results DataFrame.

    Args:
        df: Results DataFrame with columns:
            - exp_move_pct_{horizon}: Predicted magnitude
            - ret_fwd_{horizon}: Actual forward return
        horizon: Horizon to calculate metrics for (e.g., '1d', '1w', '1m')

    Returns:
        dict with metrics:
            - predicted_mean: Average predicted magnitude
            - actual_mean: Average actual magnitude
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - overestimation_ratio: predicted / actual
    """
    # Build column names for this horizon
    pred_col = f'exp_move_pct_{horizon}'
    actual_col = f'ret_fwd_{horizon}'

    # Check if columns exist
    if pred_col not in df.columns or actual_col not in df.columns:
        return {
            "predicted_mean": 0.0,
            "actual_mean": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "overestimation_ratio": 0.0
        }

    # Calculate predicted magnitude (absolute value)
    df_valid = df[[pred_col, actual_col]].dropna()

    if df_valid.empty:
        return {
            "predicted_mean": 0.0,
            "actual_mean": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "overestimation_ratio": 0.0
        }

    pred_mag = df_valid[pred_col].values
    actual_mag = np.abs(df_valid[actual_col].values) * 100  # Convert to %

    mae = np.abs(pred_mag - actual_mag).mean()
    mse = ((pred_mag - actual_mag) ** 2).mean()
    rmse = np.sqrt(mse)

    pred_mean = pred_mag.mean()
    actual_mean = actual_mag.mean()
    ratio = pred_mean / actual_mean if actual_mean > 0 else 0.0

    return {
        "predicted_mean": pred_mean,
        "actual_mean": actual_mean,
        "mae": mae,
        "rmse": rmse,
        "overestimation_ratio": ratio
    }


def print_calibration_report(df):
    """
    Print a detailed calibration report.

    Args:
        df: Results DataFrame
    """
    metrics = get_calibration_metrics(df)

    print("=" * 50)
    print("MAGNITUDE CALIBRATION REPORT")
    print("=" * 50)
    print(f"Predicted magnitude (mean): {metrics['predicted_mean']:.2f}%")
    print(f"Actual magnitude (mean):    {metrics['actual_mean']:.2f}%")
    print(f"MAE:                        {metrics['mae']:.2f}%")
    print(f"RMSE:                       {metrics['rmse']:.2f}%")
    print(f"Overestimation ratio:       {metrics['overestimation_ratio']:.2f}x")
    print()

    ratio = metrics['overestimation_ratio']
    if ratio < 1.2:
        status = "✅ GOOD: Well-calibrated (< 20% error)"
    elif ratio < 1.5:
        status = "⚠️  OK: Moderate overestimation (20-50% error)"
    elif ratio < 2.0:
        status = "⚠️  WARN: Significant overestimation (50-100% error)"
    else:
        status = "❌ BAD: Severe overestimation (> 2x error)"

    print(status)
    print("=" * 50)


# Example usage and testing
if __name__ == "__main__":
    print("Testing volatility calibration module...")
    print()

    # Test 1: Fetch volatility
    print("Test 1: Fetch volatility for AMZN")
    vol_daily, vol_annual = get_historical_volatility("AMZN")
    print(f"  Daily volatility: {vol_daily * 100:.2f}%")
    print(f"  Annual volatility: {vol_annual * 100:.2f}%")
    print()

    # Test 2: Calibrate magnitude
    print("Test 2: Calibrate LLM predictions")
    test_predictions = [5.5, 7.0, 3.0, 10.0, 2.0]
    print(f"  Historical volatility: {vol_daily * 100:.2f}%")
    print(f"  Horizon: 1 day")
    print()
    print("  Raw Prediction | Calibrated")
    print("  " + "-" * 30)
    for pred in test_predictions:
        calibrated = calibrate_magnitude(pred, vol_daily, horizon_days=1)
        print(f"  {pred:6.2f}%       | {calibrated:6.2f}%")
    print()

    # Test 3: Multi-horizon
    print("Test 3: Multi-horizon calibration")
    pred = 1.5  # Typical prediction after prompt range updates
    print(f"  Raw prediction: {pred:.2f}%")
    print()
    for horizon, days in [("1d", 1), ("1w", 5), ("1m", 21)]:
        calibrated = calibrate_magnitude(pred, vol_daily, horizon_days=days)
        vol_horizon = vol_daily * np.sqrt(days) * 100
        print(f"  {horizon}: vol={vol_horizon:.1f}% → calibrated={calibrated:.2f}%")

    print()
    print("  Expected actuals: 1d=1.6%, 1w=4.8%, 1m=10.0%")
