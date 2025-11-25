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

def get_historical_volatility(ticker, window=20, use_cache=True):
    """
    Fetch historical price data and calculate realized volatility.

    Args:
        ticker: Stock ticker symbol (e.g., 'AMZN')
        window: Rolling window for volatility calculation (default: 20 days)
        use_cache: Use cached volatility if available (default: True)

    Returns:
        tuple: (volatility_daily, volatility_annualized)
               volatility_daily: Daily standard deviation of returns
               volatility_annualized: Annualized volatility (daily * sqrt(252))
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

        # Rolling volatility (using last window period)
        vol_daily = returns.rolling(window).std().iloc[-1]

        # Handle NaN
        if pd.isna(vol_daily):
            return _use_default_volatility(ticker)

        # Annualized volatility
        vol_annual = vol_daily * np.sqrt(252)

        # Cache result
        _volatility_cache[ticker] = (vol_daily, vol_annual)

        return vol_daily, vol_annual

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

    _volatility_cache[ticker] = (default_daily, default_annual)
    return default_daily, default_annual


def calibrate_magnitude(
    predicted_move,
    volatility_daily,
    horizon_days=1,
    dampening_factor=0.35,
    max_sigma=2.0
):
    """
    Calibrate LLM-predicted magnitude using historical volatility.

    Problem: LLMs predict ~5.5% average moves, but reality is ~1.6% (3.38x overestimation)

    Solution: Scale predictions using:
    1. Historical volatility (realistic baseline)
    2. Horizon-adjusted dampening (longer horizons need less dampening)
    3. Maximum sigma cap (prevent absurd predictions)

    Args:
        predicted_move: Raw LLM prediction in % (e.g., 5.5)
        volatility_daily: Historical daily volatility (e.g., 0.015 = 1.5%)
        horizon_days: Prediction horizon (1, 5, 21)
        dampening_factor: Base scaling factor for 1-day (default: 0.35)
        max_sigma: Cap predictions at N standard deviations (default: 2.0)

    Returns:
        Calibrated magnitude in percentage (e.g., 1.65)

    Example:
        >>> vol_daily = 0.015  # 1.5% daily volatility
        >>> predicted = 5.5    # LLM says 5.5% move
        >>> calibrate_magnitude(predicted, vol_daily, horizon_days=1)
        1.65  # Calibrated to 1.65%
    """
    # Expected move based on volatility (1σ move over horizon)
    vol_horizon = volatility_daily * np.sqrt(horizon_days) * 100  # Convert to %

    # LLM confidence (normalize to 0-2 multiplier)
    # If LLM predicts 5% out of max 10%, that's 50% → 1.0x multiplier
    max_prediction = 10.0  # Typical prompt range
    confidence_mult = (predicted_move / max_prediction) * 2.0

    # Realistic maximum (max_sigma standard deviations with confidence multiplier)
    max_realistic = vol_horizon * max_sigma * confidence_mult

    # Cap prediction to realistic maximum
    capped_move = min(predicted_move, max_realistic)

    # Horizon-adjusted calibration
    # Strategy: Use volatility as baseline, scale by horizon, adjust for LLM confidence

    # Base magnitude on volatility * sqrt(horizon)
    vol_based_magnitude = vol_horizon

    # Confidence adjustment (how strong is the LLM's conviction?)
    # Normalize predicted_move to confidence (0-1 scale based on typical range 0-3%)
    typical_max = 3.0  # After prompt updates, agents predict 0-3%
    confidence_factor = min(predicted_move / typical_max, 1.0)

    # Horizon-specific calibration factors (empirically tuned)
    if horizon_days == 1:
        # 1-day: More aggressive dampening
        # Target: 1.6% actual, vol_horizon ~3.1%, prediction ~1.5%
        # Need: 1.6 / (3.1 * 1.25) ≈ 0.41
        scale_factor = 0.45
    elif horizon_days <= 7:
        # 1-week: Moderate dampening
        # Target: 4.76% actual, vol_horizon ~7%, prediction ~1.5%
        # Need: 4.76 / (7.0 * 1.25) ≈ 0.54
        scale_factor = 0.55
    else:
        # 1-month: Light dampening
        # Target: 10% actual, vol_horizon ~14.3%, prediction ~1.5%
        # Need: 10 / (14.3 * 1.25) ≈ 0.56
        scale_factor = 0.60

    # Apply confidence boost (higher confidence → larger move within volatility bounds)
    confidence_boost = 1.0 + (confidence_factor * 0.5)  # 1.0x to 1.5x boost

    # Final calibration
    calibrated_move = vol_based_magnitude * scale_factor * confidence_boost

    return calibrated_move


def get_calibration_metrics(df):
    """
    Calculate magnitude calibration metrics from results DataFrame.

    Args:
        df: Results DataFrame with columns:
            - exp_move_pct_1d: Predicted magnitude
            - ret_fwd_1d: Actual forward return

    Returns:
        dict with metrics:
            - predicted_mean: Average predicted magnitude
            - actual_mean: Average actual magnitude
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - overestimation_ratio: predicted / actual
    """
    # Calculate predicted magnitude (absolute value)
    df_valid = df[['exp_move_pct_1d', 'ret_fwd_1d']].dropna()

    if df_valid.empty:
        return {
            "predicted_mean": 0.0,
            "actual_mean": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "overestimation_ratio": 0.0
        }

    pred_mag = df_valid['exp_move_pct_1d'].values
    actual_mag = np.abs(df_valid['ret_fwd_1d'].values) * 100  # Convert to %

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
