import pandas as pd
import numpy as np
import os
from scipy import stats
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool, Div, TabPanel, Tabs
from bokeh.layouts import column, row
from bokeh.palettes import Category10_10
from config import HORIZONS
from modules.volatility_calibration import get_calibration_metrics

# Set random seed for reproducibility
np.random.seed(42)

def format_headlines_for_tooltip(headlines_str, max_lines=10):
    """
    Splits headlines by '||' delimiter and formats them as HTML lines.
    """
    if not headlines_str or pd.isna(headlines_str):
        return "No headlines"

    headlines = [h.strip() for h in str(headlines_str).split("||") if h.strip()]
    # Take first max_lines headlines
    headlines = headlines[:max_lines]
    # Format as HTML list
    formatted = "<br>".join(f"• {h[:80]}{'...' if len(h) > 80 else ''}" for h in headlines)
    return formatted

def run_permutation_test(df_h, prob_col, n_permutations=1000):
    """
    Perform permutation test to validate model is better than random noise.

    The "Shuffle Check": Does the model actually read the news, or is it guessing?

    Method:
    1. Calculate true accuracy with real news-return pairs
    2. Shuffle returns randomly (breaking news-return relationship)
    3. Recalculate accuracy on shuffled data
    4. Repeat N times to build null distribution
    5. Compare true accuracy to null distribution

    Args:
        df_h: DataFrame with predictions and actuals
        prob_col: Column name for probability predictions
        n_permutations: Number of shuffles (default: 1000)

    Returns:
        dict with:
            - true_accuracy: Real model accuracy
            - null_mean: Average accuracy on shuffled data
            - null_std: Std dev of shuffled accuracies
            - p_value: Probability true accuracy is due to chance
            - percentile: Where true accuracy ranks vs shuffled
            - null_distribution: All shuffled accuracies
    """
    # Calculate true accuracy
    true_correct = (df_h["status"] == "Correct").sum()
    true_accuracy = (true_correct / len(df_h) * 100) if len(df_h) > 0 else 0

    # Run permutations
    null_accuracies = []

    for _ in range(n_permutations):
        # Shuffle actual returns (breaking news-return relationship)
        shuffled_returns = df_h["actual_return_pct"].sample(frac=1.0, random_state=None).values

        # Recalculate predictions with shuffled data
        # Prediction remains the same, but we're testing against random actuals
        shuffled_correct = ((df_h["pred_return_pct"].values > 0) == (shuffled_returns > 0)).sum()
        shuffled_accuracy = (shuffled_correct / len(df_h) * 100)

        null_accuracies.append(shuffled_accuracy)

    null_accuracies = np.array(null_accuracies)

    # Calculate statistics
    null_mean = null_accuracies.mean()
    null_std = null_accuracies.std()

    # P-value: proportion of shuffled accuracies >= true accuracy
    p_value = (null_accuracies >= true_accuracy).sum() / n_permutations

    # Percentile: where does true accuracy rank?
    percentile = (null_accuracies < true_accuracy).sum() / n_permutations * 100

    return {
        "true_accuracy": true_accuracy,
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "percentile": percentile,
        "null_distribution": null_accuracies
    }

def calculate_calibration_thresholds(df_h, prob_col):
    """
    Calculate calibration metrics using confidence thresholds.

    For each threshold, includes ALL predictions with confidence >= threshold.
    Shows actual accuracy at each confidence level.

    This answers: "When the model is X% confident, how often is it actually correct?"
    """
    # Define confidence thresholds to evaluate
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    calibration_data = []

    for threshold in thresholds:
        # Get all predictions with confidence >= threshold
        # Confidence is distance from 50%: max(prob, 1-prob)
        confidence = df_h[prob_col].apply(lambda p: max(p, 1-p))
        mask = confidence >= threshold
        subset = df_h[mask]
        n = len(subset)

        if n > 0:  # Only include thresholds with actual predictions
            n_correct = (subset["status"] == "Correct").sum()
            accuracy = (n_correct / n * 100)

            calibration_data.append({
                "threshold": threshold,
                "threshold_label": f"≥{threshold*100:.0f}%",
                "actual_accuracy": accuracy,
                "n": n
            })

    return calibration_data

def calculate_accuracy_metrics(df_h, prob_col):
    """
    Calculate accuracy metrics - simplified to just show overall stats.
    Detailed confidence analysis moved to calibration heatmap.
    """
    # Overall accuracy
    total = len(df_h)
    correct = (df_h["status"] == "Correct").sum()
    overall_acc = (correct / total * 100) if total > 0 else 0

    # Perform t-test against 50% (random chance)
    outcomes = (df_h["status"] == "Correct").astype(int)

    if len(outcomes) > 1:
        # Check for variance to avoid precision loss warnings
        if outcomes.var() > 1e-10:  # Only do t-test if there's actual variance
            t_stat, p_value = stats.ttest_1samp(outcomes, 0.5)
            if p_value < 0.001:
                sig_label = "***"
                sig_color = "#2ca02c"
            elif p_value < 0.01:
                sig_label = "**"
                sig_color = "#8fbc8f"
            elif p_value < 0.05:
                sig_label = "*"
                sig_color = "#ffa500"
            else:
                sig_label = "ns"
                sig_color = "#888"
            t_test_info = f"<div style='font-size: 10px; color: {sig_color}; margin-top: 5px;'>t-test vs 50%: t={t_stat:.2f}, p={p_value:.4f} {sig_label}</div>"
        else:
            # All outcomes are the same (no variance)
            t_test_info = ""
    else:
        t_test_info = ""

    # Build simplified HTML (just overall stats)
    html = f"""
    <div style="background: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Overall Directional Accuracy: {overall_acc:.1f}% ({correct}/{total}){t_test_info}</div>
    </div>
    """
    return html

def generate_bokeh_dashboard(df, output_folder):
    """
    Generates a Bokeh dashboard with ticker dropdown and improved tooltips.
    Supports "Council of Agents" dynamic tooltips.
    """
    # Ensure datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    out_path = os.path.join(output_folder, "dashboard.html")
    output_file(out_path, title="GDELT Stock Dashboard")

    df_pred = df.dropna(subset=["prob_up_1d"]).copy()
    tickers = list(df["ticker"].unique())
    colors = Category10_10

    # Create tabs for each ticker
    ticker_tabs = []

    # --- DYNAMIC TOOLTIP CONSTRUCTION ---
    # Check if we have MAKER Agent data in columns
    has_agents = "agent_fundamental" in df.columns

    base_tooltip = """
        <div style="width: 900px; max-width: 95vw; padding: 10px; word-wrap: break-word;">
            <div style="font-weight: bold; font-size: 14px; margin-bottom: 8px;">
                @ticker - @date{%F}
            </div>
            <div style="margin-bottom: 5px;">
                <b>Actual:</b> @act{0.00}% | <b>Predicted:</b> @pred{0.00}% | <b>Status:</b> @status
            </div>
            <div style="margin-bottom: 5px;">
                <b>Prob Up:</b> @prob_up{0.0%}
            </div>
    """
    
    if has_agents:
        # Check if we have 4-agent system (bull/bear) or legacy 3-agent
        has_bull_bear = "agent_bull" in df.columns and "agent_bear" in df.columns

        if has_bull_bear:
            base_tooltip += """
                <hr style="margin: 8px 0;">
                <div style="background: #eef; padding: 5px; border-radius: 3px; font-size: 11px;">
                    <b>Council of Analysts:</b><br>
                    <b>Analyst 1 (Growth-focused):</b> @agent_bull<br>
                    <b>Analyst 2 (Risk-focused):</b> @agent_bear<br>
                    <b>Analyst 3 (Technical):</b> @agent_technical<br>
                    <b>Analyst 4 (Sentiment):</b> @agent_sentiment
                </div>
            """
        else:
            base_tooltip += """
                <hr style="margin: 8px 0;">
                <div style="background: #eef; padding: 5px; border-radius: 3px; font-size: 11px;">
                    <b>Council of Agents:</b><br>
                    <b>Fundamentalist:</b> @agent_fundamental<br>
                    <b>Technician:</b> @agent_technical<br>
                    <b>Psychologist:</b> @agent_risk
                </div>
            """
    
    base_tooltip += """
            <hr style="margin: 8px 0;">
            <div style="margin-bottom: 8px;">
                <b>Headlines:</b>
                <div style="font-size: 11px; color: #555;">@headlines{safe}</div>
            </div>
            <hr style="margin: 8px 0;">
            <div>
                <b>Analysis:</b>
                <div style="font-size: 11px; color: #333; font-style: italic;">@justification</div>
            </div>
        </div>
    """

    for ticker_idx, ticker in enumerate(tickers):
        ticker_color = colors[ticker_idx % 10]
        horizon_tabs = []

        # Loop through horizons - each becomes a tab
        for h_name in HORIZONS.keys():
            prob_col = f"prob_up_{h_name}"
            exp_col = f"exp_move_pct_{h_name}"
            ret_col = f"ret_fwd_{h_name}"

            df_h = df_pred[df_pred["ticker"] == ticker].dropna(subset=[ret_col, prob_col, exp_col]).copy()
            if df_h.empty:
                continue

            # Calculations
            df_h["pred_mag"] = df_h[exp_col].astype(float) / 100.0
            df_h["pred_return_pct"] = np.where(df_h[prob_col] >= 0.5, df_h["pred_mag"] * 100, -df_h["pred_mag"] * 100)
            df_h["actual_return_pct"] = df_h[ret_col] * 100
            df_h["status"] = np.where((df_h["pred_return_pct"] > 0) == (df_h["actual_return_pct"] > 0), "Correct", "Wrong")
            df_h["color"] = np.where(df_h["status"] == "Correct", "#2ca02c", "#d62728")

            # Format headlines for tooltip
            df_h["headlines_formatted"] = df_h["headline_text"].apply(format_headlines_for_tooltip)

            # Get justification if available
            if "justification" in df_h.columns:
                df_h["justification_display"] = df_h["justification"].fillna("").astype(str)
            else:
                df_h["justification_display"] = ""

            df_h = df_h.sort_values("date")

            # Calculate accuracy metrics
            metrics_html = calculate_accuracy_metrics(df_h, prob_col)

            # Run permutation test (only if we have enough data)
            perm_html = ""
            if len(df_h) >= 30:  # Need at least 30 samples for meaningful test
                perm_results = run_permutation_test(df_h, prob_col, n_permutations=1000)

                # Determine significance
                if perm_results["p_value"] < 0.001:
                    sig_label = "Highly significant (p < 0.001)"
                    sig_color = "#2ca02c"
                    sig_symbol = "***"
                elif perm_results["p_value"] < 0.01:
                    sig_label = "Very significant (p < 0.01)"
                    sig_color = "#3cb371"
                    sig_symbol = "**"
                elif perm_results["p_value"] < 0.05:
                    sig_label = "Significant (p < 0.05)"
                    sig_color = "#ffa500"
                    sig_symbol = "*"
                else:
                    sig_label = "Not significant (p >= 0.05)"
                    sig_color = "#d62728"
                    sig_symbol = "ns"

                perm_html = f"""
                <div style="background: #e8f4f8; padding: 8px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {sig_color};">
                    <div style="font-weight: bold; font-size: 11px; color: {sig_color};">
                        Permutation Test: {sig_label} {sig_symbol}
                    </div>
                    <div style="font-size: 10px; color: #333; margin-top: 3px;">
                        True accuracy: {perm_results['true_accuracy']:.1f}% vs Random baseline: {perm_results['null_mean']:.1f}% ± {perm_results['null_std']:.1f}%
                    </div>
                    <div style="font-size: 10px; color: #555; margin-top: 2px;">
                        Percentile: {perm_results['percentile']:.1f}% (better than {perm_results['percentile']:.0f}% of shuffled data)
                    </div>
                </div>
                """

            # Add magnitude calibration quality indicator per ticker per horizon
            cal_metrics = get_calibration_metrics(df_h, horizon=h_name)

            # Determine calibration quality
            ratio = cal_metrics['overestimation_ratio']
            pred_mean = cal_metrics['predicted_mean']
            actual_mean = cal_metrics['actual_mean']

            if ratio == 0.0 or actual_mean == 0.0:
                # Skip if no data
                cal_html = ""
            else:
                # Determine quality and message
                if 0.8 <= ratio <= 1.2:
                    cal_status = "Well-calibrated"
                    cal_color = "#d4edda"
                    cal_msg = f"Predictions match actuals well (within ±20%)"
                elif 0.5 <= ratio < 0.8:
                    cal_status = "Underpredicting"
                    cal_color = "#fff3cd"
                    cal_msg = f"Avg prediction: {pred_mean:.1f}% vs actual: {actual_mean:.1f}%"
                elif ratio < 0.5:
                    cal_status = "Extreme Underpredicting"
                    cal_color = "#f8d7da"
                    cal_msg = f"Avg prediction: {pred_mean:.1f}% vs actual: {actual_mean:.1f}% (ratio: {ratio:.2f}x)"
                elif 1.2 < ratio <= 1.8:
                    cal_status = "Overpredicting"
                    cal_color = "#fff3cd"
                    cal_msg = f"Avg prediction: {pred_mean:.1f}% vs actual: {actual_mean:.1f}%"
                else:  # ratio > 1.8
                    cal_status = "Extreme Overpredicting"
                    cal_color = "#f8d7da"
                    cal_msg = f"Avg prediction: {pred_mean:.1f}% vs actual: {actual_mean:.1f}% (ratio: {ratio:.2f}x)"

                cal_html = f"""
                <div style="background: {cal_color}; padding: 8px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ffc107;">
                    <div style="font-weight: bold; font-size: 11px;">
                        {ticker} {h_name.upper()} Magnitude: {cal_status}
                    </div>
                    <div style="font-size: 10px; color: #555; margin-top: 3px;">
                        {cal_msg}
                    </div>
                </div>
                """

            metrics_html = perm_html + cal_html + metrics_html

            metrics_div = Div(text=metrics_html, width=700)

            p = figure(
                title=f"{ticker} - Horizon: {h_name.upper()}",
                x_axis_type="datetime",
                width=700,
                height=350,
                background_fill_color="#fafafa"
            )

            # Prepare data source
            data_dict = dict(
                date=df_h["date"].values,
                act=df_h["actual_return_pct"].values,
                pred=df_h["pred_return_pct"].values,
                ticker=[ticker] * len(df_h),
                status=df_h["status"].values,
                color=df_h["color"].values,
                headlines=df_h["headlines_formatted"].values,
                justification=df_h["justification_display"].values,
                prob_up=df_h[prob_col].values
            )
            
            # Add Agent columns if they exist
            if has_agents:
                # Check for 4-agent system (bull/bear)
                has_bull_bear = "agent_bull" in df_h.columns and "agent_bear" in df_h.columns

                if has_bull_bear:
                    data_dict["agent_bull"] = df_h["agent_bull"].fillna("-").values
                    data_dict["agent_bear"] = df_h["agent_bear"].fillna("-").values
                    data_dict["agent_technical"] = df_h["agent_technical"].fillna("-").values
                    data_dict["agent_sentiment"] = df_h["agent_sentiment"].fillna("-").values
                else:
                    # Legacy 3-agent system
                    data_dict["agent_fundamental"] = df_h["agent_fundamental"].fillna("-").values
                    data_dict["agent_technical"] = df_h["agent_technical"].fillna("-").values
                    data_dict["agent_risk"] = df_h["agent_risk"].fillna("-").values

            source = ColumnDataSource(data=data_dict)

            # Actual returns line + circles
            p.line('date', 'act', source=source, color=ticker_color, line_width=2, alpha=0.8, legend_label="Actual")
            p.scatter('date', 'act', source=source, color=ticker_color, size=8, marker="circle")

            # Predicted returns line + diamonds (colored by correct/wrong)
            p.line('date', 'pred', source=source, color=ticker_color, line_dash="dashed", alpha=0.5, legend_label="Predicted")
            p.scatter('date', 'pred', source=source, marker="diamond", size=10, fill_color='color', line_color=ticker_color)

            # Tooltip with viewport constraints
            hover = HoverTool(
                tooltips=base_tooltip,
                formatters={"@date": "datetime"},
                mode='mouse',
                attachment='horizontal',  # Attach horizontally to prevent vertical overflow
                show_arrow=False  # Remove arrow for cleaner look with large tooltips
            )
            p.add_tools(hover)

            p.legend.click_policy = "hide"
            p.legend.location = "top_left"

            # --- WHALE SCATTER PLOT (Magnitude Analysis) ---
            # Create scatter plot for predicted vs actual absolute returns
            pred_abs = np.abs(df_h["pred_return_pct"].values)
            actual_abs = np.abs(df_h["actual_return_pct"].values)

            whale_plot = figure(
                title=f"{ticker} - {h_name.upper()} Magnitude Analysis (\"Whale Plot\")",
                x_axis_label="Predicted Absolute Return (%)",
                y_axis_label="Actual Absolute Return (%)",
                width=700,
                height=350,
                background_fill_color="#fafafa"
            )

            # Scatter plot data
            whale_source = ColumnDataSource(data=dict(
                pred_abs=pred_abs,
                actual_abs=actual_abs,
                color=df_h["color"].values,
                date=df_h["date"].values
            ))

            # Add scatter points
            whale_plot.scatter('pred_abs', 'actual_abs', source=whale_source,
                             size=8, fill_color='color', line_color='color', alpha=0.6)

            # Add 45-degree diagonal line (perfect predictions)
            max_val = max(pred_abs.max(), actual_abs.max()) * 1.1
            whale_plot.line([0, max_val], [0, max_val],
                          line_width=2, line_dash="dashed", color="gray",
                          legend_label="Perfect Calibration", alpha=0.5)

            # Calculate correlation
            if len(pred_abs) > 1:
                correlation = np.corrcoef(pred_abs, actual_abs)[0, 1]
                whale_plot.add_layout(
                    Div(text=f"<div style='font-size: 11px; color: #666;'>Correlation: {correlation:.3f}</div>"),
                    'above'
                )

            whale_plot.legend.location = "top_left"
            whale_plot.legend.click_policy = "hide"

            # --- CONFIDENCE CALIBRATION TABLE (Meta-Cognition Analysis) ---
            # Create compact table showing accuracy at different confidence thresholds
            calibration_data = calculate_calibration_thresholds(df_h, prob_col)

            if calibration_data:
                # Determine quality based on whether accuracy improves with confidence
                # Good calibration: higher confidence → higher accuracy (monotonic relationship)
                if len(calibration_data) >= 2:
                    # Simple check: is highest confidence better than lowest?
                    highest_conf_acc = calibration_data[-1]["actual_accuracy"]
                    lowest_conf_acc = calibration_data[0]["actual_accuracy"]
                    acc_improvement = highest_conf_acc - lowest_conf_acc

                    # Also check overall trend
                    is_monotonic = all(calibration_data[i]["actual_accuracy"] <= calibration_data[i+1]["actual_accuracy"]
                                      for i in range(len(calibration_data)-1))

                    if is_monotonic and acc_improvement > 10:
                        cal_quality = "Well-calibrated (confidence correlates with accuracy)"
                        cal_color = "#f0f9f4"  # Very light green
                        border_color = "#5cb85c"
                    elif acc_improvement > 5:
                        cal_quality = "Moderately calibrated"
                        cal_color = "#fefef0"  # Very light yellow
                        border_color = "#f0ad4e"
                    elif acc_improvement < -5:
                        cal_quality = "Poorly calibrated (higher confidence = lower accuracy)"
                        cal_color = "#fef5f5"  # Very light red
                        border_color = "#d9534f"
                    else:
                        cal_quality = "Poorly calibrated (confidence doesn't predict accuracy)"
                        cal_color = "#fefef0"  # Very light yellow
                        border_color = "#f0ad4e"
                else:
                    cal_quality = "Insufficient data"
                    cal_color = "#f5f5f5"  # Light gray
                    border_color = "#999"

                # Build compact table
                table_rows = ""
                for d in calibration_data:
                    threshold_label = d["threshold_label"]
                    actual = d["actual_accuracy"]
                    n = d["n"]

                    # Simple color coding: higher accuracy = greener
                    if actual >= 70:
                        accuracy_color = "#2ca02c"  # Dark green
                    elif actual >= 60:
                        accuracy_color = "#5cb85c"  # Green
                    elif actual >= 55:
                        accuracy_color = "#ffa500"  # Orange
                    else:
                        accuracy_color = "#d62728"  # Red

                    table_rows += f"""
                    <tr>
                        <td style="padding: 3px 8px;">{threshold_label}</td>
                        <td style="padding: 3px 8px; text-align: center;">{n}</td>
                        <td style="padding: 3px 8px; text-align: center; color: {accuracy_color}; font-weight: bold;">{actual:.1f}%</td>
                    </tr>
                    """

                calibration_html = f"""
                <div style="background: {cal_color}; padding: 8px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid {border_color};">
                    <div style="font-weight: bold; font-size: 11px; margin-bottom: 5px;">
                        Confidence Calibration: {cal_quality}
                    </div>
                    <table style="width: 100%; font-size: 10px; border-collapse: collapse; background: white; border-radius: 3px;">
                        <thead>
                            <tr style="background: #f0f0f0; border-bottom: 2px solid #ddd;">
                                <th style="padding: 4px 8px; text-align: left;">Confidence</th>
                                <th style="padding: 4px 8px; text-align: center;">N</th>
                                <th style="padding: 4px 8px; text-align: center;">Actual Accuracy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                    <div style="font-size: 9px; color: #666; margin-top: 5px; font-style: italic;">
                        Shows: When model is ≥X% confident, how often is it correct?
                    </div>
                </div>
                """

                calibration_div = Div(text=calibration_html, width=700)

                # Create grid layout
                # Row 1: Metrics stats and Calibration table side-by-side
                row_top = row(metrics_div, calibration_div)

                # Row 2: Main time series plot and Whale plot
                row_plots = row(p, whale_plot)

                # Combine into a single horizon layout
                horizon_layout = column([row_top, row_plots])

                # Add this horizon as a tab
                horizon_tabs.append(TabPanel(child=horizon_layout, title=h_name.upper()))
            else:
                # No calibration data, just put main plot and whale plot side by side
                row1 = row(p, whale_plot)
                horizon_layout = column([metrics_div, row1])

                # Add this horizon as a tab
                horizon_tabs.append(TabPanel(child=horizon_layout, title=h_name.upper()))

        if horizon_tabs:
            # Create nested tabs for this ticker's horizons
            horizon_tabs_widget = Tabs(tabs=horizon_tabs)
            ticker_tabs.append(TabPanel(child=horizon_tabs_widget, title=ticker))

    # Build final layout
    layout_elements = []

    # Title
    layout_elements.append(Div(text="""
        <div style="font-family: sans-serif; text-align: center; margin-bottom: 20px;">
            <h2 style="color: #2c3e50;">GDELT Stock Prediction Dashboard</h2>
            <p style="color: #7f8c8d;">Select a ticker, then choose a horizon (1D, 1W, 1M) to view predictions vs actuals</p>
        </div>
    """))

    # Add tabs
    if ticker_tabs:
        tabs = Tabs(tabs=ticker_tabs)
        layout_elements.append(tabs)

    save(column(layout_elements))
    return out_path