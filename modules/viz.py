import pandas as pd
import numpy as np
import os
from scipy import stats
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, HoverTool, Div, TabPanel, Tabs
from bokeh.layouts import column
from bokeh.palettes import Category10_10
from config import HORIZONS

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

def calculate_accuracy_metrics(df_h, prob_col):
    """
    Calculate accuracy metrics at various confidence thresholds.
    Returns a formatted HTML string with accuracy statistics.
    """
    thresholds = [
        (0.0, 1.01, "All (0-100%)"),
        (0.0, 0.2, "80%+ Down (0-20%)"),
        (0.0, 0.3, "70%+ Down (0-30%)"),
        (0.0, 0.4, "60%+ Down (0-40%)"),
        (0.6, 1.01, "60%+ Up (60-100%)"),
        (0.7, 1.01, "70%+ Up (70-100%)"),
        (0.8, 1.01, "80%+ Up (80-100%)")
    ]

    metrics = []

    # Overall accuracy
    total = len(df_h)
    correct = (df_h["status"] == "Correct").sum()
    overall_acc = (correct / total * 100) if total > 0 else 0

    # Directional accuracy by threshold
    for low, high, label in thresholds:
        mask = (df_h[prob_col] >= low) & (df_h[prob_col] < high)
        subset = df_h[mask]
        n = len(subset)

        if n > 0:
            n_correct = (subset["status"] == "Correct").sum()
            acc = (n_correct / n * 100)

            # Magnitude accuracy (MAE)
            mae = np.abs(subset["pred_return_pct"] - subset["actual_return_pct"]).mean()

            metrics.append({
                "label": label,
                "n": n,
                "dir_acc": acc,
                "mae": mae
            })

    # Perform t-test against 50% (random chance)
    # Binary outcomes: 1 = correct, 0 = wrong
    outcomes = (df_h["status"] == "Correct").astype(int)

    # One-sample t-test: H0: mean = 0.5 (50% accuracy)
    if len(outcomes) > 1:
        t_stat, p_value = stats.ttest_1samp(outcomes, 0.5)

        # Determine significance level
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
        t_test_info = ""

    # Build HTML
    html = f"""
    <div style="background: #f9f9f9; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <div style="font-weight: bold; margin-bottom: 5px;">Overall Accuracy: {overall_acc:.1f}% ({correct}/{total}){t_test_info}</div>
    """

    if metrics:
        html += "<table style='width: 100%; font-size: 11px; border-collapse: collapse;'>"
        html += "<tr style='border-bottom: 1px solid #ddd;'><th>Confidence</th><th>N</th><th>Dir Acc</th><th>MAE</th></tr>"
        for m in metrics:
            html += f"<tr><td>{m['label']}</td><td>{m['n']}</td><td>{m['dir_acc']:.1f}%</td><td>{m['mae']:.2f}%</td></tr>"
        html += "</table>"

    html += "</div>"
    return html

def generate_bokeh_dashboard(df, output_folder):
    """
    Generates a Bokeh dashboard with ticker dropdown and improved tooltips.
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

    for ticker_idx, ticker in enumerate(tickers):
        ticker_color = colors[ticker_idx % 10]
        horizon_plots = []

        # Loop through horizons
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

            # Format headlines for tooltip (multiple lines)
            df_h["headlines_formatted"] = df_h["headline_text"].apply(format_headlines_for_tooltip)

            # Get justification if available
            if "justification" in df_h.columns:
                df_h["justification_display"] = df_h["justification"].fillna("").astype(str)
            else:
                df_h["justification_display"] = ""

            df_h = df_h.sort_values("date")

            # Calculate accuracy metrics
            metrics_html = calculate_accuracy_metrics(df_h, prob_col)
            metrics_div = Div(text=metrics_html, width=1000)

            p = figure(
                title=f"{ticker} - Horizon: {h_name.upper()}",
                x_axis_type="datetime",
                width=1000,
                height=400,
                background_fill_color="#fafafa"
            )

            source = ColumnDataSource(data=dict(
                date=df_h["date"].values,
                act=df_h["actual_return_pct"].values,
                pred=df_h["pred_return_pct"].values,
                ticker=[ticker] * len(df_h),
                status=df_h["status"].values,
                color=df_h["color"].values,
                headlines=df_h["headlines_formatted"].values,
                justification=df_h["justification_display"].values,
                prob_up=df_h[prob_col].values
            ))

            # Actual returns line + circles
            p.line('date', 'act', source=source, color=ticker_color, line_width=2, alpha=0.8, legend_label="Actual")
            p.scatter('date', 'act', source=source, color=ticker_color, size=8, marker="circle")

            # Predicted returns line + diamonds (colored by correct/wrong)
            p.line('date', 'pred', source=source, color=ticker_color, line_dash="dashed", alpha=0.5, legend_label="Predicted")
            p.scatter('date', 'pred', source=source, marker="diamond", size=10, fill_color='color', line_color=ticker_color)

            # Enhanced tooltip with formatted headlines and justification
            hover = HoverTool(
                tooltips="""
                <div style="width: 400px; padding: 10px;">
                    <div style="font-weight: bold; font-size: 14px; margin-bottom: 8px;">
                        @ticker - @date{%F}
                    </div>
                    <div style="margin-bottom: 5px;">
                        <b>Actual:</b> @act{0.00}% | <b>Predicted:</b> @pred{0.00}% | <b>Status:</b> @status
                    </div>
                    <div style="margin-bottom: 5px;">
                        <b>Prob Up:</b> @prob_up{0.0%}
                    </div>
                    <hr style="margin: 8px 0;">
                    <div style="margin-bottom: 8px;">
                        <b>Headlines:</b>
                        <div style="font-size: 11px; color: #555;">@headlines{safe}</div>
                    </div>
                    <hr style="margin: 8px 0;">
                    <div>
                        <b>LLM Justification:</b>
                        <div style="font-size: 11px; color: #333; font-style: italic;">@justification</div>
                    </div>
                </div>
                """,
                formatters={"@date": "datetime"},
                mode='mouse'
            )
            p.add_tools(hover)

            p.legend.click_policy = "hide"
            p.legend.location = "top_left"

            # Add metrics div then plot
            horizon_plots.append(metrics_div)
            horizon_plots.append(p)

        if horizon_plots:
            # Combine all horizon plots for this ticker into a panel
            ticker_layout = column(horizon_plots)
            ticker_tabs.append(TabPanel(child=ticker_layout, title=ticker))

    # Build final layout
    layout_elements = []

    # Title
    layout_elements.append(Div(text="""
        <div style="font-family: sans-serif; text-align: center; margin-bottom: 20px;">
            <h2 style="color: #2c3e50;">GDELT Stock Prediction Dashboard</h2>
            <p style="color: #7f8c8d;">Select a ticker tab to view predictions vs actuals</p>
        </div>
    """))

    # Add tabs
    if ticker_tabs:
        tabs = Tabs(tabs=ticker_tabs)
        layout_elements.append(tabs)

    save(column(layout_elements))
    return out_path
