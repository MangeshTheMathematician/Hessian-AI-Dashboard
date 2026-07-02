# Import os so the app can safely read API keys and local file paths without hardcoding secrets.
import os

# Import Streamlit because it creates the interactive web dashboard.
import streamlit as st

# Import pandas because the market matrix is a table of timestamps, prices, and volumes.
import pandas as pd

# Import NumPy because the math engine needs NaN handling and numerical calculations.
import numpy as np

# Import Plotly Graph Objects because it gives us professional interactive charts.
import plotly.graph_objects as go

# Import Plotly subplots because the main chart combines price, volume, and Z-score in one visual stack.
from plotly.subplots import make_subplots

# Import dotenv so local .env files can store DATABENTO_API_KEY and GROQ_API_KEY safely.
from dotenv import load_dotenv

# Load environment variables from a local .env file if one exists.
load_dotenv()

# Try to import the OpenAI SDK, because Groq exposes an OpenAI-compatible API endpoint.
try:
    # Import OpenAI client only if the package is installed in the user's environment.
    from openai import OpenAI

# If OpenAI is not installed, keep the dashboard alive and simply disable the AI summary.
except Exception:
    # Set OpenAI to None so later code can check whether AI support is available.
    OpenAI = None


# Configure Streamlit with a wide layout so charts have enough horizontal room.
st.set_page_config(
    page_title="SPY Micro-VWAP Signal Dashboard",
    layout="wide"
)

# Add lightweight CSS to make metric cards and spacing more professional without hiding Streamlit behavior.
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.75rem;}
        .small-note {color: #9ca3af; font-size: 0.90rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# Show the dashboard title at the top of the page.
st.title("SPY Micro-VWAP, Volume Z-Score & Dynamic Exit Dashboard")

# Explain the product in one professional sentence.
st.caption(
    "A teachable intraday signal dashboard using Databento-style 1-second OHLCV data, VWAP, rolling volatility, volume Z-score, and a volatility-based exit offset."
)


# Define a helper function to create a small sample market matrix if no cache file exists.
def create_sample_spy_matrix() -> pd.DataFrame:
    # Create a timestamp range that mimics 1-second SPY intraday bars.
    timestamps = pd.date_range("2024-01-05 14:30:00", periods=240, freq="s")

    # Create a stable base price around a SPY-style level.
    base_price = 468.00

    # Create a deterministic pseudo-random generator so the sample is repeatable.
    rng = np.random.default_rng(42)

    # Simulate small 1-second returns around zero to mimic micro price movement.
    returns = rng.normal(loc=0.0, scale=0.00008, size=len(timestamps))

    # Convert returns into a price path by compounding from the base price.
    close = base_price * np.cumprod(1 + returns)

    # Create normal volumes around 1,100 shares per second.
    volume = rng.integers(low=700, high=1500, size=len(timestamps)).astype(float)

    # Force one volume spike and one price drop to demonstrate the signal visually.
    shock_index = 170

    # Increase the chosen second's volume so the volume Z-score can become extreme.
    volume[shock_index] = 7000

    # Push the close price lower after the shock so price can fall below VWAP.
    close[shock_index:] = close[shock_index:] - 0.18

    # Build a DataFrame with the same basic columns expected from the cache file.
    sample_df = pd.DataFrame(
        {
            "ts_event": timestamps,
            "close": close,
            "volume": volume,
        }
    )

    # Return the sample market matrix.
    return sample_df


# Define a helper function to normalize timestamp columns from different CSV exports.
def normalize_market_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Copy the input so the original DataFrame is not mutated unexpectedly.
    clean_df = df.copy()

    # Build a list of possible timestamp column names that may come from Databento or CSV reset_index output.
    timestamp_candidates = ["ts_event", "timestamp", "time", "datetime", "index"]

    # Find the first timestamp column that exists in the DataFrame.
    timestamp_column = next((col for col in timestamp_candidates if col in clean_df.columns), None)

    # If a timestamp column exists, convert it to pandas datetime for charting.
    if timestamp_column is not None:
        clean_df["ts_event"] = pd.to_datetime(clean_df[timestamp_column], errors="coerce")

    # If no timestamp column exists, create a simple integer timestamp index for fallback charting.
    else:
        clean_df["ts_event"] = pd.RangeIndex(start=0, stop=len(clean_df), step=1)

    # Force close prices to numeric values so arithmetic does not fail because of strings.
    clean_df["close"] = pd.to_numeric(clean_df["close"], errors="coerce")

    # Force volume to numeric values so rolling mean and standard deviation work correctly.
    clean_df["volume"] = pd.to_numeric(clean_df["volume"], errors="coerce")

    # Drop rows where either close or volume is missing, because those rows cannot be used in VWAP.
    clean_df = clean_df.dropna(subset=["close", "volume"]).reset_index(drop=True)

    # Sort by timestamp so cumulative calculations move forward in time.
    clean_df = clean_df.sort_values("ts_event").reset_index(drop=True)

    # Return the cleaned market matrix.
    return clean_df


# Define the main quantitative feature-engineering function.
def compute_signal_features(
    raw_df: pd.DataFrame,
    z_window: int,
    vol_window: int,
    z_threshold: float,
    exit_sigma_multiplier: float,
) -> pd.DataFrame:
    # Normalize and clean the raw market matrix first.
    df = normalize_market_matrix(raw_df)

    # Dollar volume means the notional dollar amount traded in each bar.
    df["dollar_volume"] = df["close"] * df["volume"]

    # Cumulative dollar volume keeps a running total of traded dollars from the beginning of the sample.
    df["cum_dollar_volume"] = df["dollar_volume"].cumsum()

    # Cumulative volume keeps a running total of traded shares from the beginning of the sample.
    df["cum_volume"] = df["volume"].cumsum()

    # Replace zero cumulative volume with NaN to avoid division-by-zero in VWAP.
    safe_cum_volume = df["cum_volume"].replace(0, np.nan)

    # VWAP is cumulative traded dollars divided by cumulative traded shares.
    df["vwap"] = df["cum_dollar_volume"] / safe_cum_volume

    # Percentage return measures how much the close price changed from one second to the next.
    df["returns"] = df["close"].pct_change()

    # Rolling volatility is the standard deviation of recent percentage returns.
    df["volatility_rolling"] = df["returns"].rolling(window=vol_window, min_periods=vol_window).std()

    # Rolling volume mean is the average recent volume over the selected window.
    df["vol_mean"] = df["volume"].rolling(window=z_window, min_periods=z_window).mean()

    # Rolling volume standard deviation measures normal volume variation over the selected window.
    df["vol_std"] = df["volume"].rolling(window=z_window, min_periods=z_window).std()

    # Replace zero standard deviation with NaN so the Z-score does not divide by zero.
    safe_vol_std = df["vol_std"].replace(0, np.nan)

    # Volume Z-score measures how many standard deviations current volume is above or below normal.
    df["vol_z_score"] = (df["volume"] - df["vol_mean"]) / safe_vol_std

    # Price-vs-VWAP spread is positive if price is above VWAP and negative if price is below VWAP.
    df["price_vwap_gap"] = df["close"] - df["vwap"]

    # VWAP gap in basis points makes the price-vs-VWAP distance comparable across price levels.
    df["price_vwap_gap_bps"] = (df["price_vwap_gap"] / df["vwap"]) * 10000

    # Price condition becomes true when spot price is below VWAP, suggesting downside pressure versus average traded cost.
    df["price_condition"] = df["close"] < df["vwap"]

    # Volume condition becomes true when the Z-score is above the selected anomaly threshold.
    df["volume_condition"] = df["vol_z_score"] > z_threshold

    # Short signal requires both price weakness and abnormal volume at the same time.
    df["short_signal"] = df["price_condition"] & df["volume_condition"]

    # Dynamic offset converts recent percentage volatility into a dollar price distance.
    df["dynamic_offset"] = df["close"] * df["volatility_rolling"] * exit_sigma_multiplier

    # For a short trade, the buy-to-cover limit is placed below the entry price by the dynamic offset.
    df["short_exit_target"] = df["close"] - df["dynamic_offset"]

    # Signal strength combines Z-score magnitude and VWAP gap direction for ranking events.
    df["signal_strength"] = np.where(
        df["short_signal"],
        df["vol_z_score"] * np.abs(df["price_vwap_gap_bps"]),
        0.0,
    )

    # Return the feature-rich DataFrame.
    return df


# Define a helper function to create a risk summary using Groq if enabled and available.
def generate_risk_summary(
    ticker: str,
    spot: float,
    vwap: float,
    z_score: float,
    volatility: float,
    exit_target: float,
    z_threshold: float,
    model_name: str,
) -> str:
    # Read the Groq API key from the environment.
    groq_key = os.getenv("GROQ_API_KEY")

    # If the OpenAI SDK is missing, return a helpful local message instead of crashing.
    if OpenAI is None:
        return "AI summary unavailable: install the OpenAI Python package with `pip install openai`."

    # If the Groq API key is missing, return a local deterministic explanation.
    if not groq_key:
        return "AI summary unavailable: GROQ_API_KEY is missing from the .env file."

    # Create an OpenAI-compatible client that points to Groq's endpoint.
    client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

    # Use a strict system prompt so the model summarizes risk instead of making trading decisions.
    system_prompt = (
        "You are a professional quantitative risk analyst. "
        "Explain only the provided signal facts. Do not recommend new trades."
    )

    # Build the user prompt from exact numeric features computed by the deterministic engine.
    user_prompt = f"""
Ticker: {ticker}
Signal type: short-entry alert
Spot price: {spot:.4f}
VWAP: {vwap:.4f}
Volume Z-score: {z_score:.2f}
Z-score threshold: {z_threshold:.2f}
Rolling volatility: {volatility:.8f}
Short exit target: {exit_target:.4f}

Write exactly two sentences explaining why the deterministic engine triggered and where the exit target was placed.
"""

    # Ask Groq for a concise risk note.
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    # Return the model's text answer.
    return response.choices[0].message.content


# -----------------------------
# Inputs used by every section.
# -----------------------------

# Create the sidebar controls before rendering the eight dashboard sections.
st.sidebar.header("Controls")

# Let the user choose the CSV cache location created by db_tp.py.
cache_file = st.sidebar.text_input("Local cache CSV", "spy_matrix_cache.csv")

# Allow the user to upload a CSV from another environment without changing the code.
uploaded_csv = st.sidebar.file_uploader("Optional CSV upload", type=["csv"])

# Let the user choose a ticker label for display and AI summary.
ticker = st.sidebar.text_input("Ticker label", "SPY")

# Let the user choose the rolling window for the volume Z-score.
z_window = st.sidebar.slider("Volume Z-score rolling window", min_value=20, max_value=1200, value=120, step=10)

# Let the user choose the rolling window for price volatility.
vol_window = st.sidebar.slider("Rolling volatility window", min_value=10, max_value=300, value=60, step=5)

# Let the user choose the anomaly threshold.
z_threshold = st.sidebar.slider("Z-score trigger threshold", min_value=0.5, max_value=6.0, value=3.0, step=0.1)

# Let the user choose how many volatility units are used for the exit target.
exit_sigma_multiplier = st.sidebar.slider("Exit offset volatility multiplier", min_value=0.25, max_value=5.0, value=1.0, step=0.25)

# Let the user control how many recent rows are emphasized in charts.
chart_tail = st.sidebar.slider("Rows shown in main chart", min_value=50, max_value=1000, value=240, step=10)

# Let the user enable or disable the LLM risk note.
enable_ai_summary = st.sidebar.checkbox("Enable Groq risk summary", value=False)

# Let the user select a Groq model name without changing code.
groq_model = st.sidebar.text_input("Groq model", "llama-3.3-70b-versatile")

# Load data from uploaded CSV, local cache, or sample fallback.
if uploaded_csv is not None:
    # Read the uploaded file when a user provides one through the browser.
    raw_market_df = pd.read_csv(uploaded_csv)

    # Track the source label for the dashboard.
    data_source_label = "uploaded CSV"

elif os.path.exists(cache_file):
    # Read the local cache file when it exists.
    raw_market_df = pd.read_csv(cache_file)

    # Track the source label for the dashboard.
    data_source_label = f"local cache: {cache_file}"

else:
    # Create sample data when no real cache exists so the dashboard remains teachable.
    raw_market_df = create_sample_spy_matrix()

    # Track the source label for the dashboard.
    data_source_label = "synthetic sample fallback"

# Compute every deterministic quant feature used by the dashboard.
df = compute_signal_features(
    raw_df=raw_market_df,
    z_window=z_window,
    vol_window=vol_window,
    z_threshold=z_threshold,
    exit_sigma_multiplier=exit_sigma_multiplier,
)

# Keep only the tail selected by the user for the main visuals.
chart_df = df.tail(chart_tail).copy()

# Extract rows where the signal fired.
signal_df = df[df["short_signal"]].copy()

# Select the first trigger if one exists.
first_signal = signal_df.iloc[0] if not signal_df.empty else None

# Select the latest usable row for current metrics.
latest_valid = df.dropna(subset=["vwap", "vol_z_score", "volatility_rolling"]).tail(1)

# If there is no fully valid row, use the last row as fallback.
latest_row = latest_valid.iloc[0] if not latest_valid.empty else df.iloc[-1]


# =========================
# 1. Executive Summary
# =========================
st.header("1. Executive Summary")

# Create eight metric columns so the user sees the state of the engine immediately.
metric_cols = st.columns(8)

# Show the latest spot close price.
metric_cols[0].metric("Latest Spot", f"${latest_row['close']:.2f}")

# Show the latest VWAP.
metric_cols[1].metric("Latest VWAP", f"${latest_row['vwap']:.2f}" if pd.notna(latest_row["vwap"]) else "N/A")

# Show the latest price-vs-VWAP gap in bps.
metric_cols[2].metric("VWAP Gap", f"{latest_row['price_vwap_gap_bps']:.1f} bps" if pd.notna(latest_row["price_vwap_gap_bps"]) else "N/A")

# Show the latest volume Z-score.
metric_cols[3].metric("Latest Vol Z", f"{latest_row['vol_z_score']:.2f}" if pd.notna(latest_row["vol_z_score"]) else "N/A")

# Show the maximum Z-score observed in the full sample.
metric_cols[4].metric("Max Vol Z", f"{df['vol_z_score'].max(skipna=True):.2f}")

# Show the number of deterministic short signals.
metric_cols[5].metric("Short Signals", f"{len(signal_df)}")

# Show the latest rolling volatility.
metric_cols[6].metric("Rolling Vol", f"{latest_row['volatility_rolling']:.5f}" if pd.notna(latest_row["volatility_rolling"]) else "N/A")

# Show the latest dynamic offset in dollars.
metric_cols[7].metric("Exit Offset", f"${latest_row['dynamic_offset']:.4f}" if pd.notna(latest_row["dynamic_offset"]) else "N/A")

# Explain whether the engine found a trigger.
if first_signal is not None:
    # Report first signal in a professional success/warning block.
    st.warning(
        f"First deterministic short signal fired at {first_signal['ts_event']} with close ${first_signal['close']:.2f}, "
        f"VWAP ${first_signal['vwap']:.2f}, and volume Z-score {first_signal['vol_z_score']:.2f}."
    )

# If no signal exists, explain that the deterministic rules did not fire.
else:
    # Report clean scan result.
    st.success("No short signal fired under the selected price-vs-VWAP and volume-Z threshold rules.")


# =========================
# 2. Inputs / Controls
# =========================
st.header("2. Inputs / Controls")

# Build a compact table that documents the exact configuration used to produce results.
input_summary = pd.DataFrame(
    {
        "Input": [
            "Data source",
            "Ticker label",
            "Rows loaded",
            "Volume Z-score window",
            "Rolling volatility window",
            "Z-score threshold",
            "Exit volatility multiplier",
            "AI summary enabled",
        ],
        "Value": [
            data_source_label,
            ticker,
            f"{len(df):,}",
            f"{z_window} rows",
            f"{vol_window} rows",
            f"> {z_threshold:.2f}",
            f"{exit_sigma_multiplier:.2f}x rolling volatility",
            str(enable_ai_summary),
        ],
    }
)

# Display the input summary table in the body so screenshots capture the assumptions.
st.dataframe(input_summary, hide_index=True, use_container_width=True)


# =========================
# 3. Key Formulas
# =========================
st.header("3. Key Formulas")

# Put formulas in an expander so the dashboard stays readable but still teachable.
with st.expander("Open formula sheet", expanded=True):
    # Split formulas into two columns for cleaner reading.
    formula_left, formula_right = st.columns(2)

    # Place VWAP and volume formulas on the left.
    with formula_left:
        # Show cumulative dollar volume formula.
        st.markdown("**Cumulative Dollar Volume**")
        st.latex(r"CDV_t = \sum_{i=1}^{t} Price_i \times Volume_i")

        # Show VWAP formula.
        st.markdown("**VWAP**")
        st.latex(r"VWAP_t = \frac{\sum_{i=1}^{t} Price_i \times Volume_i}{\sum_{i=1}^{t} Volume_i}")

        # Show volume Z-score formula.
        st.markdown("**Volume Z-Score**")
        st.latex(r"Z_{vol,t} = \frac{Volume_t - \mu_{vol,t}}{\sigma_{vol,t}}")

    # Place signal and exit formulas on the right.
    with formula_right:
        # Show rolling volatility formula.
        st.markdown("**Rolling Volatility**")
        st.latex(r"\sigma_{price,t} = std(r_{t-n+1},...,r_t)")

        # Show entry rule formula.
        st.markdown("**Short Entry Rule**")
        st.latex(r"Signal_t = (Close_t < VWAP_t) \land (Z_{vol,t} > Threshold)")

        # Show dynamic exit target formula.
        st.markdown("**Volatility-Based Short Exit Target**")
        st.latex(r"ExitTarget_t = Entry_t - Entry_t \times \sigma_{price,t} \times k")


# =========================
# 4. Main Visualization
# =========================
st.header("4. Main Visualization")

# Create a three-row subplot to connect price, volume, and volume anomaly visually.
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        "Price vs VWAP with Short-Signal Markers",
        "Volume with Rolling Mean",
        "Volume Z-Score with Trigger Threshold",
    ),
)

# Add close price to the first row.
fig.add_trace(
    go.Scatter(x=chart_df["ts_event"], y=chart_df["close"], mode="lines", name="Close"),
    row=1,
    col=1,
)

# Add VWAP to the first row.
fig.add_trace(
    go.Scatter(x=chart_df["ts_event"], y=chart_df["vwap"], mode="lines", name="VWAP"),
    row=1,
    col=1,
)

# Select signal rows that are also inside the displayed chart range.
chart_signal_df = chart_df[chart_df["short_signal"]].copy()

# Add signal markers only if at least one signal appears in the chart window.
if not chart_signal_df.empty:
    # Plot short entries as markers on the price chart.
    fig.add_trace(
        go.Scatter(
            x=chart_signal_df["ts_event"],
            y=chart_signal_df["close"],
            mode="markers",
            name="Short Signal",
            marker=dict(size=11, symbol="triangle-down"),
        ),
        row=1,
        col=1,
    )

# Add raw volume bars to the second row.
fig.add_trace(
    go.Bar(x=chart_df["ts_event"], y=chart_df["volume"], name="Volume", opacity=0.55),
    row=2,
    col=1,
)

# Add rolling volume mean to the second row.
fig.add_trace(
    go.Scatter(x=chart_df["ts_event"], y=chart_df["vol_mean"], mode="lines", name="Rolling Volume Mean"),
    row=2,
    col=1,
)

# Add volume Z-score line to the third row.
fig.add_trace(
    go.Scatter(x=chart_df["ts_event"], y=chart_df["vol_z_score"], mode="lines", name="Volume Z-Score"),
    row=3,
    col=1,
)

# Add the Z-score threshold line to the third row.
fig.add_trace(
    go.Scatter(
        x=chart_df["ts_event"],
        y=[z_threshold] * len(chart_df),
        mode="lines",
        name="Z Threshold",
        line=dict(dash="dash"),
    ),
    row=3,
    col=1,
)

# Apply a dark professional theme and chart sizing.
fig.update_layout(
    template="plotly_dark",
    height=850,
    hovermode="x unified",
    legend_title="Series",
    margin=dict(l=20, r=20, t=70, b=20),
)

# Label the y-axes so the viewer does not confuse price, shares, and Z-score.
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Shares", row=2, col=1)
fig.update_yaxes(title_text="Z-Score", row=3, col=1)

# Render the combined visualization.
st.plotly_chart(fig, use_container_width=True)

# Add a secondary chart for the dynamic exit target if signals exist.
if not signal_df.empty:
    # Create a signal-only chart for entry and exit target comparison.
    exit_fig = go.Figure()

    # Add entry close prices for signal rows.
    exit_fig.add_trace(go.Scatter(x=signal_df["ts_event"], y=signal_df["close"], mode="markers+lines", name="Short Entry"))

    # Add exit target levels for signal rows.
    exit_fig.add_trace(go.Scatter(x=signal_df["ts_event"], y=signal_df["short_exit_target"], mode="markers+lines", name="Buy-to-Cover Target"))

    # Apply dark theme and clear labels.
    exit_fig.update_layout(
        template="plotly_dark",
        height=360,
        title="Signal Entries vs Volatility-Based Exit Targets",
        xaxis_title="Timestamp",
        yaxis_title="Price",
        hovermode="x unified",
    )

    # Render the exit target chart.
    st.plotly_chart(exit_fig, use_container_width=True)


# =========================
# 5. Risk Metrics
# =========================
st.header("5. Risk Metrics")

# Calculate risk summary numbers with safe NaN handling.
max_z = df["vol_z_score"].max(skipna=True)
mean_z = df["vol_z_score"].mean(skipna=True)
latest_gap_bps = latest_row["price_vwap_gap_bps"]
max_abs_gap_bps = df["price_vwap_gap_bps"].abs().max(skipna=True)
max_rolling_vol = df["volatility_rolling"].max(skipna=True)
first_exit_offset = first_signal["dynamic_offset"] if first_signal is not None else np.nan

# Show risk metrics in columns.
risk_cols = st.columns(6)

# Show maximum observed volume anomaly.
risk_cols[0].metric("Max Volume Z", f"{max_z:.2f}" if pd.notna(max_z) else "N/A")

# Show average Z-score to detect broad volume pressure.
risk_cols[1].metric("Mean Volume Z", f"{mean_z:.2f}" if pd.notna(mean_z) else "N/A")

# Show latest VWAP gap in basis points.
risk_cols[2].metric("Latest VWAP Gap", f"{latest_gap_bps:.1f} bps" if pd.notna(latest_gap_bps) else "N/A")

# Show maximum absolute VWAP gap in basis points.
risk_cols[3].metric("Max Abs VWAP Gap", f"{max_abs_gap_bps:.1f} bps" if pd.notna(max_abs_gap_bps) else "N/A")

# Show maximum rolling volatility.
risk_cols[4].metric("Max Rolling Vol", f"{max_rolling_vol:.5f}" if pd.notna(max_rolling_vol) else "N/A")

# Show the first signal exit offset.
risk_cols[5].metric("First Signal Offset", f"${first_exit_offset:.4f}" if pd.notna(first_exit_offset) else "N/A")

# Show the signal table when there are signals.
if not signal_df.empty:
    # Choose the most relevant columns for a PM/risk review.
    signal_table = signal_df[
        [
            "ts_event",
            "close",
            "vwap",
            "price_vwap_gap_bps",
            "volume",
            "vol_mean",
            "vol_z_score",
            "volatility_rolling",
            "dynamic_offset",
            "short_exit_target",
            "signal_strength",
        ]
    ].copy()

    # Display the signal table with finance-friendly formatting.
    st.dataframe(
        signal_table.style.format(
            {
                "close": "${:.4f}",
                "vwap": "${:.4f}",
                "price_vwap_gap_bps": "{:.2f}",
                "volume": "{:,.0f}",
                "vol_mean": "{:,.2f}",
                "vol_z_score": "{:.2f}",
                "volatility_rolling": "{:.8f}",
                "dynamic_offset": "${:.4f}",
                "short_exit_target": "${:.4f}",
                "signal_strength": "{:.2f}",
            }
        ),
        use_container_width=True,
    )


# =========================
# 6. Interpretation
# =========================
st.header("6. Interpretation")

# Explain the deterministic rule in words.
st.markdown(
    f"""
The engine triggers only when **both** conditions are true:

1. The latest close is below VWAP, meaning price is trading below the volume-weighted average cost basis.
2. The volume Z-score is above **{z_threshold:.2f}**, meaning current volume is unusually high versus its recent rolling distribution.

The AI component, when enabled, does **not** decide the trade. It only summarizes the deterministic math in plain English.
"""
)

# If a signal exists, explain the first signal numerically.
if first_signal is not None:
    # Build the deterministic interpretation using exact numbers.
    deterministic_summary = (
        f"At the first signal, close was ${first_signal['close']:.4f}, VWAP was ${first_signal['vwap']:.4f}, "
        f"volume Z-score was {first_signal['vol_z_score']:.2f}, and rolling volatility was {first_signal['volatility_rolling']:.8f}. "
        f"The short exit target was ${first_signal['short_exit_target']:.4f}."
    )

    # Show deterministic summary first so the user has an AI-free explanation.
    st.info(deterministic_summary)

    # Generate AI summary only if the user explicitly enabled it.
    if enable_ai_summary:
        # Use a spinner while waiting for the API call.
        with st.spinner("Generating Groq risk brief..."):
            # Try to generate an AI summary but handle failures gracefully.
            try:
                # Call the AI summary function using first-signal facts.
                ai_summary = generate_risk_summary(
                    ticker=ticker,
                    spot=float(first_signal["close"]),
                    vwap=float(first_signal["vwap"]),
                    z_score=float(first_signal["vol_z_score"]),
                    volatility=float(first_signal["volatility_rolling"]),
                    exit_target=float(first_signal["short_exit_target"]),
                    z_threshold=float(z_threshold),
                    model_name=groq_model,
                )

            # Convert any API error into a visible dashboard message.
            except Exception as error:
                # Store the error message for display.
                ai_summary = f"AI summary failed: {error}"

        # Show the AI summary in the dashboard.
        st.success(ai_summary)

# If no signal exists, give a clear reason.
else:
    # Explain why no signal appears.
    st.info(
        "No row satisfied both conditions simultaneously. Lowering the Z-threshold may test the UI, but the stricter threshold is cleaner for real anomaly filtering."
    )


# =========================
# 7. Limitations
# =========================
st.header("7. Limitations")

# Put limitations in an expander so users can read them without cluttering the main dashboard.
with st.expander("Read limitations", expanded=True):
    # Explain practical limitations of this prototype.
    st.markdown(
        """
- This is a research and education dashboard, not an execution system.
- The strategy uses a simple short signal; it does not include order book depth, bid-ask spread, slippage, latency, borrow cost, or short-sale constraints.
- VWAP here is cumulative over the loaded sample, not necessarily the full trading day unless the input file contains the full day.
- Rolling standard deviation assumes recent history is a useful baseline; market volume has intraday seasonality and fat tails.
- A volume Z-score is an anomaly flag, not proof of a whale or institution.
- The dynamic exit target uses recent realized volatility; it does not guarantee fill probability.
- The GenAI risk brief is only a text summarizer and is deliberately separated from the deterministic signal engine.
"""
    )


# =========================
# 8. Download CSV
# =========================
st.header("8. Download CSV")

# Convert the full feature matrix into CSV bytes for download.
features_csv = df.to_csv(index=False).encode("utf-8")

# Convert the signal-only table into CSV bytes for download.
signals_csv = signal_df.to_csv(index=False).encode("utf-8")

# Convert the input summary into CSV bytes for reproducibility.
inputs_csv = input_summary.to_csv(index=False).encode("utf-8")

# Create download button columns.
download_cols = st.columns(3)

# Download full feature matrix.
with download_cols[0]:
    st.download_button(
        label="Download Full Feature Matrix",
        data=features_csv,
        file_name="spy_vwap_zscore_feature_matrix.csv",
        mime="text/csv",
    )

# Download only signal rows.
with download_cols[1]:
    st.download_button(
        label="Download Signal Events",
        data=signals_csv,
        file_name="spy_vwap_zscore_signal_events.csv",
        mime="text/csv",
    )

# Download inputs and assumptions.
with download_cols[2]:
    st.download_button(
        label="Download Input Summary",
        data=inputs_csv,
        file_name="spy_dashboard_input_summary.csv",
        mime="text/csv",
    )
