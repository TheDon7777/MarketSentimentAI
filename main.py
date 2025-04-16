# --- Main Application Entry Point ---
import streamlit as st
import pandas as pd
import plotly.graph_objects as go # Import Plotly
from plotly.subplots import make_subplots # For subplots
from typing import Any, Dict, List, Tuple, Optional
from streamlit_autorefresh import st_autorefresh # Import autorefresh
import traceback # Import traceback for error logging
import re # <--- Add re import

# Import functions from our modules (using absolute imports from src)
from src.config_manager import load_yaml_config, get_market_indices, get_stocks_to_track, get_config_parameter
from src.data_fetcher import fetch_current_data, fetch_historical_data, fetch_options_volume
from src.analysis_engine import (
    analyze_market_activity, # Main dashboard analysis
    calculate_rsi, calculate_macd, calculate_sma, calculate_bbands, calculate_obv, # Individual calcs for on-demand
    _add_status_columns # Helper for adding status columns
)
from src.news_fetcher import fetch_news_for_symbols
from src.sentiment_analyzer import initialize_google_ai, get_sentiment_for_news, analyze_symbol_sentiment_with_ta

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Market Sentiment AI", layout="wide")

# --- Initialize Session State --- #
if 'last_on_demand_ticker' not in st.session_state:
    st.session_state.last_on_demand_ticker = None
if 'last_on_demand_result' not in st.session_state:
    st.session_state.last_on_demand_result = None # Can store analysis dict or error string
if 'watchlist_tickers' not in st.session_state: # For watchlist section
    st.session_state.watchlist_tickers = []

# --- Helper Functions (Minor adjustments for Streamlit) ---

def format_structured_sentiment(analysis: dict) -> str:
    """Formats the structured sentiment analysis for display."""
    if not analysis or not isinstance(analysis, dict):
        return "*Analysis N/A*"

    sentiment = analysis.get('sentiment', 'Unknown')
    outlook = analysis.get('outlook', 'N/A')
    themes = analysis.get('themes', 'N/A')
    tech_signal = analysis.get('technical_signal', 'N/A')

    emoji = "❓"
    if sentiment == 'Positive': emoji = "🟢"
    elif sentiment == 'Negative': emoji = "🔴"
    elif sentiment == 'Neutral': emoji = "⚪"

    # Basic check for blocked/error cases
    if sentiment in ['Error', 'Blocked']:
        return f"*{emoji} Analysis Error/Blocked: ({outlook})*"

    # Return a more detailed breakdown for the on-demand section
    return (
        f"**{emoji} Sentiment:** {sentiment} | **Outlook:** {outlook}\n"
        f"> *News Themes:* {themes}\n"
        f"> *Technical Signal:* {tech_signal}"
    )

def get_current_value(df: Optional[pd.DataFrame], symbol: str, column: str, default='N/A') -> Any:
    """Safely retrieves a value from the current data DataFrame, handling potential issues."""
    if df is None or df.empty:
        # print(f"DEBUG: DataFrame empty for {symbol}/{column}") # Optional Debug
        return default
    try:
        if symbol in df.index:
            if column in df.columns:
                val = df.loc[symbol, column]
                # print(f"DEBUG: Raw value for {symbol}/{column}: {val} (type: {type(val)})") # Optional Debug
                # Explicitly check for Pandas NA, None, and potentially string 'N/A' before type checks
                if pd.isna(val) or val is None or str(val).strip() == 'N/A':
                    return default
                # Attempt conversion for numeric types, return default if fails
                if pd.api.types.is_numeric_dtype(df[column]) or column in ['Close', 'Price Change', 'Percent Change', 'Volume']:
                    try:
                        return float(val) # Convert to float for consistent numeric handling
                    except (ValueError, TypeError):
                        # print(f"DEBUG: Could not convert {val} to float for {symbol}/{column}") # Optional Debug
                        return default
                return val # Return as is for non-numeric types
            else:
                # print(f"DEBUG: Column '{column}' not found for {symbol}") # Optional Debug
                return default
        else:
            # print(f"DEBUG: Symbol '{symbol}' not found in DataFrame index") # Optional Debug
            return default
    except Exception as e:
        # print(f"DEBUG: Exception in get_current_value for {symbol}/{column}: {e}") # Optional Debug
        return default

# --- Plotting Function ---

@st.cache_data(ttl=900)
def get_plotting_data(symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Fetches sufficient historical data for plotting."""
    # Fetch more data for TA calculation stability within plot
    plot_hist = fetch_historical_data([symbol], days + 50)
    if plot_hist is not None and not plot_hist.empty:
        # Ensure index is datetime
        if isinstance(plot_hist.index, pd.MultiIndex):
            plot_hist.index = plot_hist.index.get_level_values('Date')
        return plot_hist.tail(days) # Return only the requested days
    return None

def create_stock_chart(symbol: str, hist_df: pd.DataFrame):
    """Creates an interactive Plotly chart with Price, Volume, RSI, MACD."""
    if hist_df is None or hist_df.empty:
        return go.Figure()

    # Calculate TA indicators using pandas_ta on the plotting data
    hist_df.ta.rsi(length=14, append=True)
    hist_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    # Column names from pandas_ta might vary slightly, adjust if needed
    rsi_col = next((col for col in hist_df.columns if 'RSI_14' in col), None)
    macd_col = next((col for col in hist_df.columns if 'MACD_12_26_9' in col), None)
    macd_hist_col = next((col for col in hist_df.columns if 'MACDh_12_26_9' in col), None)
    macd_signal_col = next((col for col in hist_df.columns if 'MACDs_12_26_9' in col), None)

    # Create figure with subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2]) # Adjust heights as needed

    # Candlestick Chart
    fig.add_trace(go.Candlestick(x=hist_df.index,
                                open=hist_df['Open'],
                                high=hist_df['High'],
                                low=hist_df['Low'],
                                close=hist_df['Close'],
                                name='Price'), row=1, col=1)

    # Volume Bars
    fig.add_trace(go.Bar(x=hist_df.index, y=hist_df['Volume'], name='Volume', marker_color='lightgrey'), row=1, col=1)
    # Position volume axis on the right (optional)
    # fig.update_layout(yaxis2=dict(title='Volume', overlaying='y', side='right'))

    # RSI Plot
    if rsi_col:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[rsi_col], name='RSI', line=dict(color='orange')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])

    # MACD Plot
    if macd_col and macd_signal_col and macd_hist_col:
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[macd_col], name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df[macd_signal_col], name='Signal', line=dict(color='red')), row=3, col=1)
        fig.add_trace(go.Bar(x=hist_df.index, y=hist_df[macd_hist_col], name='Histogram', marker_color='grey'), row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)

    # Update layout
    fig.update_layout(title=f'{symbol} Price & Indicators', xaxis_rangeslider_visible=False,
                      height=500, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(showticklabels=True, row=1, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    fig.update_xaxes(showticklabels=True, row=3, col=1)

    return fig

# --- New Plotting Helpers for Watchlist --- #
@st.cache_data(ttl=900) # Cache mini-chart data for 15 mins
def get_mini_plot_data(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetches minimal historical data for the mini watchlist chart."""
    plot_hist = fetch_historical_data([symbol], days + 5) # Fetch a bit extra
    if plot_hist is not None and not plot_hist.empty:
        if isinstance(plot_hist.index, pd.MultiIndex):
            plot_hist.index = plot_hist.index.get_level_values('Date')
        return plot_hist.tail(days)
    return None

def create_mini_chart(hist_df: Optional[pd.DataFrame]):
    """Creates a compact Plotly line chart for the watchlist."""
    if hist_df is None or hist_df.empty:
        # Return an empty figure or a placeholder message?
        fig = go.Figure()
        fig.update_layout(height=100, margin=dict(l=5, r=5, t=5, b=5),
                          xaxis={'visible': False}, yaxis={'visible': False},
                          annotations=[dict(text="No Data", xref="paper", yref="paper",
                                          showarrow=False, font=dict(size=10))])
        return fig

    fig = go.Figure()
    # Add sparkline trace
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], mode='lines', line=dict(width=1.5)))
    # Update layout for minimal appearance
    fig.update_layout(
        height=100, # Small height
        margin=dict(l=5, r=5, t=5, b=5), # Minimal margins
        showlegend=False,
        xaxis=dict(
            visible=False, # Hide x-axis labels/ticks
            showgrid=False,
            range=[hist_df.index.min(), hist_df.index.max()]
        ),
        yaxis=dict(
            visible=False, # Hide y-axis labels/ticks
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)' # Transparent background
    )
    return fig

# --- Data Fetching and Analysis Logic (Cached) ---

# Separate function for fetching just the latest quotes (not cached long)
@st.cache_data(ttl=55) # Cache live prices for just under a minute
def get_latest_quotes(symbols: List[str]) -> Optional[pd.DataFrame]:
    """Fetches only the latest quote data for specified symbols."""
    print(f"Fetching latest quotes for: {symbols} at {pd.Timestamp.now()}") # Log refresh
    if not symbols:
        return None
    try:
        # Use fetch_current_data as it's designed for this
        latest_data = fetch_current_data(symbols)
        return latest_data
    except Exception as e:
        st.error(f"Error fetching latest quotes: {e}")
        return None

# Cached function for the main, expensive analysis
@st.cache_data(ttl=900) # Cache analysis for 15 minutes
def load_analysis_data() -> Tuple[
    Dict[str, str],         # flagged_stocks
    Optional[pd.DataFrame], # combined_ta_data
    Optional[float],        # market_pcr
    Dict,                   # sentiment_analysis
    Optional[List[str]],    # indices
    Optional[List[str]]     # stocks
]:
    """Loads config, fetches data, runs TA and AI analysis FOR DASHBOARD."""
    print(f"Running full dashboard analysis at {pd.Timestamp.now()}")
    config = load_yaml_config()
    indices = get_market_indices()
    stocks = get_stocks_to_track()
    analysis_params = config.get('analysis_parameters', {})
    trading_lookback = analysis_params.get('trading_volume_lookback_days', 20)
    articles_per_symbol = analysis_params.get('news_articles_per_symbol', 5)
    # Get other TA params if needed directly
    if not indices and not stocks: st.error("Config error"); return {}, None, None, {}, None, None

    google_ai_model = initialize_google_ai()

    # Fetch data required for analysis_market_activity (needs MultiIndex hist, current for vol flag)
    all_symbols = list(set((indices or []) + (stocks or [])))
    if not all_symbols:
         st.warning("No symbols configured for analysis.")
         return {}, None, None, {}, None, None

    # Determine required history length (consider longest lookback needed by TA)
    # Max of SMA200, MACD(slow=26), BBands(20), RSI(14), Vol(20) -> Needs at least 200 + buffer
    required_hist_days = 250 # Sufficient for SMA200 etc.
    historical_stock_data = fetch_historical_data(all_symbols, required_hist_days)

    # Fetch temporary current data *just* for volume flagging & status calculation
    current_stock_data_for_analysis = fetch_current_data(all_symbols)
    if current_stock_data_for_analysis is None: current_stock_data_for_analysis = pd.DataFrame()

    # Fetch options data (only for tracked stocks, not indices)
    options_volume_data = []
    if stocks:
        # Use the already fetched current data to avoid another call
        stock_symbols_present = stocks
        if not current_stock_data_for_analysis.empty:
             stock_symbols_present = [s for s in stocks if s in current_stock_data_for_analysis.index.get_level_values('Ticker')]

        for symbol in stock_symbols_present:
             try: # Add try-except for options fetching
                 options_vol = fetch_options_volume(symbol)
                 if options_vol: options_volume_data.append(options_vol)
             except Exception as opt_e:
                  print(f"Warning: Failed to fetch options for {symbol}: {opt_e}")

    # --- Run Analysis (Now returns combined TA DataFrame) ---
    flagged_reasons, combined_ta_data, market_pcr = analyze_market_activity(
        current_stock_data=current_stock_data_for_analysis,
        historical_stock_data=historical_stock_data,
        options_volume_data=options_volume_data,
        config=config
    )

    # --- Fetch News and Run Sentiment (uses combined_ta_data) ---
    symbols_for_news = list(set(all_symbols))
    all_news = fetch_news_for_symbols(
        symbols=symbols_for_news,
        articles_per_symbol=articles_per_symbol
        )

    sentiment_analysis = get_sentiment_for_news(
        news_data=all_news,
        combined_ta_data=combined_ta_data,
        market_pcr=market_pcr
        )

    return flagged_reasons, combined_ta_data, market_pcr, sentiment_analysis, indices, stocks

# --- On-Demand Analysis Function (Updated) ---
def get_on_demand_analysis(symbol: str) -> Optional[Dict]:
    """Performs complete analysis for a single user-specified ticker."""
    if not symbol: st.warning("Please provide a ticker symbol."); return None
    symbol = symbol.upper().strip()
    print(f"Starting on-demand analysis for: {symbol} at {pd.Timestamp.now()}")

    try:
        # Fetch current data (needed for status cols & display)
        current_quote = fetch_current_data([symbol])
        if current_quote is None or current_quote.empty:
            st.error(f"Could not fetch current data for {symbol}. Is the ticker valid?")
            return None
        latest_close = get_current_value(current_quote, symbol, 'Close', default=None)

        # Fetch required history (long enough for all TAs)
        hist_data = fetch_historical_data([symbol], 250) # Needs OHLCV
        if hist_data is None or hist_data.empty:
            st.warning(f"Could not fetch sufficient historical data for {symbol} TA.")
            # Cannot proceed without historical data for TA
            return {"error": "Missing historical data for TA."} # Return error info
        else:
             # Ensure index is datetime for single stock
             if isinstance(hist_data.index, pd.MultiIndex):
                 hist_data.index = hist_data.index.get_level_values('Date')

        # --- Calculate All TA Indicators for Single Stock --- #
        config = load_yaml_config() # Get parameters
        analysis_params = config.get('analysis_parameters', {})
        rsi_window = analysis_params.get('rsi_window', 14)
        macd_fast = analysis_params.get('macd_fast', 12); macd_slow = analysis_params.get('macd_slow', 26); macd_signal = analysis_params.get('macd_signal', 9)
        sma_windows = analysis_params.get('sma_windows', [50, 200])
        bbands_length = analysis_params.get('bbands_length', 20); bbands_std = analysis_params.get('bbands_std', 2.0)

        ta_results = {}
        # Calculate individual TAs - they return full series/df for single stock
        rsi_full = calculate_rsi(hist_data, window=rsi_window)
        macd_full = calculate_macd(hist_data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        sma_full = calculate_sma(hist_data, windows=sma_windows)
        bbands_full = calculate_bbands(hist_data, length=bbands_length, std=bbands_std)
        obv_full = calculate_obv(hist_data)

        # Extract latest values
        ta_results['RSI'] = rsi_full.iloc[-1] if rsi_full is not None and not rsi_full.empty else None
        if macd_full is not None and not macd_full.empty:
            latest_macd = macd_full.iloc[-1]
            ta_results['MACD'] = latest_macd.get('MACD')
            ta_results['Signal'] = latest_macd.get('Signal')
            ta_results['Histogram'] = latest_macd.get('Histogram')
        if sma_full is not None and not sma_full.empty:
            latest_sma = sma_full.iloc[-1]
            for win in sma_windows: ta_results[f'SMA_{win}'] = latest_sma.get(f'SMA_{win}')
        if bbands_full is not None and not bbands_full.empty:
            latest_bbands = bbands_full.iloc[-1]
            ta_results['BB_Lower'] = latest_bbands.get('BB_Lower')
            ta_results['BB_Middle'] = latest_bbands.get('BB_Middle')
            ta_results['BB_Upper'] = latest_bbands.get('BB_Upper')
        if obv_full is not None and not obv_full.empty:
            ta_results['OBV'] = obv_full['OBV'].iloc[-1]
            # Calculate OBV trend manually for single stock
            def get_simple_trend(series, n=5):
                 if series is None or len(series) < n: return "Flat"
                 series = series.dropna();
                 if len(series) < 2: return "Flat"
                 last_val = series.iloc[-1]; prev_val = series.iloc[-min(n, len(series))] # Compare last to N periods ago
                 if pd.isna(last_val) or pd.isna(prev_val): return "Flat"
                 if last_val > prev_val: return "Rising"
                 if last_val < prev_val: return "Falling"
                 return "Flat"
            ta_results['OBV_Trend'] = get_simple_trend(obv_full['OBV'])

        # Assemble into a Series, add current Close for status calculation
        ta_data_for_symbol = pd.Series(ta_results)
        ta_data_for_symbol['Close'] = latest_close

        # Add status columns using the helper (convert Series to DataFrame temporarily)
        temp_df = pd.DataFrame([ta_data_for_symbol]) # Create 1-row DF
        temp_df_with_status = _add_status_columns(temp_df)
        ta_data_for_symbol = temp_df_with_status.iloc[0] # Convert back to Series

        # --- Fetch News --- #
        articles_per_symbol = analysis_params.get('news_articles_per_symbol', 5)
        news_data = fetch_news_for_symbols([symbol], articles_per_symbol=articles_per_symbol)
        # ... (debug prints for news) ...
        print(f"DEBUG: Fetched news_data for {symbol}: {news_data}")

        # --- Analyze Sentiment --- #
        headlines = [a.get('title', '') for a in news_data.get(symbol, []) if a and a.get('title')]
        # ... (debug prints for headlines) ...
        print(f"DEBUG: Extracted headlines for {symbol}: {headlines}")

        sentiment_result = analyze_symbol_sentiment_with_ta(
            symbol=symbol,
            headlines=headlines,
            ta_data=ta_data_for_symbol, # Pass the complete TA Series
            market_pcr=None # Market PCR not calculated for single stock
        )

        # --- Compile Results --- #
        analysis_output = {
            "current_quote": current_quote, # Keep original quote data
            "ta_data": ta_data_for_symbol, # Include the consolidated TA series
            "sentiment_analysis": sentiment_result
        }
        print(f"Completed on-demand analysis for: {symbol}")
        return analysis_output

    except Exception as e:
        st.error(f"An error occurred during the analysis for {symbol}: {e}")
        print(f"Error during on-demand analysis for {symbol}:")
        traceback.print_exc()
        return {"error": str(e)}

# --- Streamlit UI ---

st.title("📈 Market Sentiment AI Analyzer")
st.markdown("Provides insights based on market data, news sentiment (via Google Gemini), and technical indicators.")

# Add the autorefresh component
refresh_interval_seconds = 60
st_autorefresh(interval=refresh_interval_seconds * 1000, key="data_refresher")

st.divider()

# Load the expensive, cached analysis data
# This runs only once per cache TTL or if inputs change (currently none)
flagged_reasons, combined_ta_data, market_pcr, sentiment_analysis, indices, stocks = load_analysis_data()

# Determine all symbols needed for LIVE quotes (dashboard + watchlist + last on-demand ticker)
all_symbols_to_quote = list(set((indices or []) + (stocks or [])))
if st.session_state.watchlist_tickers: # Add watchlist symbols
    all_symbols_to_quote.extend(st.session_state.watchlist_tickers)
if st.session_state.last_on_demand_ticker: # Add last on-demand symbol
    all_symbols_to_quote.append(st.session_state.last_on_demand_ticker)
all_symbols_to_quote = list(set(all_symbols_to_quote)) # Ensure uniqueness

# Fetch the latest quotes for ALL required symbols
latest_quotes = get_latest_quotes(all_symbols_to_quote)

# Display Market-wide PCR if calculated
st.header("Market Condition Proxy")
if market_pcr is not None:
    pcr_interpretation = "Neutral" 
    if market_pcr > 0.7: # Example thresholds, can be refined
        pcr_interpretation = "Potentially Bearish / Fearful"
    elif market_pcr < 0.5:
         pcr_interpretation = "Potentially Bullish / Greedy"
    st.metric("Market Put/Call Ratio (Tracked Stocks)", f"{market_pcr:.3f}", delta=pcr_interpretation, delta_color="off")
else:
    st.info("Market Put/Call Ratio could not be calculated (likely missing options data).")

st.header("📊 Broad Market Indices")
if not indices:
    st.info("No indices configured.")
elif latest_quotes.empty:
    st.warning("Could not retrieve current market data for indices.")
else:
    cols = st.columns(len(indices))
    for i, index_sym in enumerate(indices):
        with cols[i]:
            # Use LATEST quote data for price/change
            price = get_current_value(latest_quotes, index_sym, 'Close')
            change = get_current_value(latest_quotes, index_sym, 'Price Change')
            pct_change = get_current_value(latest_quotes, index_sym, 'Percent Change')
            # Use CACHED sentiment data
            sentiment_data = sentiment_analysis.get(index_sym, {})

            st.metric(
                label=index_sym,
                value=f"{price:.2f}" if isinstance(price, (int, float)) else "N/A",
                delta=f"{change:.2f} ({pct_change:.2f}%) " if isinstance(change, (int, float)) and isinstance(pct_change, (int, float)) else "Data Missing",
            )
            st.markdown(format_structured_sentiment(sentiment_data))

st.divider()

st.header("🔥 Stocks with Notable Activity")

if not flagged_reasons:
    st.info("No stocks met the criteria for notable activity based on current configuration.")
elif combined_ta_data is None:
     st.warning("TA Data is unavailable, cannot display detailed stock activity.")
else:
    for symbol, reason in flagged_reasons.items():
        st.subheader(f"{symbol} ({reason})")
        col1, col2 = st.columns([0.4, 0.6])

        with col1:
            # --- Data Extraction --- #
            price = get_current_value(latest_quotes, symbol, 'Close')
            change = get_current_value(latest_quotes, symbol, 'Price Change')
            pct_change = get_current_value(latest_quotes, symbol, 'Percent Change')
            volume = get_current_value(latest_quotes, symbol, 'Volume')

            ta_series = combined_ta_data.loc[symbol] if symbol in combined_ta_data.index else pd.Series()
            sentiment_data = sentiment_analysis.get(symbol, {})

            # --- Basic Metrics --- #
            st.metric("Price", f"{price:.2f}" if isinstance(price, (int, float)) else "N/A",
                      f"{change:+.2f} ({pct_change:.2f}%) " if isinstance(change, (int, float)) and isinstance(pct_change, (int, float)) else "Data Missing")
            st.text(f"Volume: {int(volume):,}" if isinstance(volume, (int, float)) else "Volume: N/A")

            # --- Technical Indicators --- #
            st.markdown("**Technicals:**")
            rsi_val = ta_series.get('RSI')
            st.text(f"- RSI(14): {rsi_val:.1f}" if pd.notna(rsi_val) else "- RSI(14): N/A")
            # MACD Status
            macd_line = ta_series.get('MACD'); macd_signal = ta_series.get('Signal'); macd_hist = ta_series.get('Histogram')
            macd_status = "N/A"
            if pd.notna(macd_line) and pd.notna(macd_signal) and pd.notna(macd_hist):
                 if macd_line > macd_signal and macd_hist > 0: macd_status = f"Bullish (H:{macd_hist:+.2f})"
                 elif macd_line < macd_signal and macd_hist < 0: macd_status = f"Bearish (H:{macd_hist:+.2f})"
                 else: macd_status = f"Neutral/Cross (H:{macd_hist:+.2f})"
            elif pd.notna(macd_line): macd_status = f"Line: {macd_line:.2f}"
            st.text(f"- MACD: {macd_status}")
            # New Indicators
            st.text(f"- Price vs SMA50: {ta_series.get('Price_vs_SMA50', 'N/A')}")
            st.text(f"- SMA Trend: {ta_series.get('SMA50_vs_SMA200', 'N/A')}")
            st.text(f"- BBand Status: {ta_series.get('BB_Status', 'N/A')}")
            st.text(f"- OBV Trend: {ta_series.get('OBV_Trend', 'N/A')}")

            # --- Sentiments & Assessment --- #
            st.markdown("**Sentiments & Assessment:**")
            tech_sentiment = ta_series.get('Technical_Sentiment', 'N/A')
            news_sentiment = sentiment_data.get('sentiment', 'N/A')
            ai_tech_summary = sentiment_data.get('technical_signal', 'N/A')
            overall_assessment = sentiment_data.get('outlook', 'N/A')
            news_themes = sentiment_data.get('themes', 'N/A')

            st.markdown(f"- **Technical Sentiment (Rule-Based):** {tech_sentiment}")
            st.markdown(f"- **News Sentiment (AI):** {news_sentiment}")
            st.markdown(f"- **AI Tech Summary:** {ai_tech_summary}")
            st.markdown(f"- **Overall Assessment (AI):** {overall_assessment}")
            st.markdown(f"- **News Themes:** {news_themes}")

        with col2:
            # Charting remains the same
            plot_data = get_plotting_data(symbol, days=90)
            if plot_data is not None and not plot_data.empty: fig = create_stock_chart(symbol, plot_data); st.plotly_chart(fig, use_container_width=True)
            else: st.warning(f"Could not retrieve sufficient historical data for {symbol} chart.")
        st.divider()

st.divider()

# --- Session Watchlist Section --- #
st.divider()
st.header("📊 Session Watchlist (Max 16)")

# Input area for adding tickers to the watchlist
col_add, col_spacer = st.columns([0.8, 0.2])
with col_add:
    new_watchlist_tickers_str = st.text_input(
        "Add tickers to watchlist (comma or space separated):",
        key="watchlist_input"
    )
    if st.button("Add to Watchlist", key="add_watchlist_button"):
        if new_watchlist_tickers_str:
            # Parse input string: split by comma or space, strip whitespace, uppercase, unique
            parsed_tickers = [t.strip().upper() for t in re.split(r'[\s,]+', new_watchlist_tickers_str) if t.strip()]
            added_count = 0
            current_watchlist = st.session_state.watchlist_tickers
            for ticker in parsed_tickers:
                if ticker and ticker not in current_watchlist:
                    # Limit watchlist size
                    if len(current_watchlist) < 16:
                         current_watchlist.append(ticker)
                         added_count += 1
                    else:
                         st.warning("Watchlist limit (16) reached.")
                         break # Stop adding if limit reached
            if added_count > 0:
                st.session_state.watchlist_tickers = current_watchlist # Update session state
                st.rerun()
            elif parsed_tickers: # User entered something, but nothing was added
                 st.toast("Ticker(s) already in watchlist or invalid input.")
            else:
                 st.toast("Please enter ticker symbols.")

# Display the watchlist grid
if st.session_state.watchlist_tickers:
    num_tickers = len(st.session_state.watchlist_tickers)
    cols = 4 # Define number of columns
    # Calculate rows needed
    num_rows = (num_tickers + cols - 1) // cols

    ticker_index = 0
    for r in range(num_rows):
        columns = st.columns(cols)
        for c in range(cols):
            if ticker_index < num_tickers:
                symbol = st.session_state.watchlist_tickers[ticker_index]
                with columns[c]:
                    st.subheader(symbol)

                    # Button to remove the ticker
                    if st.button(f"Remove {symbol}", key=f"remove_{symbol}_{ticker_index}", help="Remove from watchlist"):
                        st.session_state.watchlist_tickers.pop(ticker_index)
                        st.rerun() # Rerun immediately after removal
                        # Important: Need to handle index shift after removal, rerun is safest

                    # Display live price metric
                    wl_price = get_current_value(latest_quotes, symbol, 'Close')
                    wl_change = get_current_value(latest_quotes, symbol, 'Price Change')
                    st.metric(
                        label="Price",
                        value=f"{wl_price:.2f}" if isinstance(wl_price, (int, float)) else "N/A",
                        delta=f"{wl_change:.2f}" if isinstance(wl_change, (int, float)) else None,
                    )

                    # Display mini-chart
                    mini_plot_data = get_mini_plot_data(symbol, days=30)
                    mini_fig = create_mini_chart(mini_plot_data)
                    st.plotly_chart(mini_fig, use_container_width=True)

                ticker_index += 1
            else:
                # Leave remaining columns empty if fewer than 4 tickers in the last row
                pass
    st.caption("Watchlist data persists only for the current browser session.")
else:
    st.info("Add tickers using the input box above to start your watchlist.")

# --- On-Demand Analysis Section ---
st.divider()
st.header("🔍 On-Demand Stock Analysis")

# Input field always visible
on_demand_ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", key="on_demand_ticker_input").upper()

# Button click logic - only calculates if ticker changes
if st.button("Analyze Ticker", key="analyze_button"):
    if on_demand_ticker_input:
        # Check if it's a new ticker compared to the last successful/attempted one
        if on_demand_ticker_input != st.session_state.get('last_on_demand_ticker'):
            with st.spinner(f"Performing full analysis for {on_demand_ticker_input}..."):
                # Run the full analysis
                analysis_result = get_on_demand_analysis(on_demand_ticker_input)
                # Store result (could be analysis dict or error dict) and ticker in session state
                st.session_state.last_on_demand_ticker = on_demand_ticker_input
                st.session_state.last_on_demand_result = analysis_result
                st.rerun() # Rerun to update display based on new session state
        else:
             # Same ticker, do nothing, analysis is already stored
             st.toast(f"Analysis for {on_demand_ticker_input} already displayed.")
    else:
        st.warning("Please enter a ticker symbol.")
        # Clear previous results if button clicked with empty input
        st.session_state.last_on_demand_ticker = None
        st.session_state.last_on_demand_result = None

# --- Display On-Demand Results (Reads from Session State) --- #
if st.session_state.last_on_demand_ticker and st.session_state.last_on_demand_result:
    # Retrieve stored data
    display_ticker = st.session_state.last_on_demand_ticker
    analysis_result = st.session_state.last_on_demand_result

    # Check if stored result is an error
    if isinstance(analysis_result, dict) and "error" in analysis_result:
        st.error(f"Analysis failed for {display_ticker}: {analysis_result['error']}")
    elif isinstance(analysis_result, dict): # Check if it's the expected analysis dict
        st.subheader(f"Analysis for {display_ticker} (Price updates live)")
        od_col1, od_col2 = st.columns([0.4, 0.6])

        with od_col1:
            # --- Data Extraction --- #
            # Get STALE quote data from stored result (needed ONLY if live quotes fail)
            # od_quote_stale = analysis_result.get("current_quote")
            od_ta_data = analysis_result.get("ta_data") # Use stored TA data
            od_sentiment = analysis_result.get("sentiment_analysis", {}) # Use stored sentiment

            # --- Basic Metrics (Use LIVE quotes) --- #
            od_price = get_current_value(latest_quotes, display_ticker, 'Close')
            od_change = get_current_value(latest_quotes, display_ticker, 'Price Change')
            od_pct_change = get_current_value(latest_quotes, display_ticker, 'Percent Change')
            od_volume = get_current_value(latest_quotes, display_ticker, 'Volume')

            st.metric("Price", f"{od_price:.2f}" if isinstance(od_price, (int, float)) else "N/A",
                      f"{od_change:+.2f} ({od_pct_change:.2f}%) " if isinstance(od_change, (int, float)) and isinstance(od_pct_change, (int, float)) else "Data Missing")
            st.text(f"Volume: {int(od_volume):,}" if isinstance(od_volume, (int, float)) else "Volume: N/A")

            # --- Technical Indicators (Use STORED TA data) --- #
            st.markdown("**Technicals:**")
            if od_ta_data is not None:
                od_rsi = od_ta_data.get('RSI'); st.text(f"- RSI(14): {od_rsi:.1f}" if pd.notna(od_rsi) else "- RSI(14): N/A")
                od_macd_line = od_ta_data.get('MACD'); od_macd_signal = od_ta_data.get('Signal'); od_macd_hist = od_ta_data.get('Histogram');
                od_macd_status = "N/A"
                if pd.notna(od_macd_line) and pd.notna(od_macd_signal) and pd.notna(od_macd_hist):
                     if od_macd_line > od_macd_signal and od_macd_hist > 0: od_macd_status = f"Bullish (H:{od_macd_hist:+.2f})"
                     elif od_macd_line < od_macd_signal and od_macd_hist < 0: od_macd_status = f"Bearish (H:{od_macd_hist:+.2f})"
                     else: od_macd_status = f"Neutral/Cross (H:{od_macd_hist:+.2f})"
                elif pd.notna(od_macd_line): od_macd_status = f"Line: {od_macd_line:.2f}"
                st.text(f"- MACD: {od_macd_status}")
                st.text(f"- Price vs SMA50: {od_ta_data.get('Price_vs_SMA50', 'N/A')}")
                st.text(f"- SMA Trend: {od_ta_data.get('SMA50_vs_SMA200', 'N/A')}")
                st.text(f"- BBand Status: {od_ta_data.get('BB_Status', 'N/A')}")
                st.text(f"- OBV Trend: {od_ta_data.get('OBV_Trend', 'N/A')}")
            else:
                st.text("(TA Data not available)")

            # --- Sentiments & Assessment (Use STORED sentiment data) --- #
            st.markdown("**Sentiments & Assessment:**")
            tech_sentiment = od_ta_data.get('Technical_Sentiment', 'N/A') if od_ta_data is not None else 'N/A'
            news_sentiment = od_sentiment.get('sentiment', 'N/A')
            ai_tech_summary = od_sentiment.get('technical_signal', 'N/A')
            overall_assessment = od_sentiment.get('outlook', 'N/A')
            news_themes = od_sentiment.get('themes', 'N/A')

            st.markdown(f"- **Technical Sentiment (Rule-Based):** {tech_sentiment}")
            st.markdown(f"- **News Sentiment (AI):** {news_sentiment}")
            st.markdown(f"- **AI Tech Summary:** {ai_tech_summary}")
            st.markdown(f"- **Overall Assessment (AI):** {overall_assessment}")
            st.markdown(f"- **News Themes:** {news_themes}")

        with od_col2:
            # Charting uses cached historical data, still relevant
            od_plot_data = get_plotting_data(display_ticker, days=90)
            if od_plot_data is not None and not od_plot_data.empty:
                od_fig = create_stock_chart(display_ticker, od_plot_data)
                st.plotly_chart(od_fig, use_container_width=True)
            else:
                st.warning(f"Could not retrieve sufficient historical data for {display_ticker} chart.")

# --- Footer --- #
st.divider()
st.markdown(f"_Dashboard prices updated approx. every {refresh_interval_seconds}s. Full analysis cached for 15 mins._") 