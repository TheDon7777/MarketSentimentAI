import pandas as pd
import pandas_ta as ta # Import pandas-ta
from typing import List, Dict, Tuple, Optional
import numpy as np # For potential calculations

# --- Technical Indicator Calculation Helpers ---

def calculate_rsi(historical_data: pd.DataFrame, window: int = 14) -> Optional[pd.Series]:
    """
    Calculates the Relative Strength Index (RSI).
    Handles MultiIndex and DateTimeIndex.
    Returns Series (or None).
    """
    if historical_data.empty or 'Close' not in historical_data.columns:
        print("Warning: RSI - Missing data or 'Close' column.")
        return None
    try:
        if isinstance(historical_data.index, pd.MultiIndex) and 'Ticker' in historical_data.index.names:
            rsi_series = historical_data.groupby(level='Ticker')['Close'].apply(
                lambda x: ta.rsi(x, length=window).iloc[-1] if len(x) > window else np.nan
            ).rename('RSI')
            return rsi_series.round(2)
        elif isinstance(historical_data.index, pd.DatetimeIndex):
            rsi = ta.rsi(historical_data['Close'], length=window)
            return rsi.round(2).rename('RSI') # Return full Series
        else:
            print("Warning: RSI - Unhandled index type.")
            return None
    except Exception as e:
        print(f"Error calculating RSI: {e}"); traceback.print_exc()
        return None

def calculate_macd(historical_data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[pd.DataFrame]:
    """
    Calculates MACD, Signal, Histogram.
    Handles MultiIndex and DateTimeIndex.
    Returns DataFrame (or None).
    """
    required_columns = {'Close'} # Only need Close for MACD calc
    if historical_data.empty or not required_columns.issubset(historical_data.columns):
        print("Warning: MACD - Missing data or 'Close' column.")
        return None
    try:
        # Calculate directly, handles DateTimeIndex
        historical_data.ta.macd(fast=fast, slow=slow, signal=signal, append=True)
        macd_col = f'MACD_{fast}_{slow}_{signal}'; sig_col = f'MACDs_{fast}_{slow}_{signal}'; hist_col = f'MACDh_{fast}_{slow}_{signal}'
        macd_cols = [macd_col, sig_col, hist_col]
        if not all(col in historical_data.columns for col in macd_cols):
            print(f"Warning: MACD columns not found after calculation.")
            return None
        macd_results = historical_data[macd_cols].rename(columns={macd_col: 'MACD', sig_col: 'Signal', hist_col: 'Histogram'})
        if isinstance(historical_data.index, pd.MultiIndex) and 'Ticker' in historical_data.index.names:
            # Group multi-index result to get latest per ticker
            return macd_results.groupby(level='Ticker').last().round(3)
        elif isinstance(historical_data.index, pd.DatetimeIndex):
            # Return the full DF for single stock, caller takes last row
            return macd_results.round(3)
        else:
            print("Warning: MACD - Unhandled index type.")
            return None
    except Exception as e:
        print(f"Error calculating MACD: {e}"); import traceback; traceback.print_exc()
        return None

def calculate_sma(historical_data: pd.DataFrame, windows: List[int] = [50, 200]) -> Optional[pd.DataFrame]:
    """
    Calculates Simple Moving Averages (SMA) for specified windows.
    Handles MultiIndex and DateTimeIndex.
    Returns DataFrame (or None).
    """
    if historical_data.empty or 'Close' not in historical_data.columns:
        print("Warning: SMA - Missing data or 'Close' column.")
        return None
    try:
        results = {} # Store results for each window
        if isinstance(historical_data.index, pd.MultiIndex) and 'Ticker' in historical_data.index.names:
            # Multi-stock: Group by ticker, calculate SMA, get last value
            for window in windows:
                sma_col = f'SMA_{window}'
                results[sma_col] = historical_data.groupby(level='Ticker')['Close'].apply(
                    lambda x: ta.sma(x, length=window).iloc[-1] if len(x) >= window else np.nan
                )
            sma_df = pd.DataFrame(results)
            return sma_df.round(2)
        elif isinstance(historical_data.index, pd.DatetimeIndex):
            # Single-stock: Calculate SMA directly
            sma_df = pd.DataFrame(index=historical_data.index)
            for window in windows:
                 sma_df[f'SMA_{window}'] = ta.sma(historical_data['Close'], length=window)
            return sma_df.round(2) # Return full DF, caller takes last row
        else:
            print("Warning: SMA - Unhandled index type.")
            return None
    except Exception as e:
        print(f"Error calculating SMA: {e}"); import traceback; traceback.print_exc()
        return None

def calculate_bbands(historical_data: pd.DataFrame, length: int = 20, std: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Calculates Bollinger Bands (Lower, Middle, Upper).
    Handles MultiIndex and DateTimeIndex.
    Returns DataFrame (or None).
    """
    if historical_data.empty or 'Close' not in historical_data.columns:
        print("Warning: BBands - Missing data or 'Close' column.")
        return None
    try:
        # Standard pandas_ta column names for BBands
        bbl_col = f'BBL_{length}_{std}'; bbm_col = f'BBM_{length}_{std}'; bbu_col = f'BBU_{length}_{std}'
        bband_cols = [bbl_col, bbm_col, bbu_col]
        # Calculate directly
        historical_data.ta.bbands(length=length, std=std, append=True)
        if not all(col in historical_data.columns for col in bband_cols):
            print(f"Warning: BBands columns not found after calculation.")
            return None
        # Rename columns
        bb_results = historical_data[bband_cols].rename(columns={bbl_col: 'BB_Lower', bbm_col: 'BB_Middle', bbu_col: 'BB_Upper'})

        if isinstance(historical_data.index, pd.MultiIndex) and 'Ticker' in historical_data.index.names:
            return bb_results.groupby(level='Ticker').last().round(2)
        elif isinstance(historical_data.index, pd.DatetimeIndex):
            return bb_results.round(2) # Return full DF, caller takes last row
        else:
            print("Warning: BBands - Unhandled index type.")
            return None
    except Exception as e:
        print(f"Error calculating BBands: {e}"); import traceback; traceback.print_exc()
        return None

def calculate_obv(historical_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calculates On-Balance Volume (OBV) and its simple trend.
    Handles MultiIndex and DateTimeIndex.
    Returns DataFrame (or None).
    """
    if historical_data.empty or 'Close' not in historical_data.columns or 'Volume' not in historical_data.columns:
        print("Warning: OBV - Missing data, 'Close' or 'Volume' column.")
        return None
    try:
        obv_col = 'OBV' # pandas_ta default OBV column name
        # Calculate OBV directly
        historical_data.ta.obv(append=True)
        if obv_col not in historical_data.columns:
            print(f"Warning: OBV column not found after calculation.")
            return None

        # Define a simple trend function (compare last value to mean of last N)
        def get_simple_trend(series, n=5):
            if series is None or len(series) < n: return "Flat"
            series = series.dropna()
            if len(series) < 2: return "Flat"
            last_val = series.iloc[-1]
            # Use rolling mean excluding last point? Or simple diff?
            # Let's use diff of last point vs N period ago
            prev_val = series.iloc[-min(n, len(series))] # Compare last to N periods ago
            if pd.isna(last_val) or pd.isna(prev_val): return "Flat"
            if last_val > prev_val: return "Rising"
            if last_val < prev_val: return "Falling"
            return "Flat"

        if isinstance(historical_data.index, pd.MultiIndex) and 'Ticker' in historical_data.index.names:
            # Multi-stock: Group by ticker, calculate trend, get last OBV value
            obv_trend = historical_data.groupby(level='Ticker')[obv_col].apply(get_simple_trend).rename('OBV_Trend')
            latest_obv = historical_data.groupby(level='Ticker')[obv_col].last().rename('OBV')
            result_df = pd.DataFrame({'OBV': latest_obv, 'OBV_Trend': obv_trend})
            return result_df
        elif isinstance(historical_data.index, pd.DatetimeIndex):
            # Single-stock: Return full OBV series, caller calculates trend & takes last
            obv_df = pd.DataFrame({obv_col: historical_data[obv_col]})
            return obv_df # Return full DF
        else:
            print("Warning: OBV - Unhandled index type.")
            return None
    except Exception as e:
        print(f"Error calculating OBV: {e}"); import traceback; traceback.print_exc()
        return None

# --- Rule-Based Technical Sentiment --- #
def calculate_technical_sentiment(ta_series: pd.Series) -> str:
    """Assigns a simple sentiment based on a combination of TA indicators."""
    if ta_series is None or ta_series.empty:
        return "Neutral" # Default if no data

    score = 0

    # RSI
    rsi = ta_series.get('RSI')
    if pd.notna(rsi):
        if rsi > 55: score += 1
        elif rsi < 45: score -= 1
        if rsi > 70: score += 1 # Extra weight for overbought/oversold
        elif rsi < 30: score -= 1

    # MACD
    macd_line = ta_series.get('MACD')
    macd_signal = ta_series.get('Signal')
    macd_hist = ta_series.get('Histogram')
    if pd.notna(macd_line) and pd.notna(macd_signal) and pd.notna(macd_hist):
        if macd_line > macd_signal and macd_hist > 0: score += 2 # Bullish crossover
        elif macd_line < macd_signal and macd_hist < 0: score -= 2 # Bearish crossover
        elif macd_line > macd_signal: score += 1 # Above signal line
        elif macd_line < macd_signal: score -= 1 # Below signal line

    # Price vs SMAs
    price_vs_sma50 = ta_series.get('Price_vs_SMA50')
    price_vs_sma200 = ta_series.get('Price_vs_SMA200')
    if price_vs_sma50 == 'Above': score += 1
    elif price_vs_sma50 == 'Below': score -= 1
    if price_vs_sma200 == 'Above': score += 1 # Longer term trend
    elif price_vs_sma200 == 'Below': score -= 1

    # SMA Trend (Golden/Death Cross)
    sma50_vs_sma200 = ta_series.get('SMA50_vs_SMA200')
    if sma50_vs_sma200 is not None:
        if 'Golden Cross' in sma50_vs_sma200: score += 2
        elif 'Death Cross' in sma50_vs_sma200: score -= 2

    # Bollinger Bands
    bb_status = ta_series.get('BB_Status')
    if bb_status == 'Near Upper': score -= 0.5 # Slight negative for potential reversal
    elif bb_status == 'Near Lower': score += 0.5 # Slight positive for potential bounce

    # OBV Trend
    obv_trend = ta_series.get('OBV_Trend')
    if obv_trend == 'Rising': score += 1
    elif obv_trend == 'Falling': score -= 1

    # Determine final sentiment label
    if score >= 3: return "Positive"
    if score <= -3: return "Negative"
    return "Neutral"

def _add_status_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    """Adds status columns based on existing TA columns AND technical sentiment."""
    # Work on a copy to avoid modifying the original DataFrame view passed in
    df = df_input.copy()

    # Price vs SMA
    if 'Close' in df.columns and 'SMA_50' in df.columns:
        df.loc[df['Close'] > df['SMA_50'], 'Price_vs_SMA50'] = 'Above'
        df.loc[df['Close'] <= df['SMA_50'], 'Price_vs_SMA50'] = 'Below'
    if 'Close' in df.columns and 'SMA_200' in df.columns:
        df.loc[df['Close'] > df['SMA_200'], 'Price_vs_SMA200'] = 'Above'
        df.loc[df['Close'] <= df['SMA_200'], 'Price_vs_SMA200'] = 'Below'

    # SMA50 vs SMA200 (Golden/Death Cross proxy)
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        df.loc[df['SMA_50'] > df['SMA_200'], 'SMA50_vs_SMA200'] = 'Golden Cross Signal'
        df.loc[df['SMA_50'] <= df['SMA_200'], 'SMA50_vs_SMA200'] = 'Death Cross Signal'

    # Bollinger Bands Status
    # Use the known renamed columns directly
    bbl_col_name = 'BB_Lower'
    bbu_col_name = 'BB_Upper'
    if bbl_col_name in df.columns and bbu_col_name in df.columns and 'Close' in df.columns:
        # Ensure columns are numeric before comparison, handle potential NaNs
        df.loc[pd.to_numeric(df['Close'], errors='coerce') <= pd.to_numeric(df[bbl_col_name], errors='coerce'), 'BB_Status'] = 'Testing Lower Band'
        df.loc[pd.to_numeric(df['Close'], errors='coerce') >= pd.to_numeric(df[bbu_col_name], errors='coerce'), 'BB_Status'] = 'Testing Upper Band'
        df.loc[(pd.to_numeric(df['Close'], errors='coerce') > pd.to_numeric(df[bbl_col_name], errors='coerce')) & 
               (pd.to_numeric(df['Close'], errors='coerce') < pd.to_numeric(df[bbu_col_name], errors='coerce')), 'BB_Status'] = 'Within Bands'
        # Handle cases where comparison couldn't be made (e.g., NaNs in BBands or Close)
        df['BB_Status'].fillna('N/A', inplace=True)
    else:
        print(f"Warning: Missing required columns for BB_Status calculation (need Close, {bbl_col_name}, {bbu_col_name})")
        df.loc[:, 'BB_Status'] = 'N/A' # Assign default if columns not found

    # MACD Status (using histogram)
    macd_hist_col = next((col for col in df.columns if col.startswith('MACDh_')), None)
    macd_line_col = next((col for col in df.columns if col.startswith('MACD_') and not col.startswith('MACDh') and not col.startswith('MACDs')), None)
    macd_signal_col = next((col for col in df.columns if col.startswith('MACDs_')), None)

    if macd_hist_col and macd_line_col and macd_signal_col:
        df.loc[df[macd_hist_col] > 0, 'MACD_Status'] = 'Histogram Positive'
        df.loc[df[macd_hist_col] <= 0, 'MACD_Status'] = 'Histogram Negative'
        # Add crossover status if helpful
        df.loc[(df[macd_line_col] > df[macd_signal_col]), 'MACD_Crossover'] = 'Bullish Crossover Likely'
        df.loc[(df[macd_line_col] <= df[macd_signal_col]), 'MACD_Crossover'] = 'Bearish Crossover Likely'
    else:
        df.loc[:, 'MACD_Status'] = 'N/A'
        df.loc[:, 'MACD_Crossover'] = 'N/A'

    # Add Technical Sentiment
    df = _add_technical_sentiment(df) # Call the helper

    return df

def _add_technical_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a rule-based technical sentiment score based on available indicators."""

    # --- Identify expected TA column names --- #
    # Note: Adjust these based on the actual parameters used in run_analysis_batch
    rsi_col = next((col for col in df.columns if col.startswith('RSI_')), None)
    macd_status_col = 'MACD_Status' # This is calculated in _add_status_columns
    price_vs_sma50_col = 'Price_vs_SMA50' # Calculated in _add_status_columns
    sma50_vs_sma200_col = 'SMA50_vs_SMA200' # Calculated in _add_status_columns
    bb_status_col = 'BB_Status' # Calculated in _add_status_columns
    obv_trend_col = 'OBV_Trend' # Calculated in _add_status_columns

    # Check if the core status columns (which rely on base TA) exist
    required_status_cols = [macd_status_col, price_vs_sma50_col, sma50_vs_sma200_col, bb_status_col, obv_trend_col]
    if not all(col in df.columns for col in required_status_cols):
         print(f"Warning: Skipping Technical Sentiment calculation due to missing base status columns: {[col for col in required_status_cols if col not in df.columns]}")
         df.loc[:, 'Technical_Sentiment'] = 'N/A'
         return df

    # --- Define conditions using the identified column names --- #
    conditions = {
        'bullish': [
            (lambda x: x.get(price_vs_sma50_col, '') == 'Above'),
            (lambda x: x.get(sma50_vs_sma200_col, '') == 'Golden Cross Signal'),
            (lambda x: x.get(macd_status_col, '') == 'Histogram Positive'),
            (lambda x: x.get(obv_trend_col, '') == 'Rising'),
            # Optional RSI condition (check if rsi_col exists)
            (lambda x: rsi_col and x.get(rsi_col, 50) < 70 and x.get(rsi_col, 50) > 50), # RSI bullish but not overbought
            # Optional BB condition
            (lambda x: x.get(bb_status_col, '') != 'Testing Upper Band')
        ],
        'bearish': [
            (lambda x: x.get(price_vs_sma50_col, '') == 'Below'),
            (lambda x: x.get(sma50_vs_sma200_col, '') == 'Death Cross Signal'),
            (lambda x: x.get(macd_status_col, '') == 'Histogram Negative'),
            (lambda x: x.get(obv_trend_col, '') == 'Falling'),
             # Optional RSI condition
            (lambda x: rsi_col and x.get(rsi_col, 50) > 30 and x.get(rsi_col, 50) < 50), # RSI bearish but not oversold
            # Optional BB condition
            (lambda x: x.get(bb_status_col, '') != 'Testing Lower Band')
        ]
    }

    def calculate_sentiment(row):
        # Filter out None results from conditions where columns might be missing
        bull_score = sum(cond(row) for cond in conditions['bullish'] if cond(row) is not None)
        bear_score = sum(cond(row) for cond in conditions['bearish'] if cond(row) is not None)

        # Adjust threshold based on available signals? Maybe require >= 2 signals?
        min_signals_threshold = 2
        if bull_score > bear_score and bull_score >= min_signals_threshold:
            return 'Bullish'
        elif bear_score > bull_score and bear_score >= min_signals_threshold:
            return 'Bearish'
        else:
            return 'Neutral/Mixed'

    # Apply using .loc to avoid SettingWithCopyWarning
    df.loc[:, 'Technical_Sentiment'] = df.apply(calculate_sentiment, axis=1)
    return df

# --- Main Analysis Function ---

def analyze_market_activity(
    current_stock_data: Optional[pd.DataFrame], # Make optional as it's only for volume flagging and status cols
    historical_stock_data: pd.DataFrame,
    options_volume_data: List[Dict],
    config: dict
) -> Tuple[Dict[str, str], Optional[pd.DataFrame], Optional[float]]: # Return: {Flagged}, TA_DataFrame, PCR
    """
    Performs analysis: flags stocks, calculates RSI, MACD, SMA, BBands, OBV, Put/Call ratio.
    Returns a consolidated DataFrame of TA results.
    """
    flagged_reasons = {}
    combined_ta_df = None
    market_pcr = None

    analysis_params = config.get('analysis_parameters', {})
    trading_lookback = analysis_params.get('trading_volume_lookback_days', 20)
    trading_threshold = analysis_params.get('trading_volume_threshold_ratio', 1.5)
    # options_top_n = analysis_params.get('options_volume_top_n', 5) # Use if find_high_options_volume_stocks uses it
    options_top_n = 5 # Hardcoded for now as in the function
    rsi_window = analysis_params.get('rsi_window', 14)
    macd_fast = analysis_params.get('macd_fast', 12)
    macd_slow = analysis_params.get('macd_slow', 26)
    macd_signal = analysis_params.get('macd_signal', 9)
    sma_windows = analysis_params.get('sma_windows', [50, 200])
    bbands_length = analysis_params.get('bbands_length', 20)
    bbands_std = analysis_params.get('bbands_std', 2.0)

    if historical_stock_data.empty:
        print("Error: Historical stock data is empty, cannot perform TA.")
        return flagged_reasons, None, None

    if not isinstance(historical_stock_data.index, pd.MultiIndex):
        print("Warning: Expected MultiIndex for historical_stock_data in analyze_market_activity.")
        # Attempt to proceed, but results might be incomplete
        # For now, let's enforce MultiIndex requirement here for simplicity
        print("Error: analyze_market_activity currently requires historical_data with MultiIndex ('Date', 'Ticker').")
        return flagged_reasons, None, None

    # --- Calculate All Technical Indicators ---
    print("Calculating technical indicators...")
    all_ta_calcs = {
        'RSI': calculate_rsi(historical_stock_data, window=rsi_window),
        'MACD': calculate_macd(historical_stock_data, fast=macd_fast, slow=macd_slow, signal=macd_signal),
        'SMA': calculate_sma(historical_stock_data, windows=sma_windows),
        'BBands': calculate_bbands(historical_stock_data, length=bbands_length, std=bbands_std),
        'OBV': calculate_obv(historical_stock_data)
    }

    # --- Combine TA Results into a single DataFrame ---
    ticker_index = historical_stock_data.index.get_level_values('Ticker').unique()
    # Ensure combined_ta_df is a copy initially
    combined_ta_df = pd.DataFrame(index=ticker_index).copy()

    for name, df_ta_result in all_ta_calcs.items(): # Renamed df to df_ta_result
        if df_ta_result is not None:
            if isinstance(df_ta_result, pd.Series):
                # Ensure Series name matches if not already set
                if df_ta_result.name is None and name == 'RSI': df_ta_result = df_ta_result.rename('RSI')
                if df_ta_result.name is not None:
                     combined_ta_df = combined_ta_df.join(df_ta_result)
                else:
                     print(f"Warning: TA Series for {name} has no name, cannot join.")
            elif isinstance(df_ta_result, pd.DataFrame):
                combined_ta_df = combined_ta_df.join(df_ta_result)
            else:
                 print(f"Warning: Unexpected type for TA result {name}: {type(df_ta_result)}")

    # --- Add Status Columns (requires current price) ---
    if current_stock_data is not None and not current_stock_data.empty and not combined_ta_df.empty:
         # Ensure current data has 'Ticker' index if multi-index
         if isinstance(current_stock_data.index, pd.MultiIndex):
             current_prices = current_stock_data.xs(current_stock_data.index.levels[0][-1], level='Date')['Close'] # Get latest Close
         else: # Assume single index with tickers
             current_prices = current_stock_data['Close']

         combined_ta_df = combined_ta_df.join(current_prices.rename('Close'))
         # --> Add print statement here <--
         print("DEBUG: Columns before calling _add_status_columns:", combined_ta_df.columns)
         combined_ta_df = _add_status_columns(combined_ta_df) # Add Price_vs_SMA, BB_Status etc.

    # --- Volume Analysis (uses current_stock_data and historical_stock_data) ---
    if current_stock_data is not None and not current_stock_data.empty:
        print("Analyzing trading volume...")
        high_trading_vol_stocks = find_high_trading_volume_stocks(current_data=current_stock_data, historical_data=historical_stock_data, lookback_days=trading_lookback, threshold_ratio=trading_threshold)
        for symbol in high_trading_vol_stocks:
             if symbol in flagged_reasons:
                  flagged_reasons[symbol] += ", High Trading Volume"
             else: flagged_reasons[symbol] = 'High Trading Volume'
    else:
        print("Warning: Current stock data not available for trading volume analysis.")

    # --- Options Volume Analysis ---
    if options_volume_data:
        print("Analyzing options volume...")
        high_options_vol_stocks = find_high_options_volume_stocks(options_data=options_volume_data, top_n=options_top_n)
        for symbol in high_options_vol_stocks:
            if symbol in flagged_reasons:
                flagged_reasons[symbol] += ', High Options Volume'
            else: flagged_reasons[symbol] = 'High Options Volume'

    # --- Put/Call Ratio Calculation ---
    print("Calculating Market Put/Call Ratio...")
    market_pcr = calculate_market_put_call_ratio(options_volume_data)

    print("Analysis complete.")
    return flagged_reasons, combined_ta_df, market_pcr


# --- Volume/Options Analysis Functions (Keep existing implementations) ---

def calculate_average_volumes(historical_data: pd.DataFrame, lookback_days: int) -> pd.Series:
    if historical_data.empty or 'Volume' not in historical_data.columns:
        return pd.Series(dtype=float)
    historical_data = historical_data.sort_index(level='Date')
    avg_vol_data = historical_data.groupby(level='Ticker').apply(
        lambda x: x.iloc[:-1]['Volume'].mean() if not x.empty and len(x) > 1 else 0,
        include_groups=False
    )
    return avg_vol_data.rename('Average Volume')

def find_high_trading_volume_stocks(current_data: pd.DataFrame,
                                  historical_data: pd.DataFrame,
                                  lookback_days: int,
                                  threshold_ratio: float) -> List[str]:
    flagged_stocks = []
    if current_data.empty or 'Volume' not in current_data.columns or historical_data.empty:
        return flagged_stocks
    avg_volumes = calculate_average_volumes(historical_data, lookback_days)
    if avg_volumes.empty:
        return flagged_stocks
    latest_volumes = current_data['Volume']
    volume_comparison = pd.DataFrame({'Latest': latest_volumes, 'Average': avg_volumes})
    volume_comparison = volume_comparison.dropna()
    high_volume_mask = (volume_comparison['Latest'] > volume_comparison['Average'] * threshold_ratio) & (volume_comparison['Average'] > 0)
    flagged_stocks = volume_comparison[high_volume_mask].index.tolist()
    return flagged_stocks

def find_high_options_volume_stocks(options_data: List[Dict],
                                  top_n: int = 5) -> List[str]:
    if not options_data: return []
    valid_options_data = [d for d in options_data if d and 'total_options_volume' in d and d['total_options_volume'] > 0]
    if not valid_options_data: return []
    sorted_options = sorted(valid_options_data, key=lambda x: x['total_options_volume'], reverse=True)
    flagged_stocks = [d['symbol'] for d in sorted_options[:top_n]]
    return flagged_stocks

def calculate_market_put_call_ratio(options_data: List[Dict]) -> Optional[float]:
    if not options_data: return None
    total_calls = sum(d.get('total_call_volume', 0) for d in options_data if d)
    total_puts = sum(d.get('total_put_volume', 0) for d in options_data if d)
    if total_calls > 0: return round(total_puts / total_calls, 3)
    else: return None


# --- Example Usage (Needs Update) ---
if __name__ == '__main__':
    # This example needs significant updates to generate appropriate dummy data
    # with MultiIndex and required columns for all the new TA calculations.
    # For now, just indicate it needs updating.
    print("\n--- analyze_market_activity Example Usage (Needs Update) ---")
    print("Dummy data generation and function call need revision for new TA indicators.")
    # ... (Keep dummy data setup conceptually, but it needs OHLCV) ...
    # ... (Call analyze_market_activity and print results) ...