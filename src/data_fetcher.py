import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Data Fetching Functions ---

def fetch_current_data(tickers: list[str]) -> pd.DataFrame:
    """
    Fetches the most recent available market data for a list of tickers.
    Handles potential variations in yfinance output structure.

    Args:
        tickers: A list of stock/index symbols.

    Returns:
        A pandas DataFrame indexed by Ticker, containing recent data
        (e.g., price, volume, change), or an empty DataFrame if fetching fails.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        ticker_data = yf.Tickers(tickers)
        hist = ticker_data.history(period="5d", progress=False)

        if hist.empty:
            print(f"Warning: No history data returned for tickers: {tickers}")
            return pd.DataFrame()

        # --- Robust handling of yfinance output structure ---
        hist_processed = hist
        if isinstance(hist.columns, pd.MultiIndex) and len(hist.columns.levels) > 1:
            # Check if the Ticker symbol might be in the column names (level 1)
            if hist.columns.names[1] is not None and 'ticker' in str(hist.columns.names[1]).lower():
                 hist_processed = hist.stack(level=1) # Stack the ticker level
            elif hist.columns.names[0] is not None and 'ticker' in str(hist.columns.names[0]).lower():
                 hist_processed = hist.stack(level=0)
            else: # Attempt default stacking if names are None or unexpected
                 try:
                     hist_processed = hist.stack(level=1)
                 except Exception as stack_err:
                     print(f"Warning: Failed to stack yfinance history columns, proceeding with original structure. Error: {stack_err}")
                     hist_processed = hist

        # --- Get the latest data row for each ticker ---
        latest_data = pd.DataFrame()
        if isinstance(hist_processed.index, pd.MultiIndex):
            # Check for standard ('Date', 'Ticker') index
            if all(name in hist_processed.index.names for name in ['Date', 'Ticker']):
                latest_data = hist_processed.groupby(level="Ticker").tail(1)
                # Drop Date level from index after grouping
                latest_data = latest_data.reset_index(level='Date', drop=True)
            # Check for ('Ticker', 'Date') index
            elif all(name in hist_processed.index.names for name in ['Ticker', 'Date']):
                 latest_data = hist_processed.groupby(level="Ticker").tail(1)
                 latest_data = latest_data.reset_index(level='Date', drop=True)
            else:
                 # Fallback for unknown MultiIndex structure
                 print("Warning: Unknown MultiIndex structure, attempting to get last row per ticker.")
                 try:
                    latest_data = hist_processed.groupby(level=0).tail(1) # Assume ticker is level 0 if not named
                    # Try to ensure index is Ticker
                    if not isinstance(latest_data.index, pd.Index) or latest_data.index.name != 'Ticker':
                         latest_data = latest_data.reset_index(level=1, drop=True) # Drop the assumed Date level
                 except Exception as group_err:
                    print(f"Error grouping processed history: {group_err}")
                    # Fallback to just taking the last row overall if grouping fails
                    if not hist_processed.empty:
                        latest_data = hist_processed.iloc[[-1]]

        elif not hist_processed.empty:
            # Not a MultiIndex, potentially single ticker result or flat structure
            latest_data = hist_processed.iloc[[-1]]
            # If index is Date, we lose ticker info here without more logic
            # If index is already Ticker, this is fine.
            if isinstance(latest_data.index, pd.DatetimeIndex) and len(tickers) == 1:
                # If we know it was one ticker, set the index
                latest_data.index = pd.Index([tickers[0]], name='Ticker')

        if latest_data.empty:
            print(f"Warning: Could not extract latest data rows for {tickers}")
            return pd.DataFrame()

        # Ensure the index is named 'Ticker' if possible
        if latest_data.index.name != 'Ticker':
             latest_data.index.name = 'Ticker'

        # --- Extract relevant columns ---
        relevant_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols_to_select = [col for col in relevant_cols if col in latest_data.columns]
        if not cols_to_select:
            print(f"Warning: None of the relevant columns {relevant_cols} found.")
            return pd.DataFrame()

        data_subset = latest_data[cols_to_select].copy() # Use .copy() early

        # --- Calculate price change ---
        if 'Open' in data_subset.columns and 'Close' in data_subset.columns:
            data_subset['Price Change'] = data_subset['Close'] - data_subset['Open']
            data_subset['Percent Change'] = (data_subset['Price Change'] / data_subset['Open']) * 100

        return data_subset

    except Exception as e:
        print(f"Error processing current data for {tickers}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_historical_data(tickers: list[str], lookback_days: int) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for calculating averages.
    Ensures Ticker is in the index if possible.

    Args:
        tickers: A list of stock/index symbols.
        lookback_days: Number of past days to fetch data for.

    Returns:
        A pandas DataFrame with historical data, indexed by Date and Ticker.
        Returns an empty DataFrame if fetching fails.
    """
    if not tickers or lookback_days <= 0:
        return pd.DataFrame()
    try:
        # Fetch a bit more to ensure enough trading days
        period_str = f"{lookback_days+15}d" # Increased buffer slightly

        ticker_data = yf.Tickers(tickers)
        hist = ticker_data.history(period=period_str, interval="1d", progress=False)

        if hist.empty:
            print(f"Warning: No historical data returned for tickers: {tickers}")
            return pd.DataFrame()

        # --- Handle potential structure variations ---
        if isinstance(hist.columns, pd.MultiIndex) and len(hist.columns.levels) > 1:
            # Stack the Ticker level from columns into the index
            hist_processed = hist.stack(level=1)
        else:
            # Assume Ticker is already in index or handle other structures if needed
            hist_processed = hist

        # Ensure index names include Date and Ticker if possible
        # yfinance might return just Date index for single ticker
        if isinstance(hist_processed.index, pd.MultiIndex):
            if 'Ticker' not in hist_processed.index.names or 'Date' not in hist_processed.index.names:
                 # Try to assign standard names if unnamed MultiIndex
                 try:
                     hist_processed.index.names = ['Date', 'Ticker']
                 except Exception as name_e:
                     print(f"Warning: Could not assign standard index names: {name_e}")
        elif isinstance(hist_processed.index, pd.DatetimeIndex) and len(tickers) == 1:
            # If single ticker, add Ticker level to index
            hist_processed['Ticker'] = tickers[0]
            hist_processed = hist_processed.set_index([hist_processed.index, 'Ticker'])
            hist_processed.index.names = ['Date', 'Ticker']
        else:
            print("Warning: Unexpected index structure in historical data.")


        # Filter to roughly the desired number of days *after* processing structure
        # Group by ticker and take tail
        if isinstance(hist_processed.index, pd.MultiIndex) and 'Ticker' in hist_processed.index.names:
            hist_filtered = hist_processed.groupby(level='Ticker').tail(lookback_days + 1) # +1 to have data for avg calc
            return hist_filtered
        else:
             # If we couldn't guarantee Ticker in index, return processed data
             # Analysis engine will need to be robust
             return hist_processed

    except Exception as e:
        print(f"Error fetching/processing historical data for {tickers}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def fetch_options_volume(ticker_symbol: str) -> dict:
    """
    Fetches the total volume for call and put options for the nearest expiry dates.
    Note: This fetches data for ONE ticker at a time as yfinance options handling
          for multiple tickers simultaneously can be complex.

    Args:
        ticker_symbol: The stock symbol.

    Returns:
        A dictionary containing total call and put volume (e.g.,
        {'total_call_volume': 1000, 'total_put_volume': 500})
        or an empty dict if data is unavailable or fetching fails.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        # Get available option expiry dates
        expiries = stock.options
        if not expiries:
            # print(f"No option expiry dates found for {ticker_symbol}")
            return {}

        # Often focus on the nearest expiry dates (e.g., first few)
        total_call_volume = 0
        total_put_volume = 0

        # Let's sum volume across the first few expiries (e.g., up to 2)
        for expiry in expiries[:2]:
            opt_chain = stock.option_chain(expiry)
            # Sum volume from calls and puts for this expiry
            call_vol = opt_chain.calls['volume'].sum() if 'volume' in opt_chain.calls.columns else 0
            put_vol = opt_chain.puts['volume'].sum() if 'volume' in opt_chain.puts.columns else 0
            total_call_volume += call_vol
            total_put_volume += put_vol

        return {
            'symbol': ticker_symbol,
            'total_call_volume': total_call_volume,
            'total_put_volume': total_put_volume,
            'total_options_volume': total_call_volume + total_put_volume
        }

    except Exception as e:
        # yfinance often raises errors for non-existent tickers or options data
        # print(f"Could not fetch options data for {ticker_symbol}: {e}")
        return {}

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # Use symbols from config or define test symbols
    test_indices = ['^GSPC', '^IXIC']
    # test_stocks = ['AAPL', 'MSFT', 'NVDA', 'NONEXISTENT'] # Include a fake one
    test_stocks = ['AAPL', 'MSFT', 'NVDA']


    print("--- Fetching Current Data ---")
    current_data = fetch_current_data(test_indices + test_stocks)
    print(current_data)
    print("\n")

    print("--- Fetching Historical Data (for avg volume calc later) ---")
    hist_data = fetch_historical_data(test_stocks, lookback_days=20)
    if not hist_data.empty:
        if isinstance(hist_data.index, pd.MultiIndex) and 'Ticker' in hist_data.index.names:
             print(hist_data.groupby(level='Ticker').tail(2)) # Show last 2 days per ticker
        else:
             print(hist_data.tail()) # Show tail if structure is unexpected
    else:
        print("Historical data fetch failed or returned empty.")
    print("\n")

    print("--- Fetching Options Volume (per stock) ---")
    for stock in test_stocks:
        options_vol = fetch_options_volume(stock)
        if options_vol:
            print(options_vol)
        else:
            print(f"No options data retrieved for {stock}") 