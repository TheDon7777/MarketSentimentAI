import yfinance as yf
import pandas as pd # For timestamp conversion
from typing import List, Dict, Any

# No longer needed: from newsapi import NewsApiClient
# No longer needed: from .config_manager import get_api_key, get_config_parameter, load_yaml_config

# --- News Fetching Function ---

def fetch_news_for_symbols(symbols: List[str],
                           articles_per_symbol: int = 5 # yfinance doesn't limit by count easily, but we can slice
                           ) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches recent news articles for a list of stock/index symbols using yfinance.Search.

    Args:
        symbols: List of symbols (e.g., ['AAPL', '^GSPC']).
        articles_per_symbol: Max number of news articles to retrieve per symbol.

    Returns:
        A dictionary mapping each symbol to a list of its news articles.
        Each article is a dict with keys like 'title', 'link', 'publisher', 'publishedAt'.
    """
    news_results = {}

    for symbol in symbols:
        try:
            # Use the symbol itself as the search query
            query = symbol
            print(f"Searching yfinance news for query: '{query}' (limit: {articles_per_symbol})...")

            # Use yf.Search().news
            search_results = yf.Search(query, news_count=articles_per_symbol)
            news_list = search_results.news # Access the news attribute

            if news_list:
                processed_articles = []
                # Loop through the fetched articles (already limited by news_count)
                for article in news_list:
                    # --- Add a debug print for the raw article --- # Keep for now
                    print(f"DEBUG: Raw article data for {symbol} via Search: {article}")
                    # --- End DEBUG ---

                    # Convert Unix timestamp safely
                    publish_time_str = 'N/A'
                    # Check for both possible timestamp keys yfinance might use
                    ts_key = 'providerPublishTime' if 'providerPublishTime' in article else 'publishTime' if 'publishTime' in article else None
                    if ts_key:
                        try:
                            ts = article.get(ts_key)
                            if ts is not None:
                                publish_time_dt = pd.to_datetime(ts, unit='s', utc=True)
                                publish_time_str = publish_time_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                        except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as ts_err:
                             print(f"Warning: Timestamp conversion error for {symbol} article: {ts_err}")
                             pass

                    # Safely get other fields using .get()
                    processed_articles.append({
                        'uuid': article.get('uuid', 'N/A'),
                        'title': article.get('title'),
                        'link': article.get('link'),
                        'publisher': article.get('publisher'),
                        'publishedAt': publish_time_str,
                        # Add other potential fields if needed from search result
                    })
                news_results[symbol] = processed_articles
            else:
                print(f"No news found for query '{query}' via yfinance.Search.")
                news_results[symbol] = []

        except Exception as e:
            # Catch potential errors during yf.Search() or accessing .news
            print(f"An exception occurred searching yfinance news for query '{query}': {e}")
            import traceback
            traceback.print_exc()
            news_results[symbol] = []

    return news_results

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing News Fetcher (yfinance) ---")
    test_syms = ['^GSPC', 'AAPL', 'NVDA', 'NONEXISTENTTICKER']
    num_articles = 3 # Example number

    all_news = fetch_news_for_symbols(test_syms, articles_per_symbol=num_articles)

    if all_news:
        for symbol, news_list in all_news.items():
            print(f"\n--- News for {symbol} ---")
            if news_list:
                for i, article in enumerate(news_list):
                    print(f"  {i+1}. {article.get('title', '[No Title]')} ({article.get('publisher', 'N/A')})")
                    # print(f"     Published: {article.get('publishedAt')}")
                    # print(f"     Link: {article.get('link')}")
            else:
                print("  No news found or fetch error.")
    else:
        print("News fetching failed for all symbols.") 