# Project: Market Sentiment AI Analyzer (V1)

**1. Project Goal:**

To create a tool that integrates real-time market data (from Yahoo Finance) and news sentiment analysis (via Google Gemini API) to provide insights into broad market trends and specific stock activity, focusing on accessible technical signals like options volume and trading volume. The project is designed with a modular architecture for future expansion.

**2. Core Features (Initial Scope - V1):**

*   Fetch daily market data for major US indices (S&P 500, Nasdaq).
*   Fetch daily market & options data for a predefined list of ~10-20 stocks.
*   Identify Top ~5 stocks based on relative total options volume.
*   Identify Top ~5 stocks based on relative trading volume (compared to average).
*   Fetch recent news headlines (Top 3-5) for indices and flagged stocks via News API.
*   Analyze sentiment (Positive/Negative/Neutral) of each news headline using Google Gemini API.
*   Aggregate sentiment scores per index/stock.
*   Present a console-based summary including index performance, flagged stocks (with reason), and associated sentiment scores.
*   Utilize free-tier APIs (Yahoo Finance via `yfinance`, News API, Google AI Gemini).
*   Manage API keys and parameters via configuration files.

**3. Technology Stack:**

*   **Language:** Python 3.x
*   **Market Data:** `yfinance` library
*   **News Data:** `newsapi-python` library (or direct `requests` to News API)
*   **AI Sentiment:** `google-generativeai` library (for Google Gemini API)
*   **Data Handling:** `pandas`
*   **Configuration:** `python-dotenv` (for `.env`), `PyYAML` (for `config.yaml`)
*   **Environment:** `requirements.txt` for dependency management

**4. Project Structure:**

```
market_sentiment_ai/
├── src/
│   ├── __init__.py
│   ├── config_manager.py     # Loads config files (.env, config.yaml)
│   ├── data_fetcher.py       # Fetches market/options data via yfinance
│   ├── news_fetcher.py       # Fetches news via News API
│   ├── analysis_engine.py    # Performs technical analysis (options/volume screening)
│   ├── sentiment_analyzer.py # Analyzes sentiment via Gemini API
│   └── main.py               # Orchestrates the workflow, presents results
├── .env                      # Stores API keys (GIT IGNORED!)
├── config.yaml               # Stores parameters (indices, stock list, thresholds)
├── requirements.txt          # Python dependencies
├── README.md                 # Project description, setup, usage
├── FRAMEWORK.md              # This file!
└── .gitignore                # Git ignore file
```

**5. Module Responsibilities:**

*   **`config_manager.py`**: Provides functions to load API keys from `.env` and parameters (stock lists, index lists, analysis thresholds) from `config.yaml`.
*   **`data_fetcher.py`**: Contains functions to get dataframes/dictionaries of market data (quotes, historical for average volume calc) and options data (volumes) for specified symbols using `yfinance`.
*   **`news_fetcher.py`**: Contains functions to query the News API for articles related to given keywords (symbols/indices) and return relevant data (headlines, URLs, snippets). Requires News API key.
*   **`analysis_engine.py`**: Takes market/options data (from `data_fetcher`) and applies rules to identify stocks meeting criteria (e.g., options volume significantly above average, trading volume significantly above average). Returns a list of flagged symbols and the reason.
*   **`sentiment_analyzer.py`**: Takes text (news headlines/snippets) and uses the Google Gemini API to determine sentiment. Requires Google AI API key. Returns sentiment scores (e.g., {'Positive': 0.7, 'Negative': 0.1, 'Neutral': 0.2}). Handles API interaction and basic prompt engineering.
*   **`main.py`**: The entry point. Uses `config_manager` to get settings. Calls `data_fetcher`, then `analysis_engine` to get flagged stocks. Calls `news_fetcher` for relevant news. Calls `sentiment_analyzer` for sentiment. Compiles and prints the final summary to the console.

**6. Data Flow:**

1.  `main.py` loads config (`config_manager`).
2.  `main.py` requests market/options data for configured symbols (`data_fetcher`).
3.  `main.py` passes data to `analysis_engine` to get flagged symbols.
4.  `main.py` requests news for indices and flagged symbols (`news_fetcher`).
5.  `main.py` passes news headlines to `sentiment_analyzer`.
6.  `main.py` aggregates results and prints the summary.

**7. Configuration Files:**

*   **`.env`**:
    ```dotenv
    NEWS_API_KEY=YOUR_NEWS_API_KEY
    GOOGLE_AI_API_KEY=YOUR_GOOGLE_AI_API_KEY
    ```
*   **`config.yaml`**:
    ```yaml
    market_indices:
      - '^GSPC' # S&P 500
      - '^IXIC' # Nasdaq
    stocks_to_track:
      - AAPL
      - MSFT
      - GOOGL
      - AMZN
      - TSLA
      - JPM
      - NVDA
      - V
      - WMT
      - UNH
    analysis_parameters:
      options_volume_lookback_days: 10 # Days to average options volume over
      options_volume_threshold_ratio: 2.0 # Flag if today's volume > X * average
      trading_volume_lookback_days: 20 # Days to average trading volume over
      trading_volume_threshold_ratio: 1.5 # Flag if today's volume > Y * average
      news_articles_per_symbol: 3
      sentiment_model: 'gemini-1.5-flash-latest' # Updated model
    ```

**8. Future Enhancements (Post V1):**

*   More sophisticated technical indicators.
*   Alternative data sources (social media, more news APIs).
*   Historical data storage and trend analysis.
*   Web-based UI (Streamlit, Dash, Flask).
*   User accounts and personalized watchlists.
*   Alerting system.
*   More advanced AI models or fine-tuning. 