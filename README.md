# Market Sentiment AI Analyzer 📈

This project provides a Streamlit-based dashboard that analyzes market sentiment by integrating data from various sources:

*   **Market Data:** Fetches current and historical stock/index data using `yfinance`.
*   **Technical Analysis:** Calculates key indicators like RSI and MACD locally using `pandas-ta`. It also calculates a proxy Market Put/Call Ratio based on options volume for tracked stocks.
*   **News Aggregation:** Retrieves recent news headlines relevant to market indices and specific stocks using `yfinance`.
*   **AI-Powered Sentiment Analysis:** Leverages the Google Gemini API to analyze news headlines (from `yfinance`) *in conjunction with* technical indicators (RSI, MACD, PCR) to provide a synthesized outlook, overall sentiment score, key news themes, and a summary of technical signals.

The goal is to offer a more holistic view than looking at news or technicals in isolation, utilizing free-tier APIs and local calculations where possible.

![Demo GIF](demo.gif)

## Features ✨

*   **Dashboard UI:** Interactive web interface built with Streamlit.
*   **Index Monitoring:** Displays current price, change, and AI-generated sentiment analysis for major market indices (configurable in `config.yaml`).
*   **Notable Stock Activity:** Identifies stocks from a tracked list (configurable in `config.yaml`) exhibiting:
    *   High trading volume relative to their recent average.
    *   High *absolute* options volume (among tracked stocks).
*   **Detailed Stock View:** For flagged stocks, shows:
    *   Current Price, Change, Volume.
    *   Calculated RSI(14) and MACD status.
    *   Structured AI Analysis (powered by Google Gemini):
        *   Overall Sentiment (Positive/Negative/Neutral)
        *   Key News Themes (extracted from recent headlines)
        *   Technical Signal Summary (AI interpretation of RSI/MACD/PCR context)
        *   Synthesized Outlook (combining news and technicals)
    *   Interactive Plotly chart showing Price (Candlestick), Volume, RSI, and MACD.
*   **Market Sentiment Proxy:** Displays a simple Market Put/Call Ratio calculated from the options volume of tracked stocks.
*   **Caching:** Uses Streamlit\'s caching (`@st.cache_data`) to improve performance on subsequent loads (data/analysis cached for 15 minutes).
*   **Configuration:** Uses `.env` for API keys and `config.yaml` for parameters like tracked symbols, lookback periods, and thresholds.
*   **Modularity:** Code is structured into distinct modules for data fetching, analysis, news, sentiment, and the main app logic.

## Project Structure 🏗️

```
market_sentiment_ai/
├── src/
│   ├── __init__.py
│   ├── config_manager.py     # Loads config files
│   ├── data_fetcher.py       # Fetches market/options data (yfinance)
│   ├── analysis_engine.py    # Performs TA (pandas-ta, custom calcs)
│   ├── news_fetcher.py       # Fetches news (yfinance)
│   └── sentiment_analyzer.py # Analyzes sentiment (Google Gemini API)
├── .env                      # Stores API keys (GIT IGNORED!)
├── config.yaml               # Stores parameters (symbols, thresholds)
├── requirements.txt          # Python dependencies
├── main.py                   # Main Streamlit application script
├── FRAMEWORK.md              # Initial project plan document
├── README.md                 # This file!
└── .gitignore                # Git ignore file
```

## Setup & Installation ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd market_sentiment_ai
    ```

2.  **Create and Activate a Virtual Environment:**
    It is highly recommended to use a Python virtual environment to manage dependencies and avoid conflicts. Create one using your preferred method (e.g., `venv`, `conda`). Remember to activate it before installing dependencies.

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This installs specific versions, including downgrading numpy if necessary for pandas-ta compatibility.*

4.  **Obtain API Keys:**
    *   **Google AI (Gemini):** Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

5.  **Configure API Keys:**
    *   Create a file named `.env` in the `market_sentiment_ai` root directory (or rename the template if one exists).
    *   Add your key to the `.env` file like this:
      ```dotenv
      GOOGLE_AI_API_KEY=YOUR_GOOGLE_AI_API_KEY_HERE
      ```
    *   **Important:** The `.gitignore` file should prevent this file from being committed. **Never commit your API keys directly to GitHub.**

6.  **Customize Configuration (Optional):**
    *   Edit `config.yaml` to change:
        *   `market_indices`: List of index symbols (e.g., `^GSPC`, `^IXIC`).
        *   `stocks_to_track`: List of stock symbols (e.g., `AAPL`, `MSFT`).
        *   `analysis_parameters`: Thresholds for volume, lookback periods, number of news articles, etc.

## Usage ▶️

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the project root directory (`market_sentiment_ai`) in your terminal.**
3.  **Run the Streamlit application:**
    ```bash
    python -m streamlit run main.py
    ```
4.  Streamlit will start a local server and usually open the application automatically in your web browser (typically at `http://localhost:8501`).

The application will fetch fresh data (or use cached data), perform the analysis, and display the dashboard.

## Future Enhancements 🚀

*   Integrate more diverse data sources (RSS, alternative sentiment APIs, economic data).
*   Implement more sophisticated options analysis (IV Rank, specific unusual activity screening).
*   Add historical sentiment tracking and visualization.
*   Improve error handling and user feedback within the UI.
*   Refine AI prompts for even more nuanced analysis.
*   Package the project properly for easier distribution.

## Contributing 🤝

Contributions are welcome! Please feel free to submit pull requests or open issues. 