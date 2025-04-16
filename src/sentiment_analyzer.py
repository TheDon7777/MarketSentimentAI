import google.generativeai as genai
import time
import re
import pandas as pd # Added pandas import for Series typing
from typing import List, Dict, Any, Tuple, Optional

# Import config functions
from .config_manager import get_api_key, get_config_parameter

# --- Global Variables ---
_model = None

# --- Initialization ---

def initialize_google_ai():
    """Initializes the Google Generative AI client and model."""
    global _model
    if _model:
        return _model

    api_key = get_api_key('GOOGLE_AI_API_KEY')
    if not api_key or api_key == 'YOUR_GOOGLE_AI_API_KEY':
        print("Error: Google AI API key not found or not set in .env file.")
        print("Please obtain a key from https://aistudio.google.com/app/apikey and add it.")
        return None

    try:
        genai.configure(api_key=api_key)
        model_name = get_config_parameter('analysis_parameters', 'sentiment_model', 'gemini-1.5-flash-latest')
        # Adjusted config for potentially longer, more structured response
        generation_config = {
            "temperature": 0.6, # Slightly lower temp for more predictable structure
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 150, # Allow more tokens for summary
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        _model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        print(f"Google AI Model '{model_name}' initialized successfully.")
        return _model
    except Exception as e:
        print(f"Error initializing Google AI model: {e}")
        if "API key not valid" in str(e):
            print("Please ensure your GOOGLE_AI_API_KEY in .env is correct.")
        _model = None
        return None

# --- Sentiment Analysis Functions (Batch Approach with Enhanced TA) ---

def parse_structured_response(response_text: str) -> Dict[str, Any]:
    """Extracts structured fields (Sentiment, Themes, Technical Signal, Outlook) from the response."""
    parsed = {
        'sentiment': 'Unknown',
        'themes': 'N/A',
        'technical_signal': 'N/A',
        'outlook': 'Could not parse response.'
    }
    try:
        # Use regex to find the structured fields, case-insensitive, multiline
        sentiment_match = re.search(r"Overall Sentiment: (.*?)$", response_text, re.IGNORECASE | re.MULTILINE)
        themes_match = re.search(r"Key News Themes:(.*?)(?:Technical Signal:|Synthesized Outlook:|$)", response_text, re.IGNORECASE | re.DOTALL)
        technical_match = re.search(r"Technical Signal:(.*?)(?:Key News Themes:|Synthesized Outlook:|$)", response_text, re.IGNORECASE | re.DOTALL)
        outlook_match = re.search(r"Synthesized Outlook:(.*?)(?:Key News Themes:|Technical Signal:|$)", response_text, re.IGNORECASE | re.DOTALL)

        if sentiment_match:
            sentiment_word = sentiment_match.group(1).strip().capitalize()
            if sentiment_word in ['Positive', 'Negative', 'Neutral']:
                 parsed['sentiment'] = sentiment_word

        if themes_match:
            themes = themes_match.group(1).strip()
            # Clean up potential bullet points/newlines
            themes = re.sub(r'^\s*[-*]\s*', '', themes, flags=re.MULTILINE)
            themes = themes.replace('\n', ' ').strip()
            parsed['themes'] = themes if themes else 'N/A'

        if technical_match:
            tech_signal = technical_match.group(1).strip()
            parsed['technical_signal'] = tech_signal if tech_signal else 'N/A'

        if outlook_match:
            outlook = outlook_match.group(1).strip()
            parsed['outlook'] = outlook if outlook else 'N/A'
        elif sentiment_match: # If outlook wasn't found explicitly, maybe it followed sentiment?
            # This is a less reliable fallback
             potential_outlook = response_text[sentiment_match.end():].strip()
             # Try to filter out other sections if they exist
             if themes_match: potential_outlook = potential_outlook.split("Key News Themes:")[0].strip()
             if technical_match: potential_outlook = potential_outlook.split("Technical Signal:")[0].strip()
             if potential_outlook and len(potential_outlook) > 5: # Adjusted length check slightly
                 parsed['outlook'] = potential_outlook

    except Exception as e:
        print(f"Error parsing structured response: {e}. Response was: {response_text[:100]}...")
        # Keep default error message in outlook
        pass # Keep defaults

    return parsed

def interpret_rsi(rsi: Optional[float]) -> str:
    if pd.isna(rsi): return "RSI: N/A"
    status = "Neutral"
    if rsi > 70: status = "Overbought"
    elif rsi < 30: status = "Oversold"
    return f"RSI: {rsi:.1f} ({status})"

def interpret_macd(macd: Optional[float], signal: Optional[float], hist: Optional[float]) -> str:
    if pd.isna(macd) or pd.isna(signal) or pd.isna(hist): return "MACD: N/A"
    status = "Neutral/Crossing"
    if macd > signal and hist > 0: status = "Bullish Crossover"
    elif macd < signal and hist < 0: status = "Bearish Crossover"
    elif macd > signal: status = "Bullish (Above Signal)"
    elif macd < signal: status = "Bearish (Below Signal)"
    return f"MACD: {status} (Hist: {hist:+.2f})"

def interpret_sma(close: Optional[float], sma50: Optional[float], sma200: Optional[float]) -> str:
    if pd.isna(close) or pd.isna(sma50) or pd.isna(sma200): return "SMA Trend: N/A"
    price_vs_50 = "Above" if close > sma50 else "Below"
    price_vs_200 = "Above" if close > sma200 else "Below"
    sma50_vs_200 = "Above" if sma50 > sma200 else "Below"

    trend = "Mixed"
    if price_vs_50 == "Above" and price_vs_200 == "Above" and sma50_vs_200 == "Above":
        trend = "Strong Uptrend (Price > SMA50 > SMA200)"
    elif price_vs_50 == "Below" and price_vs_200 == "Below" and sma50_vs_200 == "Below":
        trend = "Strong Downtrend (Price < SMA50 < SMA200)"
    elif price_vs_50 == "Above" and sma50_vs_200 == "Above":
        trend = "Uptrend (Price > SMA50, SMA50 > SMA200)"
    elif price_vs_50 == "Below" and sma50_vs_200 == "Below":
        trend = "Downtrend (Price < SMA50, SMA50 < SMA200)"
    # Add other conditions if needed (e.g., price crossing)
    return f"SMA Trend: {trend}"

def interpret_bbands(close: Optional[float], lower: Optional[float], upper: Optional[float]) -> str:
    if pd.isna(close) or pd.isna(lower) or pd.isna(upper): return "Bollinger Bands: N/A"
    if close >= upper: return "Bollinger Bands: Testing Upper Band"
    if close <= lower: return "Bollinger Bands: Testing Lower Band"
    return "Bollinger Bands: Within Bands"

def interpret_obv(trend: Optional[str]) -> str:
    if pd.isna(trend) or trend is None: return "OBV Trend: N/A"
    return f"OBV Trend: {trend}"

def interpret_pcr(pcr_value: Optional[float]) -> str:
    """Provides a simple text interpretation of Put/Call ratio."""
    if pcr_value is None: return "(Market PCR N/A)"
    status = "Neutral"
    if pcr_value > 0.7: status = "Bearish Skew"
    elif pcr_value < 0.5: status = "Bullish Skew"
    return f"Market PCR={pcr_value:.3f} ({status})"

def analyze_symbol_sentiment_with_ta(
    symbol: str,
    headlines: List[str],
    ta_data: Optional[pd.Series] = None,
    market_pcr: Optional[float] = None,
    retry_count=2,
    delay=5
) -> Dict[str, Any]:
    """Analyzes sentiment for a symbol using headlines and consolidated TA, returning structured results."""
    global _model
    default_error_result = {'sentiment': 'Error', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': 'Model not initialized.'}
    if not _model: return default_error_result

    if not headlines:
        return {'sentiment': 'Neutral', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': 'No headlines provided.'}

    formatted_headlines = "\n".join([f"- {h}" for h in headlines if h])
    if not formatted_headlines: return {'sentiment': 'Neutral', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': 'No valid headlines found.'}

    # --- Construct Enhanced Prompt ---
    tech_context_lines = []
    if ta_data is not None and isinstance(ta_data, pd.Series):
        # Extract raw values
        rsi = ta_data.get('RSI')
        macd_line = ta_data.get('MACD')
        macd_signal = ta_data.get('Signal')
        macd_hist = ta_data.get('Histogram')
        close = ta_data.get('Close') # Need price for SMA/BB interpretation
        sma50 = ta_data.get('SMA_50')
        sma200 = ta_data.get('SMA_200')
        bb_lower = ta_data.get('BB_Lower')
        bb_upper = ta_data.get('BB_Upper')
        obv_trend = ta_data.get('OBV_Trend')

        # Get interpreted strings
        tech_context_lines.append(f"- {interpret_rsi(rsi)}")
        tech_context_lines.append(f"- {interpret_macd(macd_line, macd_signal, macd_hist)}")
        tech_context_lines.append(f"- {interpret_sma(close, sma50, sma200)}")
        tech_context_lines.append(f"- {interpret_bbands(close, bb_lower, bb_upper)}")
        tech_context_lines.append(f"- {interpret_obv(obv_trend)}")

    else:
        tech_context_lines.append("(Technical indicator data not available for this symbol)")

    # Market-wide context
    pcr_text = interpret_pcr(market_pcr)
    tech_context_lines.append(f"- {pcr_text}") # Add market context

    tech_context_str = "\n".join(tech_context_lines)

    prompt = (
        f"Analyze the short-term outlook for stock symbol \"{symbol}\" based *primarily* on the following recent news headlines, "
        f"while *considering* the provided technical context. Do not use outside knowledge."
        f"\n\nTechnical Context:\n{tech_context_str}\n"
        f"\nRecent News Headlines:\n{formatted_headlines}\n\n"
        f"Provide your analysis in the following structured format EXACTLY, with each field on a new line:\n"
        f"Overall Sentiment: [One word: Positive, Negative, or Neutral] based mainly on news\n"
        f"Key News Themes: [Bulleted list or comma-separated phrases for 1-3 key topics/events from headlines]\n"
        f"Technical Signal: [One sentence summarizing the combined message of the technical indicators provided]\n"
        f"Synthesized Outlook: [One concise sentence combining news sentiment and technical signals for a brief outlook]"
    )

    current_retry = 0
    while current_retry <= retry_count:
        try:
            response = _model.generate_content(prompt)
            if not response.parts:
                block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
                print(f"Warning: Content generation blocked for symbol {symbol}. Reason: {block_reason}")
                return {'sentiment': 'Blocked', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': f'Content generation blocked: {block_reason}'}

            response_text = response.text.strip()
            parsed_result = parse_structured_response(response_text)
            return parsed_result

        except Exception as e:
            print(f"Error calling Google AI API for symbol {symbol}: {e}")
            if "Resource has been exhausted" in str(e) or "429" in str(e):
                print(f"Rate limit likely exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                current_retry += 1
                delay *= 2
            else:
                return {'sentiment': 'Error', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': f'API call failed: {e}'}
        except ValueError as ve:
             print(f"ValueError processing response for symbol {symbol}: {ve}")
             return {'sentiment': 'Error', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': f'Value error processing response: {ve}'}

    print(f"Failed to get sentiment for symbol {symbol} after {retry_count} retries.")
    return {'sentiment': 'Error', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': 'API call failed after retries.'}

# --- Aggregation Function (Updated for Consolidated TA) ---

def get_sentiment_for_news(
    news_data: Dict[str, List[Dict[str, Any]]],
    combined_ta_data: Optional[pd.DataFrame] = None, # Changed to accept the consolidated TA DataFrame
    market_pcr: Optional[float] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes sentiment for batches of news articles per symbol, incorporating consolidated TA data.

    Args:
        news_data: Dictionary from fetch_news_for_symbols {symbol: [article_list]}.
        combined_ta_data: DataFrame containing latest TA indicators for all stocks.
        market_pcr: Float representing the calculated market Put/Call ratio.

    Returns:
        Dictionary mapping symbols to their structured analysis results.
    """
    global _model
    _model = initialize_google_ai()
    default_error = {'sentiment': 'Error', 'themes': 'N/A', 'technical_signal': 'N/A', 'outlook': 'Google AI model not initialized.'}
    if not _model:
        print("Skipping sentiment analysis due to initialization failure.")
        return {symbol: default_error for symbol in news_data}

    sentiment_results = {}
    total_symbols = len(news_data)
    processed_count = 0

    print(f"Starting sentiment analysis for {total_symbols} symbols...")

    for symbol, articles in news_data.items():
        processed_count += 1
        print(f"Analyzing sentiment for {symbol} ({processed_count}/{total_symbols})...")

        headlines = [article.get('title') for article in articles if article.get('title')] # Filter None titles

        # Extract the TA data Series for the current symbol
        ta_data_for_symbol = None
        if combined_ta_data is not None and symbol in combined_ta_data.index:
            ta_data_for_symbol = combined_ta_data.loc[symbol]
        else:
            print(f"Warning: TA data not found for symbol {symbol} in combined DataFrame.")

        # Call the analysis function with the extracted TA Series
        analysis = analyze_symbol_sentiment_with_ta(
            symbol=symbol,
            headlines=headlines,
            ta_data=ta_data_for_symbol, # Pass the Series for this symbol
            market_pcr=market_pcr
        )
        sentiment_results[symbol] = analysis

        # Optional: Add a small delay to potentially avoid rate limits if analyzing many symbols
        # time.sleep(0.5) # Adjust delay as needed

    print("Sentiment analysis complete.")
    return sentiment_results

# --- Example Usage (Updated for Structured Output) ---
if __name__ == '__main__':
    print("--- Testing Sentiment Analyzer (Batch Mode with Enhanced TA) ---")
    model_instance = initialize_google_ai()

    if model_instance:
        dummy_news = {
            'AAPL': [
                {'title': 'Apple unveils stunning new iPhone, shares jump 5%'},
                {'title': 'Analysts raise Apple price target on strong services growth'},
                {'title': 'EU regulators fine Apple over app store practices'}
            ],
            'TSLA': [
                 {'title': 'Tesla recalls 50,000 vehicles over software glitch'},
                 {'title': 'Tesla stock drops amid concerns about competition and delivery numbers'}
            ]
        }
        dummy_rsi = pd.Series({'AAPL': 75.2, 'TSLA': 25.8}, name='RSI')
        dummy_macd = pd.DataFrame({
            'MACD': [1.5, -0.8],
            'Histogram': [0.2, -0.3],
            'Signal': [1.3, -0.5]
            }, index=['AAPL', 'TSLA']) # Index should match symbols
        dummy_pcr = 0.85

        print("\n--- Analyzing Dummy News Data (Batch with Enhanced TA) ---")
        results = get_sentiment_for_news(
            dummy_news,
            rsi_data=dummy_rsi,
            macd_data=dummy_macd,
            market_pcr=dummy_pcr
            )

        print("\n--- Aggregated Sentiment Results (Structured) ---")
        import json
        print(json.dumps(results, indent=2))
    else:
        print("Sentiment analysis cannot proceed without initialized Google AI model.") 