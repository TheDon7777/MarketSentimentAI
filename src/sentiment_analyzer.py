import google.generativeai as genai
import time
import re
import pandas as pd # Added pandas import for Series typing
from typing import List, Dict, Any, Tuple, Optional
import traceback

# Import config functions
from .config_manager import get_api_key, get_config_parameter

# --- Global Variables / Constants ---
_model = None
# Define safety settings at the module level
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

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
            "max_output_tokens": 512, # Increased token limit
        }
        _model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=SAFETY_SETTINGS # Use module-level constant
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

# --- New Portfolio Overview Function --- #
def analyze_portfolio_overview(
    portfolio_df: pd.DataFrame,
    individual_analyses: Dict[str, Dict],
    safety_settings_param: List[Dict],
    model: Optional[genai.GenerativeModel] = None
) -> Dict[str, str]:
    """
    Analyzes the overall portfolio using AI based on composition and individual sentiments.

    Args:
        portfolio_df: DataFrame containing portfolio holdings information.
        individual_analyses: Dictionary with individual analysis results for symbols.
        safety_settings_param: The safety settings list to use for the API call.
        model: The initialized Google Gemini model instance.

    Returns:
        A dictionary containing the overview analysis ('Overall Risk Assessment', 'Diversification Comments', 'Holistic Outlook')
        or an error message.
    """
    global _model # Ensure we use the global model if one isn't passed
    if model is None:
        if _model is None: # Check if global model is initialized
             _model = initialize_google_ai() # Initialize if not
        model = _model # Use the global one
        if model is None:
            return {"error": "AI Model not initialized.", "status": "Error"}

    # --- Input Validation ---
    if not isinstance(portfolio_df, pd.DataFrame) or portfolio_df.empty:
        return {"error": "Portfolio data is missing or invalid.", "status": "Error"}
    if not isinstance(individual_analyses, dict):
         return {"error": "Individual analysis results are missing or invalid.", "status": "Error"}

    # --- Prepare Data for Prompt ---
    essential_composition_cols = ['Symbol', 'Quantity', 'Cost Basis', 'Current Value']
    if not all(col in portfolio_df.columns for col in essential_composition_cols):
        return {"error": f"Portfolio DataFrame missing one or more essential columns: {essential_composition_cols}", "status": "Error"}

    optional_pl_cols = ['P/L $', 'P/L %']
    composition_cols = essential_composition_cols + [col for col in optional_pl_cols if col in portfolio_df.columns]

    composition_summary = portfolio_df[composition_cols].copy()
    for col in ['Quantity', 'Cost Basis', 'Current Value', 'P/L $', 'P/L %']:
        if col in composition_summary.columns:
             composition_summary[col] = pd.to_numeric(composition_summary[col], errors='coerce')

    def safe_format(value, fmt):
        try:
            return fmt.format(value) if pd.notna(value) else 'N/A'
        except (ValueError, TypeError):
            return 'N/A'

    if 'Cost Basis' in composition_summary: composition_summary['Cost Basis'] = composition_summary['Cost Basis'].apply(lambda x: safe_format(x, '${:,.2f}'))
    if 'Current Value' in composition_summary: composition_summary['Current Value'] = composition_summary['Current Value'].apply(lambda x: safe_format(x, '${:,.2f}'))
    if 'P/L $' in composition_summary: composition_summary['P/L $'] = composition_summary['P/L $'].apply(lambda x: safe_format(x, '${:,.2f}'))
    if 'P/L %' in composition_summary: composition_summary['P/L %'] = composition_summary['P/L %'].apply(lambda x: safe_format(x, '{:.2f}%'))
    if 'Quantity' in composition_summary: composition_summary['Quantity'] = composition_summary['Quantity'].apply(lambda x: safe_format(x, '{:,.4f}'))

    composition_summary_str = composition_summary.to_string(index=False, na_rep='N/A')

    analysis_summary_list = []
    for symbol, analysis in individual_analyses.items():
         if isinstance(analysis, dict) and analysis.get('status') == 'Completed':
              analysis_summary_list.append({
                   'Symbol': symbol,
                   'Sentiment': analysis.get('sentiment', 'N/A'),
                   'Outlook': analysis.get('outlook', 'N/A')
              })

    if not analysis_summary_list:
         return {"error": "No completed individual analyses found to generate overview.", "status": "Error"}

    analysis_summary_df = pd.DataFrame(analysis_summary_list)
    analysis_summary_str = analysis_summary_df.to_string(index=False, na_rep='N/A')

    total_portfolio_value = pd.to_numeric(portfolio_df['Current Value'], errors='coerce').sum()
    if pd.isna(total_portfolio_value): total_portfolio_value = 0

    # --- Construct the Prompt --- #
    prompt = f"""
Analyze the following investment portfolio based on its composition and the AI-generated sentiment analysis of its individual holdings. The total current value of the portfolio is approximately ${total_portfolio_value:,.2f}.

**Portfolio Composition:**
```
{composition_summary_str}
```

**Individual Holding Analysis Summary (Only includes successfully analyzed holdings):**
```
{analysis_summary_str}
```

**Analysis Tasks:**
1.  **Overall Risk Assessment:** Based on the portfolio's composition (e.g., concentration by value - compare individual holding 'Current Value' to total portfolio value, types of assets like stocks vs. crypto identifiable by symbol format like '-USD'), and the general sentiment/outlook of the holdings analyzed, provide a brief assessment of the portfolio's potential risk level (e.g., High, Medium, Low). Explain the key contributing factors based *only* on the provided data.
2.  **Diversification Comments:** Comment on the portfolio's diversification based *only* on the list of symbols and their 'Current Value'. Note any significant concentrations (e.g., any single asset representing >20% of the total portfolio value). Avoid making assumptions about sectors if not explicitly provided. Mention if only a subset of holdings could be analyzed for the summary.
3.  **Holistic Outlook:** Provide a synthesized outlook for the portfolio as a whole, considering the combined sentiment of its *analyzed* holdings and any notable concentrations or risks identified in the previous steps. Briefly summarize the key positive and negative factors influencing this outlook based *only* on the provided data. Acknowledge if the outlook is based on partial analysis.

**Output Format:**
Please provide the analysis clearly structured under the exact headings: "Overall Risk Assessment:", "Diversification Comments:", and "Holistic Outlook:". Be concise and focus on insights derived directly from the provided data. Do not give financial advice or make recommendations.
"""

    # --- Call the AI Model --- #
    try:
        print("--- Sending Portfolio Overview request to AI ---")
        print(f"DEBUG: Checking safety_settings_param within analyze_portfolio_overview: {safety_settings_param}")
        response = model.generate_content(prompt, safety_settings=safety_settings_param)
        response.resolve()
        analysis_text = response.text
        print("--- Received Portfolio Overview response from AI ---")

        # --- Parse the Response --- #
        parsed_analysis = {
            "Overall Risk Assessment": "N/A",
            "Diversification Comments": "N/A",
            "Holistic Outlook": "N/A",
            "raw_text": analysis_text,
            "status": "Parsing Error"
        }
        risk_match = re.search(r"Overall Risk Assessment:\s*(.*?)(?:Diversification Comments:|Holistic Outlook:|\Z)", analysis_text, re.IGNORECASE | re.DOTALL)
        if risk_match: parsed_analysis["Overall Risk Assessment"] = risk_match.group(1).strip()

        div_match = re.search(r"Diversification Comments:\s*(.*?)(?:Overall Risk Assessment:|Holistic Outlook:|\Z)", analysis_text, re.IGNORECASE | re.DOTALL)
        if div_match: parsed_analysis["Diversification Comments"] = div_match.group(1).strip()

        outlook_match = re.search(r"Holistic Outlook:\s*(.*?)(?:Overall Risk Assessment:|Diversification Comments:|\Z)", analysis_text, re.IGNORECASE | re.DOTALL)
        if outlook_match: parsed_analysis["Holistic Outlook"] = outlook_match.group(1).strip()

        if parsed_analysis["Overall Risk Assessment"] == "N/A" and parsed_analysis["Diversification Comments"] == "N/A" and parsed_analysis["Holistic Outlook"] == "N/A":
             print("Warning: Could not parse AI response structure for portfolio overview. Check raw_text.")
             parsed_analysis["error"] = "Failed to parse AI response structure. See raw_text."
             if not risk_match: parsed_analysis["Overall Risk Assessment"] = analysis_text
             parsed_analysis["status"] = "Parsing Error"
        else:
            parsed_analysis["status"] = "Completed"

        # del parsed_analysis["raw_text"]

        return parsed_analysis

    except Exception as e:
        print(f"Error during portfolio overview AI call: {e}")
        traceback.print_exc()
        error_message = f"AI portfolio overview failed: {type(e).__name__}"
        if hasattr(e, 'message') and e.message: error_message += f" - {e.message}"
        elif hasattr(e, 'args') and e.args: error_message += f" - {str(e.args)}"
        return {"error": error_message, "status": "Error", "Overall Risk Assessment": "N/A", "Diversification Comments": "N/A", "Holistic Outlook": "N/A"} 