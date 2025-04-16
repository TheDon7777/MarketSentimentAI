import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

# Determine the base directory of the project (market_sentiment_ai)
# This assumes config_manager.py is in market_sentiment_ai/src/
BASE_DIR = Path(__file__).resolve().parent.parent

# Construct paths to config files relative to the base directory
ENV_PATH = BASE_DIR / '.env'
CONFIG_PATH = BASE_DIR / 'config.yaml'

# Load environment variables from .env file
load_dotenv(dotenv_path=ENV_PATH)

# --- Configuration Loading Functions ---

def load_yaml_config() -> dict:
    """Loads the configuration from the config.yaml file."""
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, 'r') as stream:
            config = yaml.safe_load(stream)
            if config is None: # Handle empty YAML file
                return {}
            return config
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        raise # Re-raise the exception after printing
    except Exception as e:
        print(f"An unexpected error occurred while reading {CONFIG_PATH}: {e}")
        raise # Re-raise the exception

def get_api_key(key_name: str) -> str | None:
    """Retrieves an API key from environment variables."""
    return os.getenv(key_name)

# --- Accessor Functions ---

# Load config once when the module is imported
_config = load_yaml_config()

def get_config_parameter(section: str, parameter: str, default=None):
    """
    Retrieves a specific parameter from the loaded YAML configuration.

    Args:
        section: The top-level section in the YAML file (e.g., 'analysis_parameters').
        parameter: The specific parameter key within the section.
        default: The value to return if the parameter is not found. Defaults to None.

    Returns:
        The value of the parameter or the default value.
    """
    return _config.get(section, {}).get(parameter, default)

def get_market_indices() -> list[str]:
    """Returns the list of market indices from the config."""
    return _config.get('market_indices', [])

def get_stocks_to_track() -> list[str]:
    """Returns the list of stocks to track from the config."""
    return _config.get('stocks_to_track', [])

# Example usage (optional, can be removed or kept for testing)
if __name__ == '__main__':
    print(f"Base Directory: {BASE_DIR}")
    print(f"Checking for .env at: {ENV_PATH}")
    print(f"Checking for config.yaml at: {CONFIG_PATH}")

    # news_key = get_api_key('NEWS_API_KEY') # Removed - No longer using NewsAPI
    google_key = get_api_key('GOOGLE_AI_API_KEY')
    # print(f"News API Key Loaded: {'Yes' if news_key else 'No'}") # Removed
    print(f"Google AI Key Loaded: {'Yes' if google_key else 'No'}")

    indices = get_market_indices()
    stocks = get_stocks_to_track()
    sentiment_model = get_config_parameter('analysis_parameters', 'sentiment_model')

    print(f"Market Indices: {indices}")
    print(f"Stocks to Track: {stocks}")
    print(f"Sentiment Model: {sentiment_model}")
    print("Config Parameters:", _config) 