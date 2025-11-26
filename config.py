import os
from dotenv import load_dotenv
from datetime import date

# Load environment variables
load_dotenv()

# --- API KEYS ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# assert OPENAI_API_KEY, "⚠️ Please set OPENAI_API_KEY in your .env file"

# --- MODEL SETTINGS ---
OPENAI_MODEL = "gpt-4o-mini"
CONFIDENCE_THRESHOLD = 0.70

# --- DATA SETTINGS ---
# Define the assets to track
ASSETS = [
    {"symbol": "GOOGL", "name": "Alphabet (Google)"},
]

# GDELT Settings
DOC_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Prediction Horizons (Key: Days)
HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
}

# --- CENTRALIZED DATE SETTINGS ---
# The date the news analysis starts.
DATA_START_DATE = date(2025, 6, 1)

# The last date the LLM will analyze news for (the end of your prediction period).
NEWS_PREDICTION_END_DATE = date(2025, 9, 30) 

# The date price data must extend to, required to evaluate the final 1m prediction.
# (NEWS_PREDICTION_END_DATE + ~21 trading days)
PRICE_COLLECTION_END_DATE = date(2025, 10, 30)

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_warehouse")
RESULTS_DIR = os.path.join(BASE_DIR, "results")