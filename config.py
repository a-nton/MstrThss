import os
from dotenv import load_dotenv

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
    {"symbol": "META", "name": "Meta"},
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "ORCL", "name": "Oracle"},
]

# GDELT Settings
DOC_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Prediction Horizons (Key: Days)
HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
}

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_warehouse")
RESULTS_DIR = os.path.join(BASE_DIR, "results")