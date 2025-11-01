import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 


# ###############################################
#  ------------- Core configuration ---
#################################################

LANGUAGE_NAME = os.getenv("LANGUAGE_NAME", "Bambara")
CACHE_DIR = Path(os.getenv("TRANSLATION_CACHE", "instruction_generation_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# #########################################################
#  Timing, retry,Performance tuning and timeout settings---
###########################################################
try:
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))
    BATCH_TIMEOUT = int(os.getenv("BATCH_TIMEOUT", 1200))
    CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 10))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 20))
    TIMEOUT = float(os.getenv("TIMEOUT", 600.0))
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 20))
    CONCURRENT_BATCHES = int(os.getenv("CONCURRENT_BATCHES", 20))
    RATE_LIMITER = int(os.getenv("RATE_LIMITER", 100))
except ValueError as e:
    raise ValueError(f"Invalid numeric environment variable: {e}")

###################################
# --- Model & API configuration ---
###################################
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
BASE_URL = os.getenv("BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
TOP_P = float(os.getenv("TOP_P", 0.9))

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not set")

# ################################################
# ------- Linguistic resources files -----------
##################################################
GLOSSARY_FILE = os.getenv("GLOSSARY_FILE", "linguistic_resources/glossary.json")
GRAMMAR_RULE_FILE = os.getenv("GRAMMAR_RULE_FILE", "linguistic_resources/grammar_rule.json")
CONLLU_FILE = os.getenv("CONLLU_FILE", "linguistic_resources/bm_crb-ud-test.conllu")

# ################################################
# ------- dataset files [jsonl] -----------
##################################################
DATASETS_FOLDER = os.getenv("DATASETS_FOLDER", "datasets")

for f in [GLOSSARY_FILE, GRAMMAR_RULE_FILE, CONLLU_FILE]:
    if not Path(f).exists():
        print(f"[WARN] Missing file: {f}")