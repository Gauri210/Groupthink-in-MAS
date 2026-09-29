import os

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "phi4-mini"
CONTRARIAN_MODEL = "phi4-mini"

N_AGENTS = 3
TEMPERATURE = 0.7

JUDGE_MODEL = "gemma2:9b"
JUDGE_TEMPERATURE = 0

MAX_DEBATE_ROUNDS = 1

CONDITIONS = [
    "solo_baseline",
]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS_PATH = os.path.join(_PROJECT_ROOT, "data", "questions.json")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results", "raw")