OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen2.5:7b"
CONTRARIAN_MODEL = "qwen2.5:7b"

N_AGENTS = 3
TEMPERATURE = 0.7

MAX_DEBATE_ROUNDS = 2

CONDITIONS = [
    "solo_baseline",
    "group_deliberation",
    "group_with_contrarian",
]

QUESTIONS_PATH = "data/questions.json"
RESULTS_DIR = "results/raw"