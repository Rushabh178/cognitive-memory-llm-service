# ============================================================
# FILE: config.py
#
# WHAT THIS FILE DOES:
# Reads all configuration values from environment variables (loaded from a .env file).
# Acts as the single source of truth for every setting used across the project.
# Raises clear errors at startup if required secrets are missing, so the app
# never starts in a broken state.
#
# KEY CONCEPT FOR LEARNING:
# Twelve-Factor App principle: configuration lives in the environment, not the code.
# This means you can deploy the same code to dev/staging/prod just by changing
# the .env file — no code changes needed, no secrets accidentally committed to git.
#
# WHERE IT FITS IN THE PIPELINE:
# .env file → config.py (loads & validates) → every other module imports from here
# ============================================================

import os
from dotenv import load_dotenv

# Load the .env file into environment variables so os.getenv() can read them.
# If the file doesn't exist, this is a no-op — values must then come from
# the real environment (e.g. Docker, Kubernetes, or the host shell).
load_dotenv()

# --- LLM provider settings ---

# Groq API key — required. Get a free key from https://console.groq.com
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Which Groq model to use.
# llama-3.1-8b-instant is fast and free-tier friendly.
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# LLM_MODEL is an alias for GROQ_MODEL so the rest of the codebase
# (main.py log line, ai_service.py call) can reference a single name.
LLM_MODEL: str = GROQ_MODEL

# Maximum number of tokens the LLM is allowed to generate in one response.
# 1024 tokens ≈ 750 words — plenty for conversational replies.
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Temperature controls randomness: 0.0 = deterministic, 1.0 = very creative.
# 0.7 is a good balance for conversational AI.
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# --- ChromaDB / vector store settings ---

# Path on disk where ChromaDB will persist its data between restarts.
CHROMA_PERSIST_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "./chroma_data")

# --- Embedding model settings ---

# The sentence-transformers model used to convert text into vectors.
# all-MiniLM-L6-v2 is small (80 MB), fast, and good enough for semantic search.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Graph memory settings ---

# Path on disk where per-user graph JSON files are stored.
GRAPH_PERSIST_PATH: str = os.getenv("GRAPH_PERSIST_PATH", "./graph_data")

# --- PostgreSQL connection for graph storage ---
# Uses the same database as Spring Boot
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "cognitive_memory")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

# --- API security ---

# The bearer token that the Spring Boot API sends in every request header.
API_BEARER_TOKEN: str = os.getenv("API_BEARER_TOKEN", "")

# --- Server settings ---

PORT: int = int(os.getenv("PORT", "8000"))

# --- Startup validation ---
# Check for required secrets BEFORE the app accepts any traffic.

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set. "
        "Get a free key from https://console.groq.com "
        "and add it to your .env file."
    )

if not API_BEARER_TOKEN:
    raise ValueError(
        "API_BEARER_TOKEN is not set. "
        "Add it to your .env file or set it as an environment variable. "
        "This token is used to authenticate requests from the Spring Boot API."
    )
