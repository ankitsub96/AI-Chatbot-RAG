import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = os.getenv("MODEL")

REDIS_HOST = os.getenv("REDIS_HOST")

REDIS_PORT = int(os.getenv("REDIS_PORT"))

UPLOAD_DIR = BASE_DIR / "uploads"

VECTOR_DIR = BASE_DIR / "app/vector_store"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

TOP_K = 10

CHUNK_SIZE = 400
MAX_RETRIES: int = 2
CONFIDENCE_THRESHOLD: int = 7

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", 5432))

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


REACT_MAX_ITERATIONS: int = 5
PLANNER_MAX_STEPS: int = 4
WEB_SEARCH_MAX_RESULTS: int = 5
