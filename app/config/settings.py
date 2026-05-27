import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = os.getenv("MODEL")

REDIS_HOST = os.getenv("REDIS_HOST")

REDIS_PORT = int(os.getenv("REDIS_PORT"))

UPLOAD_DIR = "app/uploads"

VECTOR_DIR = "app/vector_store"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

TOP_K = 30

CHUNK_SIZE = 400
