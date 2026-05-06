from sentence_transformers import SentenceTransformer

from app.config.settings import EMBED_MODEL


embedding_model = SentenceTransformer(
    EMBED_MODEL
)