import json
import faiss
import numpy as np
import pickle
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config.settings import EMBED_MODEL
from app.utils.helpers import timer

embedding_model = SentenceTransformer(EMBED_MODEL)


def create_embedding(text: str):

    embedding = embedding_model.encode([text])

    embedding = np.array(embedding).astype("float32")

    faiss.normalize_L2(embedding)

    return embedding


def create_embeddings(
    texts: list[str],
    batch_size: int = 8,
    show_progress_bar: bool = False,
):
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    return embeddings


def load_index_and_metadata(index_path: str, metadata_path: str):

    index = load_faiss_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:

        metadata = json.load(f)

    return index, metadata


def semantic_search(
    *,
    index,
    metadata,
    query_embedding,
    top_k: int,
):

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx >= len(metadata):

            continue

        results.append({"score": float(distance), "data": metadata[idx]})

    return results


@timer
def create_hnsw_index(
    embeddings,
    hnsw_m: int = 32,
    ef_construction: int = 200,
):

    dimension = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(
        dimension,
        hnsw_m,
    )

    index.hnsw.efConstruction = ef_construction

    index.metric_type = faiss.METRIC_INNER_PRODUCT

    index.add(embeddings)

    return index


def create_flat_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


@timer
def save_faiss_index(index, path: str):

    faiss.write_index(index, path)


def load_faiss_index(path: str):

    return faiss.read_index(path)


# =========================
# BM25 INDEX
# =========================


def build_bm25_index(documents: list[dict]) -> BM25Okapi:
    """
    Build BM25 index from document chunks.
    Tokenizes each chunk's text by whitespace.
    """
    corpus = [doc["text"].lower().split() for doc in documents]
    return BM25Okapi(corpus)


def save_bm25_index(index: BM25Okapi, path: str):
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_bm25_index(path: str) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================
# HYBRID SEARCH
# =========================


def hybrid_search(
    *,
    faiss_index,
    bm25_index: BM25Okapi,
    metadata: list[dict],
    query_embedding,
    query: str,
    top_k: int,
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """
    Combines FAISS dense search and BM25 keyword search
    using Reciprocal Rank Fusion (RRF).

    faiss_weight + bm25_weight should equal 1.0.
    Tune faiss_weight higher for semantic queries,
    bm25_weight higher for keyword/name-heavy queries.
    """

    n = len(metadata)
    retrieve_k = min(top_k * 3, n)  # retrieve more, rerank down to top_k

    # =========================
    # FAISS DENSE SEARCH
    # =========================

    distances, indices = faiss_index.search(query_embedding, retrieve_k)

    faiss_ranks = {}

    for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
        if idx < n:
            faiss_ranks[idx] = {
                "rank": rank + 1,
                "score": float(score),
            }

    # =========================
    # BM25 KEYWORD SEARCH
    # =========================

    tokenized_query = query.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked = sorted(
        range(n),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:retrieve_k]

    bm25_ranks = {
        idx: {"rank": rank + 1, "score": float(bm25_scores[idx])}
        for rank, idx in enumerate(bm25_ranked)
    }

    # =========================
    # RECIPROCAL RANK FUSION
    # =========================

    rrf_k = 60  # standard RRF constant

    all_indices = set(faiss_ranks.keys()) | set(bm25_ranks.keys())

    fused_scores = {}

    for idx in all_indices:
        faiss_rrf = (
            faiss_weight * (1 / (rrf_k + faiss_ranks[idx]["rank"]))
            if idx in faiss_ranks
            else 0
        )
        bm25_rrf = (
            bm25_weight * (1 / (rrf_k + bm25_ranks[idx]["rank"]))
            if idx in bm25_ranks
            else 0
        )
        fused_scores[idx] = faiss_rrf + bm25_rrf

    # =========================
    # SORT AND RETURN TOP K
    # =========================

    top_indices = sorted(
        fused_scores,
        key=lambda i: fused_scores[i],
        reverse=True,
    )[:top_k]

    return [
        {
            "score": fused_scores[idx],
            "faiss_score": faiss_ranks.get(idx, {}).get("score", 0),
            "bm25_score": bm25_ranks.get(idx, {}).get("score", 0),
            "data": metadata[idx],
        }
        for idx in top_indices
    ]
