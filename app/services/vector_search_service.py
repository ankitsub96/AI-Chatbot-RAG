import json
import faiss
import numpy as np
import faiss

from app.services.vector_service import embedding_model


def create_embedding(text: str):

    embedding = embedding_model.encode([text])

    embedding = np.array(embedding).astype("float32")

    faiss.normalize_L2(embedding)

    return embedding


def create_embeddings(
    texts: list[str],
    show_progress_bar: bool = False,
):

    embeddings = embedding_model.encode(
        texts,
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


def save_faiss_index(index, path: str):

    faiss.write_index(index, path)


def load_faiss_index(path: str):

    return faiss.read_index(path)
