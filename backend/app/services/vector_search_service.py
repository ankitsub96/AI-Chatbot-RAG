from sentence_transformers import SentenceTransformer
from sqlalchemy import func
from sqlmodel import Session, select
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from app.config.settings import EMBED_MODEL
from app.services.database import engine
from app.models.document_chunk import DocumentChunk

embedding_model = SentenceTransformer(EMBED_MODEL)


# =====================================================
# EMBEDDINGS
# =====================================================


def create_embedding(text: str):
    return embedding_model.encode(
        [text],
        normalize_embeddings=True,
    )


def create_embeddings(
    texts: list[str],
    batch_size: int = 8,
    show_progress_bar: bool = False,
):
    return embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
    )


# =====================================================
# ROW → DICT
# =====================================================


def _chunk_to_dict(row: DocumentChunk) -> dict:
    """Consistent serialization for all search functions."""
    return {
        "id": row.id,
        "document_id": row.document_id,
        "page": row.page,
        "section": row.section,
        "chunk_type": row.chunk_type,
        "parent_id": row.parent_id,
        "child_id": row.child_id,
        "chunk_index": row.chunk_index,
        "text": row.text,
        "chunk_metadata": row.chunk_metadata,
    }


# =====================================================
# VECTOR SEARCH (PGVECTOR)
# =====================================================


def vector_search(
    *,
    session: Session,
    query_embedding,
    document_id: str | None = None,
    top_k: int = 20,
):
    print("vector_search::")
    stmt = select(DocumentChunk)

    if document_id:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    stmt = stmt.order_by(
        DocumentChunk.embedding.op("<=>")(query_embedding[0].tolist())
    ).limit(top_k)

    rows = session.exec(stmt).all()

    return [
        {
            **_chunk_to_dict(row),
            "score": rank + 1,
        }
        for rank, row in enumerate(rows)
    ]


# =====================================================
# KEYWORD SEARCH (POSTGRES FTS)
# =====================================================


def keyword_search(
    *,
    session: Session,
    query: str,
    document_id: str | None = None,
    top_k: int = 20,
):
    ts_query = func.plainto_tsquery("simple", query)

    stmt = select(
        DocumentChunk,
        func.ts_rank(DocumentChunk.tsv, ts_query).label("score"),
    ).where(DocumentChunk.tsv.op("@@")(ts_query))

    if document_id:
        stmt = stmt.where(DocumentChunk.document_id == document_id)

    stmt = stmt.order_by(func.ts_rank(DocumentChunk.tsv, ts_query).desc()).limit(top_k)

    rows = session.exec(stmt).all()

    return [
        {
            **_chunk_to_dict(chunk),
            "score": float(score),
        }
        for chunk, score in rows
    ]


# =====================================================
# HYBRID SEARCH — single document_id (RRF)
# =====================================================


def hybrid_search(
    *,
    session: Session,
    query: str,
    query_embedding,
    document_id: str | None = None,
    top_k: int = 10,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
):
    print("hybrid_search::")

    vector_results = vector_search(
        session=session,
        query_embedding=query_embedding,
        document_id=document_id,
        top_k=top_k * 3,
    )

    keyword_results = keyword_search(
        session=session,
        query=query,
        document_id=document_id,
        top_k=top_k * 3,
    )

    vector_ranks = {row["id"]: rank + 1 for rank, row in enumerate(vector_results)}
    keyword_ranks = {row["id"]: rank + 1 for rank, row in enumerate(keyword_results)}

    all_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())

    rrf_k = 60

    merged = {}
    for row in vector_results:
        merged[row["id"]] = row
    for row in keyword_results:
        merged[row["id"]] = row

    fused_scores = {}
    for doc_id in all_ids:
        vector_score = (
            vector_weight * (1 / (rrf_k + vector_ranks[doc_id]))
            if doc_id in vector_ranks
            else 0
        )
        keyword_score = (
            keyword_weight * (1 / (rrf_k + keyword_ranks[doc_id]))
            if doc_id in keyword_ranks
            else 0
        )
        fused_scores[doc_id] = vector_score + keyword_score

    return [
        {
            **merged[doc_id],
            "hybrid_score": fused_scores[doc_id],
        }
        for doc_id in sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    ]


# =====================================================
# HYBRID SEARCH — single document_id (standalone, opens own session)
# =====================================================


def hybrid_document_search(
    document_id: str,
    query: str,
    query_embedding,
    top_k: int,
):
    print("ENTER hybrid_document_search")

    with Session(engine) as session:
        semantic_results = _semantic_document_search(
            session=session,
            document_id=document_id,
            query_embedding=query_embedding,
            top_k=top_k * 3,
        )

        keyword_results = _keyword_document_search(
            session=session,
            document_id=document_id,
            query=query,
            top_k=top_k * 3,
        )

    semantic_ranks = {
        item["id"]: rank for rank, item in enumerate(semantic_results, start=1)
    }
    keyword_ranks = {
        item["id"]: rank for rank, item in enumerate(keyword_results, start=1)
    }

    all_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

    rrf_k = 60
    semantic_map = {item["id"]: item for item in semantic_results}
    keyword_map = {item["id"]: item for item in keyword_results}

    merged = {}
    for chunk_id in all_ids:
        semantic_rrf = (
            0.6 * (1 / (rrf_k + semantic_ranks[chunk_id]))
            if chunk_id in semantic_ranks
            else 0
        )
        keyword_rrf = (
            0.4 * (1 / (rrf_k + keyword_ranks[chunk_id]))
            if chunk_id in keyword_ranks
            else 0
        )
        total_score = semantic_rrf + keyword_rrf
        base = semantic_map.get(chunk_id) or keyword_map.get(chunk_id)
        merged[chunk_id] = {
            "score": total_score,
            "semantic_score": semantic_rrf,
            "keyword_score": keyword_rrf,
            "data": base["data"],
        }

    results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    for rank, row in enumerate(results[:10], start=1):
        print(
            {
                "rank": rank,
                "page": row["data"].get("page"),
                "hybrid_score": row["score"],
                "semantic_score": row["semantic_score"],
                "keyword_score": row["keyword_score"],
            }
        )
        print(row["data"]["text"][:500])
        print("-" * 80)

    return results[:top_k]


# =====================================================
# HYBRID SEARCH — multiple document_ids (parallel)
# =====================================================


def hybrid_document_search_multiple(
    document_ids: list[str],
    query: str,
    query_embedding,
    top_k: int,
):
    def search_document(document_id: str):
        results = hybrid_document_search(
            document_id=document_id,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        for result in results:
            result["data"]["source_document_id"] = document_id
        return results

    worker_count = min(len(document_ids), 8)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        all_results = list(executor.map(search_document, document_ids))

    merged = []
    for result_set in all_results:
        merged.extend(result_set)

    merged.sort(key=lambda x: x["score"], reverse=True)

    return merged[:top_k]


# =====================================================
# INTERNAL HELPERS (used by hybrid_document_search)
# =====================================================


def _semantic_document_search(
    *,
    session: Session,
    document_id: str,
    query_embedding,
    top_k: int,
):
    rows = session.exec(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.embedding.cosine_distance(query_embedding[0].tolist()))
        .limit(top_k)
    ).all()

    return [
        {
            "id": row.id,
            "score": 1 / (rank + 1),
            "semantic_score": 1 / (rank + 1),
            "keyword_score": 0,
            "data": _chunk_to_dict(row),
        }
        for rank, row in enumerate(rows)
    ]


def _keyword_document_search(
    *,
    session: Session,
    document_id: str,
    query: str,
    top_k: int,
):
    tsquery = func.plainto_tsquery("simple", query)
    rank_expr = func.ts_rank(DocumentChunk.tsv, tsquery)

    rows = session.exec(
        select(DocumentChunk, rank_expr.label("rank_score"))
        .where(DocumentChunk.document_id == document_id)
        .where(DocumentChunk.tsv.op("@@")(tsquery))
        .order_by(rank_expr.desc())
        .limit(top_k)
    ).all()

    return [
        {
            "id": row.id,
            "score": float(rank_score),
            "semantic_score": 0,
            "keyword_score": float(rank_score),
            "data": _chunk_to_dict(row),
        }
        for row, rank_score in rows
    ]
