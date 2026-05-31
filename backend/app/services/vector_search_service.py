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
    embedding = embedding_model.encode(
        [text],
        normalize_embeddings=True,
    )

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
        normalize_embeddings=True,
    )

    return embeddings


# =====================================================
# VECTOR SEARCH (PGVECTOR)
# =====================================================


def vector_search(
    *,
    session: Session,
    query_embedding,
    filename: str | None = None,
    top_k: int = 20,
):
    print("vector_search::")
    stmt = select(DocumentChunk)

    if filename:
        stmt = stmt.where(
            DocumentChunk.filename == filename,
        )

    stmt = stmt.order_by(
        DocumentChunk.embedding.op("<=>")(query_embedding[0].tolist())
    ).limit(top_k)

    rows = session.exec(stmt).all()

    results = []

    for rank, row in enumerate(rows):
        results.append(
            {
                "id": row.id,
                "filename": row.filename,
                "page": row.page,
                "section": row.section,
                "chunk_type": row.chunk_type,
                "text": row.text,
                "chunk_metadata": row.chunk_metadata,
                "score": rank + 1,
            }
        )

    return results


# =====================================================
# KEYWORD SEARCH (POSTGRES FTS)
# =====================================================


def keyword_search(
    *,
    session: Session,
    query: str,
    filename: str | None = None,
    top_k: int = 20,
):
    ts_query = func.plainto_tsquery(
        "simple",
        query,
    )

    stmt = select(
        DocumentChunk,
        func.ts_rank(
            DocumentChunk.tsv,
            ts_query,
        ).label("score"),
    ).where(DocumentChunk.tsv.op("@@")(ts_query))

    if filename:
        stmt = stmt.where(
            DocumentChunk.filename == filename,
        )

    stmt = stmt.order_by(
        func.ts_rank(
            DocumentChunk.tsv,
            ts_query,
        ).desc()
    ).limit(top_k)

    rows = session.exec(stmt).all()

    results = []

    for chunk, score in rows:
        results.append(
            {
                "id": chunk.id,
                "filename": chunk.filename,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type,
                "text": chunk.text,
                "chunk_metadata": chunk.chunk_metadata,
                "score": float(score),
            }
        )

    return results


# =====================================================
# HYBRID SEARCH (RRF)
# =====================================================


def hybrid_search(
    *,
    session: Session,
    query: str,
    query_embedding,
    filename: str | None = None,
    top_k: int = 10,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
):
    print("hybrid_search::")
    vector_results = vector_search(
        session=session,
        query_embedding=query_embedding,
        filename=filename,
        top_k=top_k * 3,
    )

    keyword_results = keyword_search(
        session=session,
        query=query,
        filename=filename,
        top_k=top_k * 3,
    )

    vector_ranks = {row["id"]: rank + 1 for rank, row in enumerate(vector_results)}

    keyword_ranks = {row["id"]: rank + 1 for rank, row in enumerate(keyword_results)}

    all_ids = set(vector_ranks.keys()) | set(keyword_ranks.keys())

    rrf_k = 60

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

    merged = {}

    for row in vector_results:
        merged[row["id"]] = row

    for row in keyword_results:
        merged[row["id"]] = row

    return [
        {
            **merged[doc_id],
            "hybrid_score": fused_scores[doc_id],
        }
        for doc_id in sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]
    ]


def semantic_document_search(
    filename: str,
    query_embedding,
    top_k: int,
):
    print("ENTER semantic_document_search")
    print(type(query_embedding))
    print(query_embedding.shape)

    print(type(query_embedding[0]))
    print(query_embedding[0].shape)

    print(type(query_embedding[0].tolist()))
    print(len(query_embedding[0].tolist()))

    # vector = [float(x) for x in query_embedding[0]]

    with Session(engine) as session:

        rows = session.exec(
            select(DocumentChunk)
            .where(DocumentChunk.filename == filename)
            .order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding[0].tolist())
                # DocumentChunk.embedding.cosine_distance(vector)
            )
            .limit(top_k)
        ).all()

        results = []

        for rank, row in enumerate(rows, start=1):

            score = 1 / rank

            results.append(
                {
                    "id": row.id,
                    "score": score,
                    "semantic_score": score,
                    "keyword_score": 0,
                    "data": row.model_dump(),
                }
            )

        return results


def keyword_document_search(
    filename: str,
    query: str,
    top_k: int,
):
    print("ENTER keyword_document_search")
    with Session(engine) as session:

        tsquery = func.plainto_tsquery(
            "simple",
            query,
        )

        rank_expr = func.ts_rank(
            DocumentChunk.tsv,
            tsquery,
        )

        rows = session.exec(
            select(
                DocumentChunk,
                rank_expr.label("rank_score"),
            )
            .where(DocumentChunk.filename == filename)
            .where(DocumentChunk.tsv.op("@@")(tsquery))
            .order_by(rank_expr.desc())
            .limit(top_k)
        ).all()

        results = []

        for row, rank_score in rows:

            results.append(
                {
                    "id": row.id,
                    "score": float(rank_score),
                    "semantic_score": 0,
                    "keyword_score": float(rank_score),
                    "data": row.model_dump(),
                }
            )

        return results


def keyword_document_search(
    filename: str,
    query: str,
    top_k: int,
):
    with Session(engine) as session:

        tsquery = func.plainto_tsquery(
            "simple",
            query,
        )

        rank_expr = func.ts_rank(
            DocumentChunk.tsv,
            tsquery,
        )

        rows = session.exec(
            select(
                DocumentChunk,
                rank_expr.label("rank_score"),
            )
            .where(DocumentChunk.filename == filename)
            .where(DocumentChunk.tsv.op("@@")(tsquery))
            .order_by(rank_expr.desc())
            .limit(top_k)
        ).all()

        results = []

        for row, rank_score in rows:

            results.append(
                {
                    "id": row.id,
                    "score": float(rank_score),
                    "semantic_score": 0,
                    "keyword_score": float(rank_score),
                    "data": row.model_dump(),
                }
            )

        return results


def hybrid_document_search(
    filename: str,
    query: str,
    query_embedding,
    top_k: int,
):
    print("ENTER hybrid_document_search")
    semantic_results = semantic_document_search(
        filename=filename,
        query_embedding=query_embedding,
        top_k=top_k * 3,
    )

    keyword_results = keyword_document_search(
        filename=filename,
        query=query,
        top_k=top_k * 3,
    )

    semantic_ranks = {}

    for rank, item in enumerate(
        semantic_results,
        start=1,
    ):
        semantic_ranks[item["id"]] = rank

    keyword_ranks = {}

    for rank, item in enumerate(
        keyword_results,
        start=1,
    ):
        keyword_ranks[item["id"]] = rank

    all_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

    rrf_k = 60

    merged = {}

    semantic_map = {item["id"]: item for item in semantic_results}

    keyword_map = {item["id"]: item for item in keyword_results}

    for chunk_id in all_ids:

        semantic_rrf = 0
        keyword_rrf = 0

        if chunk_id in semantic_ranks:

            semantic_rrf = 0.6 * (1 / (rrf_k + semantic_ranks[chunk_id]))

        if chunk_id in keyword_ranks:

            keyword_rrf = 0.4 * (1 / (rrf_k + keyword_ranks[chunk_id]))

        total_score = semantic_rrf + keyword_rrf

        base_result = semantic_map.get(chunk_id) or keyword_map.get(chunk_id)

        merged[chunk_id] = {
            "score": total_score,
            "semantic_score": semantic_rrf,
            "keyword_score": keyword_rrf,
            "data": base_result["data"],
        }

    results = sorted(
        merged.values(),
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]


def hybrid_document_search_multiple(
    filenames: list[str],
    query: str,
    query_embedding,
    top_k: int,
):
    def search_file(filename):

        results = hybrid_document_search(
            filename=filename,
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        for result in results:

            result["data"]["source_file"] = filename

        return results

    worker_count = min(
        len(filenames),
        8,
    )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:

        all_results = list(
            executor.map(
                search_file,
                filenames,
            )
        )

    merged = []

    for result_set in all_results:

        merged.extend(result_set)

    merged.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return merged[:top_k]
