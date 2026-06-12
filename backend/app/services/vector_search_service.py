from sentence_transformers import SentenceTransformer
from sqlalchemy import func
from sqlmodel import Session, select
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from app.config.settings import EMBED_MODEL
from app.services.database import engine
from app.models.document_chunk import DocumentChunk
from app.utils.helpers import timer

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
    document_ids: list[str] | None = None,
    top_k: int = 20,
):
    print("vector_search::")
    stmt = select(DocumentChunk)

    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

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
    document_ids: list[str] | None = None,
    top_k: int = 20,
):
    ts_query = func.plainto_tsquery("simple", query)

    stmt = select(
        DocumentChunk,
        func.ts_rank(DocumentChunk.tsv, ts_query).label("score"),
    ).where(DocumentChunk.tsv.op("@@")(ts_query))

    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

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


@timer
def hybrid_search(
    *,
    query: str,
    query_embedding,
    document_ids: list[str] | None = None,
    top_k: int = 10,
    vector_weight: float = 0.6,
    keyword_weight: float = 0.4,
):
    print("hybrid_search::")

    @timer
    def run_vector():
        with Session(engine) as session:
            return vector_search(
                session=session,
                query_embedding=query_embedding,
                document_ids=document_ids,
                top_k=top_k * 3,
            )

    @timer
    def run_keyword():
        with Session(engine) as session:
            return keyword_search(
                session=session,
                query=query,
                document_ids=document_ids,
                top_k=top_k * 3,
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(run_vector)
        keyword_future = executor.submit(run_keyword)

        vector_results = vector_future.result()
        keyword_results = keyword_future.result()

    # Reciprocal Rank Fusion (RRF):
    # Combines vector and keyword search rankings using:
    #
    #     score = weight * (1 / (k + rank))
    #
    # Documents appearing in both result sets receive
    # contributions from both retrievers and are ranked higher.
    # RRF uses rank positions instead of raw similarity scores,
    # making it robust across different retrieval methods.
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


# =====================================================
# PARENT-AWARE RETRIEVAL (Phase 5)
# =====================================================


def group_chunks_by_parent(results: list[dict]) -> dict[str, dict]:
    """
    Group child chunk results by parent_id.

    Returns:
        {
            parent_id: {
                "parent_id": str,
                "score": float,          # max child score (Step 5.3)
                "chunks": [chunk_dict]   # all matched children
            }
        }
    """
    grouped = defaultdict(lambda: {"score": 0.0, "chunks": []})

    for result in results:
        data = result.get("data", result)
        parent_id = data.get("parent_id")

        if not parent_id:
            # chunk has no parent — treat itself as a standalone group
            parent_id = f"standalone_{data.get('id')}"

        score = result.get("score", result.get("hybrid_score", 0.0))

        # parent score = max child score (Step 5.3)
        if score > grouped[parent_id]["score"]:
            grouped[parent_id]["score"] = score

        grouped[parent_id]["parent_id"] = parent_id
        grouped[parent_id]["chunks"].append(data)

    return dict(grouped)


def select_top_parents(
    grouped: dict[str, dict],
    top_n: int,
) -> list[dict]:
    """
    Select top N parents ranked by their max child score (Step 5.4).
    """
    ranked = sorted(
        grouped.values(),
        key=lambda x: x["score"],
        reverse=True,
    )
    return ranked[:top_n]


@timer
def expand_parent_chunks(
    *,
    session: Session,
    parent_ids: list[str],
    document_ids: list[str] | None = None,
) -> dict[str, list[dict]]:
    """
    For each selected parent_id, retrieve ALL its children from DB (Step 5.5).
    Returns them sorted by chunk_index for coherent reading order.

    This is context expansion — we searched with children,
    but answer using the full parent window.
    """
    stmt = select(DocumentChunk).where(DocumentChunk.parent_id.in_(parent_ids))

    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))

    # sort by chunk_index so siblings are in reading order
    stmt = stmt.order_by(DocumentChunk.chunk_index)

    rows = session.exec(stmt).all()

    expanded: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        expanded[row.parent_id].append(_chunk_to_dict(row))

    return dict(expanded)


def build_parent_context_blocks(
    top_parents: list[dict],
    expanded: dict[str, list[dict]],
) -> str:
    """
    Build structured context string for LLM (Step 5.6).

    Produces:

        [PARENT: parent_0_page_1]
        [PAGE: 1] [SCORE: 0.0123]

          chunk text 1...

          chunk text 2...

        [PARENT: parent_1_page_3]
        ...
    """
    blocks = []

    for parent in top_parents:
        parent_id = parent["parent_id"]
        children = expanded.get(parent_id, parent["chunks"])

        # sort children by chunk_index if available
        children = sorted(
            children,
            key=lambda c: c.get("chunk_index") or 0,
        )

        page = children[0].get("page", "?") if children else "?"

        header = (
            f"[PARENT: {parent_id}]\n" f"[PAGE: {page}] [SCORE: {parent['score']:.6f}]"
        )

        body = "\n\n".join(f"  {child['text']}" for child in children)

        blocks.append(f"{header}\n\n{body}")

    return "\n\n" + ("\n\n" + "─" * 60 + "\n\n").join(blocks) + "\n\n"
