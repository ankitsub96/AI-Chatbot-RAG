import os

os.environ["UNSTRUCTURED_SKIP_TORCH"] = "1"
import json
import asyncio
from fastapi import HTTPException, BackgroundTasks
import numpy as np
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from sqlmodel import Session, select
from sqlalchemy import text

logging.getLogger("pdfminer").setLevel(logging.ERROR)

from app.config.settings import TOP_K, CHUNK_SIZE, UPLOAD_DIR
from app.services.llm_service import generate_response
from app.utils.file_utils import get_index_path, get_metadata_path, load_documents
from app.utils.helpers import timer
from app.services.semantic_cache_service import (
    get_exact_cache,
    get_semantic_cache,
    set_exact_cache,
    set_semantic_cache,
)
from app.services.memory_service import (
    retrieve_relevant_memories,
    save_conversation_turn,
    load_summary,
)
from app.services.vector_search_service import (
    create_embedding,
    # load_index_and_metadata,
    # semantic_search,
    create_embeddings,
    # create_hnsw_index,
    # save_faiss_index,
    # load_faiss_index,
    hybrid_document_search_multiple,
)
from app.services.pdf_service import (
    group_pages,
    chunk_pages,
    process_document_pages,
    extract_pdf_text,
)

# from app.services.vector_search_service import (
# build_bm25_index,
# save_bm25_index,
# load_bm25_index,
# hybrid_search,
# )
from app.utils.file_utils import get_bm25_path
from app.utils.helpers import thinking
from app.services.database import engine
from app.models.document_chunk import DocumentChunk
from app.models.document import Document
from app.models.session_document import SessionDocument

# =========================
# EMBEDDING CONFIG
# =========================

EMBEDDING_BATCH_SIZE = 8  # reduce if laptop struggles (try 4)
CHECKPOINT_EVERY = 50  # embed and report every N chunks

MAX_CONCURRENT_EMBEDDING_BATCHES = 1
INSERT_BATCH_SIZE = 50

# =========================
# BUILD VECTOR DATABASE
# =========================


@timer
@timer
async def build_vector_database(document_id: str, stored_filename: str):
    pdf_path = UPLOAD_DIR / stored_filename

    print({"pdf_path": pdf_path})
    print("\n" + "=" * 80)
    print("BUILDING VECTOR DATABASE (PARENT-CHILD)")
    print("=" * 80)

    pages = await asyncio.to_thread(extract_pdf_text, pdf_path)
    grouped_pages = await asyncio.to_thread(group_pages, pages)
    documents = await asyncio.to_thread(chunk_pages, grouped_pages)

    print(f"Child chunks created: {len(documents)}")

    texts = [f"""
Represent this document for retrieval.

Parent ID:
{doc.get('parent_id', '')}

Child ID:
{doc.get('child_id', '')}

Page:
{doc.get('page', '')}

Section:
{doc.get('section', '')}

Content:
{doc['text']}
""" for doc in documents]

    total = len(texts)
    inserted = 0

    # delete any existing chunks for this document (safe re-index)
    with Session(engine) as session:
        session.execute(
            text("DELETE FROM document_chunks WHERE document_id = :document_id"),
            {"document_id": document_id},
        )
        session.commit()

    with Session(engine) as session:
        for start in range(0, total, CHECKPOINT_EVERY):
            end = min(start + CHECKPOINT_EVERY, total)

            print(f"\nEmbedding chunks {start + 1}-{end} of {total}")

            batch_texts = texts[start:end]
            batch_docs = documents[start:end]

            embeddings = await asyncio.to_thread(
                create_embeddings,
                batch_texts,
                EMBEDDING_BATCH_SIZE,
                True,
            )

            rows = []
            for chunk_index, (doc, embedding) in enumerate(
                zip(batch_docs, embeddings), start=start
            ):
                rows.append(
                    DocumentChunk(
                        document_id=document_id,
                        page=doc.get("page"),
                        section=doc.get("section"),
                        chunk_type=doc.get("type"),
                        parent_id=doc.get("parent_id"),
                        child_id=doc.get("child_id"),
                        chunk_index=chunk_index,
                        text=doc["text"],
                        chunk_metadata=doc,
                        embedding=embedding.tolist(),
                    )
                )

            for i in range(0, len(rows), INSERT_BATCH_SIZE):
                session.add_all(rows[i : i + INSERT_BATCH_SIZE])

            session.commit()
            inserted += len(rows)

            print(f"Inserted {inserted}/{total} chunks")

            del embeddings
            del rows

        # update TSV for full-text search
        session.execute(
            text("""
                UPDATE document_chunks
                SET tsv = to_tsvector('simple', text)
                WHERE document_id = :document_id
            """),
            {"document_id": document_id},
        )
        session.commit()

    # mark document as ready
    with Session(engine) as session:
        doc_row = session.get(Document, document_id)
        if doc_row:
            doc_row.status = "ready"
            doc_row.chunk_count = inserted
            doc_row.page_count = len(set(d.get("page") for d in documents))
            session.add(doc_row)
            session.commit()

    print("\n" + "=" * 80)
    print("VECTOR DATABASE READY (PARENT-CHILD)")
    print("=" * 80)
    print(f"Stored chunks: {inserted}")


# =========================
# LIST READY DOCS
# =========================


def list_ready_documents():
    """Return all documents that have been fully indexed."""
    with Session(engine) as session:
        rows = session.exec(select(Document).where(Document.status == "ready")).all()

        return [
            {
                "document_id": doc.id,
                "original_filename": doc.original_filename,
                "stored_filename": doc.stored_filename,
                "checksum": doc.checksum,
                "page_count": doc.page_count,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at.isoformat(),
            }
            for doc in rows
        ]


# =========================
# ASK DOCUMENT
# =========================


async def ask_document(
    session_id: str,
    question: str,
    background_tasks: BackgroundTasks,
    document_ids: list[str] | None = None,  # None = all docs in session
):
    print("\n" + "=" * 80)
    print("NEW QUESTION")
    print("=" * 80)
    print(
        {
            "session_id": session_id,
            "question": question,
            "document_ids_filter": document_ids,
        }
    )

    # =========================
    # RESOLVE DOCUMENT IDS
    # =========================

    if document_ids:
        resolved_ids = document_ids
    else:
        with Session(engine) as db:
            rows = db.exec(
                select(SessionDocument.document_id).where(
                    SessionDocument.session_id == session_id
                )
            ).all()
            resolved_ids = list(rows)

    if not resolved_ids:
        return "No documents found for this session."

    # =========================
    # CACHE KEY
    # =========================

    cache_key = session_id + "|" + "|".join(sorted(resolved_ids))

    # =========================
    # EXACT CACHE
    # =========================

    exact_cached = get_exact_cache(cache_key, question)
    if exact_cached:
        print("\nEXACT CACHE HIT")
        return exact_cached

    # =========================
    # QUERY EMBEDDING
    # =========================

    print("\nGenerating query embedding...")
    query_embedding = await asyncio.to_thread(create_embedding, question)

    # =========================
    # SEMANTIC CACHE
    # =========================

    semantic_cached = get_semantic_cache(cache_key, query_embedding)
    if semantic_cached:
        print("\nSEMANTIC CACHE HIT")
        return semantic_cached

    # =========================
    # PARALLEL: MEMORY + SEARCH
    # =========================

    memory_task = asyncio.to_thread(
        retrieve_relevant_memories,
        session_id,
        question,
    )

    document_task = asyncio.to_thread(
        hybrid_document_search_multiple,
        resolved_ids,
        question,
        query_embedding,
        TOP_K,
    )

    memory_context, results = await asyncio.gather(memory_task, document_task)

    print("\n[document_loaded] Documents searched")

    # =========================
    # SEARCH RESULTS
    # =========================

    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    context = ""

    for rank, result in enumerate(results, start=1):
        score = result["score"]
        doc = result["data"]

        print(
            {
                "rank": rank,
                "score": score,
                "page": doc.get("page"),
                "document_id": doc.get("document_id"),
                "parent_id": doc.get("parent_id"),
            }
        )

        context += f"""
[DOCUMENT ID: {doc.get('document_id')}]
[PAGE: {doc.get('page')}]

{doc.get('text')}

"""

    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are a helpful document assistant.

You may receive context from multiple documents.

Use:
1. Retrieved document context
2. Relevant conversation memory

Only answer when supported by context.

If uncertain say: "I could not find that in the documents."

==================================================
RELEVANT PAST CONVERSATIONS
==================================================

{memory_context}

==================================================
DOCUMENT CONTEXT
==================================================

{context}

==================================================
QUESTION
==================================================

{question}
"""

    print("\n" + "=" * 80)
    print("FINAL PROMPT SENT TO LLM")
    print("=" * 80)
    print(prompt[:12000])
    print("=" * 80)

    # =========================
    # LLM
    # =========================

    print("\nGenerating LLM response...")

    response = generate_response(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = response.choices[0].message.content

    print("\nANSWER GENERATED")

    # =========================
    # BACKGROUND TASKS
    # =========================

    background_tasks.add_task(set_exact_cache, cache_key, question, answer)
    background_tasks.add_task(
        set_semantic_cache, cache_key, question, query_embedding, answer
    )
    background_tasks.add_task(save_conversation_turn, session_id, question, answer)

    return answer
