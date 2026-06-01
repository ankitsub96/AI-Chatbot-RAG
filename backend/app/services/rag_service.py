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
async def build_vector_database(filename: str):
    pdf_path = UPLOAD_DIR / filename

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

    with Session(engine) as session:
        session.execute(
            text("DELETE FROM document_chunks WHERE filename = :filename"),
            {"filename": filename},
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

            for doc, embedding in zip(batch_docs, embeddings):

                rows.append(
                    DocumentChunk(
                        filename=filename,
                        page=doc.get("page"),
                        section=doc.get("section"),
                        chunk_type=doc.get("type"),
                        text=doc["text"],
                        chunk_metadata={
                            **doc,
                            "parent_id": doc.get("parent_id"),
                            "child_id": doc.get("child_id"),
                        },
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

        session.execute(
            text("""
                UPDATE document_chunks
                SET tsv = to_tsvector('simple', text)
                WHERE filename = :filename
            """),
            {"filename": filename},
        )

        session.commit()

    print("\n" + "=" * 80)
    print("VECTOR DATABASE READY (PARENT-CHILD)")
    print("=" * 80)

    print(f"Stored chunks: {inserted}")


# =========================
# LIST READY DOCS
# =========================


def list_ready_documents():

    with Session(engine) as session:

        rows = session.exec(select(DocumentChunk.filename).distinct()).all()

        return sorted(rows)


# =========================
# ASK DOCUMENT
# =========================


async def ask_document(
    filenames: list[str],
    question: str,
    session_id: str,
    background_tasks: BackgroundTasks,
):
    print("\n" + "=" * 80)
    print("NEW QUESTION")
    print("=" * 80)

    print(
        {
            "filenames": filenames,
            "question": question,
            "session_id": session_id,
        }
    )

    # =========================
    # CACHE KEY
    # =========================

    cache_key = "|".join(sorted(filenames))

    # =========================
    # EXACT CACHE
    # =========================

    exact_cached = get_exact_cache(
        cache_key,
        question,
    )

    if exact_cached:
        print("\nEXACT CACHE HIT")
        return exact_cached

    # =========================
    # QUERY EMBEDDING
    # =========================

    print("\nGenerating query embedding...")

    query_embedding = await asyncio.to_thread(
        create_embedding,
        question,
    )

    # =========================
    # SEMANTIC CACHE
    # =========================

    semantic_cached = get_semantic_cache(
        cache_key,
        query_embedding,
    )

    if semantic_cached:
        print("\nSEMANTIC CACHE HIT")
        return semantic_cached

    # =========================
    # PARALLEL LOAD
    # =========================

    memory_task = asyncio.to_thread(
        retrieve_relevant_memories,
        session_id,
        question,
    )

    document_task = asyncio.to_thread(
        hybrid_document_search_multiple,
        filenames,
        question,
        query_embedding,
        TOP_K,
    )

    (
        memory_context,
        results,
    ) = await asyncio.gather(
        memory_task,
        document_task,
    )

    print(
        "\n[document_loaded] Documents + BM25 indexes loaded"
    )  # <-- was yield thinking(...)

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
                "file": doc.get("filename"),
            }
        )

        context += f"""

[FILE: {doc.get('filename')}]

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

If uncertain say:

"I could not find that in the documents."

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
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    answer = response.choices[0].message.content

    print("\nANSWER GENERATED")

    # =========================
    # BACKGROUND TASKS
    # =========================

    background_tasks.add_task(
        set_exact_cache,
        cache_key,
        question,
        answer,
    )

    background_tasks.add_task(
        set_semantic_cache,
        cache_key,
        question,
        query_embedding,
        answer,
    )

    background_tasks.add_task(
        save_conversation_turn,
        session_id,
        question,
        answer,
    )

    return answer
