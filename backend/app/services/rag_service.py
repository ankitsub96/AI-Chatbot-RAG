import os

os.environ["UNSTRUCTURED_SKIP_TORCH"] = "1"
import json
import asyncio
from fastapi import HTTPException, BackgroundTasks
import numpy as np
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.getLogger("pdfminer").setLevel(logging.ERROR)

from app.config.settings import TOP_K, CHUNK_SIZE
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
    load_index_and_metadata,
    semantic_search,
    create_embeddings,
    create_hnsw_index,
    save_faiss_index,
    load_faiss_index,
)
from app.services.pdf_service import (
    group_pages,
    chunk_pages,
    process_document_pages,
    extract_pdf_text,
)
from app.services.vector_search_service import (
    build_bm25_index,
    save_bm25_index,
    load_bm25_index,
    hybrid_search,
)
from app.utils.file_utils import get_bm25_path

# =========================
# EMBEDDING CONFIG
# =========================

EMBEDDING_BATCH_SIZE = 8  # reduce if laptop struggles (try 4)
CHECKPOINT_EVERY = 50  # embed and report every N chunks

# =========================
# BUILD VECTOR DATABASE
# =========================


@timer
async def build_vector_database(filename: str):

    pdf_path = f"app/uploads/{filename}"

    print("\n" + "=" * 80)
    print("BUILDING VECTOR DATABASE")
    print("=" * 80)

    print({"filename": filename})

    # =========================
    # PDF EXTRACTION
    # =========================

    print("\nExtracting PDF text...")

    pages = extract_pdf_text(pdf_path)

    # =========================================================
    # PAGE GROUPING
    # =========================================================

    grouped_pages = group_pages(pages)

    # =========================================================
    # CHUNKING
    # =========================================================

    documents = chunk_pages(grouped_pages)

    # =========================
    # EMBEDDING TEXT PREP
    # =========================

    texts = [f"""
        Represent this document for retrieval.

        Page:
        {doc.get('page', '')}

        Section:
        {doc.get('section', '')}

        Type:
        {doc.get('type', '')}

        Content:
        {doc['text']}
        """ for doc in documents]

    # =========================
    # BATCH EMBEDDINGS WITH CHECKPOINTING
    # =========================

    print(f"\nGenerating embeddings...")
    print(
        f"Batch size: {EMBEDDING_BATCH_SIZE} | "
        f"Checkpoint every: {CHECKPOINT_EVERY} chunks"
    )

    all_embeddings = []
    total = len(texts)

    for i in range(0, total, CHECKPOINT_EVERY):

        batch_texts = texts[i : i + CHECKPOINT_EVERY]

        print(
            f"\nEmbedding chunks {i + 1}–{min(i + CHECKPOINT_EVERY, total)} "
            f"of {total}..."
        )

        batch_embeddings = create_embeddings(
            batch_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
        )

        all_embeddings.append(batch_embeddings)

        print(f"Done — {batch_embeddings.shape[0]} vectors embedded")

    embeddings = np.vstack(all_embeddings)

    print(f"\nFinal embedding shape: {embeddings.shape}")

    # =========================
    # INDEX CREATION
    # =========================

    print("\nCreating FAISS HNSW index...")

    # index = create_hnsw_index(
    #     embeddings=embeddings,
    #     hnsw_m=32,
    #     ef_construction=200,
    # )
    faiss_task = asyncio.to_thread(
        create_hnsw_index,
        embeddings,
        32,
        200,
    )

    bm25_task = asyncio.to_thread(
        build_bm25_index,
        documents,
    )

    index, bm25_index = await asyncio.gather(
        faiss_task,
        bm25_task,
    )

    print({"total_vectors": index.ntotal})

    # =========================
    # SAVE PATHS
    # =========================

    index_path = get_index_path(filename)
    metadata_path = get_metadata_path(filename)
    bm25_path = get_bm25_path(filename)

    print({"index_path": index_path, "metadata_path": metadata_path})

    # =========================
    # SAVE INDEX
    # =========================

    print("\nSaving FAISS index...")

    await asyncio.gather(
        asyncio.to_thread(save_faiss_index, index, index_path),
        asyncio.to_thread(save_bm25_index, bm25_index, bm25_path),
        asyncio.to_thread(
            lambda: json.dump(
                documents,
                open(metadata_path, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
        ),
    )

    print("\n" + "=" * 80)
    print("VECTOR DATABASE READY")
    print("=" * 80)

    print(f"Saved index:          {index_path}")
    print(f"Saved metadata:       {metadata_path}")
    print(f"Total chunks indexed: {len(documents)}")
    print("=" * 80)


# =========================
# LIST READY DOCS
# =========================


def list_ready_documents():

    files = []

    metadata_dir = "app/vector_store/metadata"

    if not os.path.exists(metadata_dir):

        return []

    for file in os.listdir(metadata_dir):

        if file.endswith(".json"):

            files.append(file.replace(".json", ".pdf"))

    return files


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

    print({"filename": filename, "question": question, "session_id": session_id})

    # =========================
    # EXACT CACHE
    # =========================

    exact_cached = get_exact_cache(filename, question)

    if exact_cached:

        print("\nEXACT CACHE HIT")

        return exact_cached

    # =========================
    # QUERY EMBEDDING
    # =========================

    print("\nGenerating query embedding...")

    query_embedding = create_embedding(question)

    # =========================
    # SEMANTIC CACHE
    # =========================

    semantic_cached = get_semantic_cache(filename, query_embedding)

    if semantic_cached:

        print("\nSEMANTIC CACHE HIT")

        return semantic_cached

    # =========================
    # FILE VALIDATION
    # =========================

    index_path = get_index_path(filename)

    metadata_path = get_metadata_path(filename)

    print({"index_path": index_path, "metadata_path": metadata_path})

    if not os.path.exists(index_path):

        raise HTTPException(
            status_code=404, detail=(f"Document '{filename}' " f"is still processing")
        )

    if not os.path.exists(metadata_path):

        raise HTTPException(
            status_code=404,
            detail=(f"Metadata for '{filename}' " f"is still processing"),
        )

    # =========================
    # PARALLEL LOADING
    # =========================

    print("\nLoading retrieval systems in parallel...")

    bm25_path = get_bm25_path(filename)

    index_metadata_task = asyncio.to_thread(
        load_index_and_metadata,
        index_path,
        metadata_path,
    )

    bm25_task = (
        asyncio.to_thread(load_bm25_index, bm25_path)
        if os.path.exists(bm25_path)
        else asyncio.sleep(0, result=None)
    )

    memory_task = asyncio.to_thread(
        retrieve_relevant_memories,
        session_id,
        question,
    )

    summary_task = asyncio.to_thread(
        load_summary,
        session_id,
    )

    (
        index_metadata,
        bm25_index,
        memory_context,
        summary,
    ) = await asyncio.gather(
        index_metadata_task,
        bm25_task,
        memory_task,
        summary_task,
    )

    index, documents = index_metadata

    # =========================
    # HYBRID SEARCH
    # =========================

    print("\nRunning vector similarity search...")

    if bm25_index is not None:

        print("Mode: Hybrid (FAISS + BM25)")

        results = hybrid_search(
            faiss_index=index,
            bm25_index=bm25_index,
            metadata=documents,
            query_embedding=query_embedding,
            query=question,
            top_k=TOP_K,
        )

    else:

        print("Mode: Semantic only (BM25 index not found)")

        results = semantic_search(
            index=index,
            metadata=documents,
            query_embedding=query_embedding,
            top_k=TOP_K,
        )

    context = ""

    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        score = result["score"]

        doc = result["data"]

        print(f"\nRESULT #{rank}")

        print("-" * 80)

        print(f"Page: {doc['page']}")

        print(f"Score: {score}")

        if bm25_index is not None:
            print(f"FAISS Score: {result.get('faiss_score', 'n/a')}")
            print(f"BM25 Score:  {result.get('bm25_score', 'n/a')}")

        print("\nCHUNK:")

        print(doc["text"][:1000])

        print("-" * 80)

        context += f"""

[Page {doc['page']}]

{doc['text']}
"""

    # =========================
    # FINAL PROMPT
    # =========================

    prompt = f"""
You are a helpful PDF assistant.

Use:
1. relevant document context
2. relevant past conversation memory
3. system summary

Only answer confidently when supported.

If uncertain, say:
"I could not find that in the document."

==================================================
SYSTEM SUMMARY
==================================================

{summary}

==================================================
RELEVANT PAST CONVERSATIONS
==================================================

{memory_context}

==================================================
DOCUMENT CONTEXT
==================================================

{context}

==================================================
CURRENT QUESTION
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
        messages=[{"role": "user", "content": prompt}], temperature=0
    )

    answer = response.choices[0].message.content

    print("\nANSWER GENERATED")

    # =========================
    # BACKGROUND TASKS
    # =========================

    print("\nScheduling background tasks...")

    background_tasks.add_task(set_exact_cache, filename, question, answer)

    background_tasks.add_task(
        set_semantic_cache, filename, question, query_embedding, answer
    )

    background_tasks.add_task(save_conversation_turn, session_id, question, answer)

    return answer
