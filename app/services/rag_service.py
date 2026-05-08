import os
import json
import fitz
import asyncio
from fastapi import HTTPException, BackgroundTasks

from app.config.settings import MODEL, TOP_K, CHUNK_SIZE
from app.services.vector_service import embedding_model
from app.services.llm_service import generate_response
from app.utils.file_utils import get_index_path, get_metadata_path, load_documents
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

# =========================
# CHUNKING
# =========================


def chunk_text(text, chunk_size=400):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk_words = words[i : i + chunk_size]

        chunks.append(" ".join(chunk_words))

    return chunks


# =========================
# PDF EXTRACTION
# =========================


def extract_pdf_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num, page in enumerate(doc):

        pages.append({"page": page_num + 1, "text": page.get_text()})

    return pages


# =========================
# BUILD VECTOR DATABASE
# =========================


def build_vector_database(filename: str):

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

    documents = []

    # =========================
    # CHUNKING
    # =========================

    print("\nChunking document...")

    for page_data in pages:

        chunks = chunk_text(
            page_data["text"],
            CHUNK_SIZE,
        )

        for chunk in chunks:

            documents.append(
                {
                    "page": page_data["page"],
                    "text": chunk,
                }
            )

    print(f"\nTotal chunks created: {len(documents)}")

    # =========================
    # EMBEDDING TEXT PREP
    # =========================

    texts = [f"""
Represent this document for retrieval.

Page {doc['page']}

{doc['text']}
""" for doc in documents]

    # =========================
    # EMBEDDINGS
    # =========================

    print("\nGenerating embeddings...")

    embeddings = create_embeddings(
        texts,
        show_progress_bar=True,
    )

    print({"embedding_shape": embeddings.shape})

    # =========================
    # INDEX CREATION
    # =========================

    print("\nCreating FAISS HNSW index...")

    index = create_hnsw_index(
        embeddings=embeddings,
        hnsw_m=32,
        ef_construction=200,
    )

    print({"total_vectors": index.ntotal})

    # =========================
    # SAVE PATHS
    # =========================

    index_path = get_index_path(filename)

    metadata_path = get_metadata_path(filename)

    print(
        {
            "index_path": index_path,
            "metadata_path": metadata_path,
        }
    )

    # =========================
    # SAVE INDEX
    # =========================

    print("\nSaving FAISS index...")

    save_faiss_index(
        index,
        index_path,
    )

    # =========================
    # SAVE METADATA
    # =========================

    print("\nSaving metadata...")

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            documents,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("VECTOR DATABASE READY")
    print("=" * 80)

    print(f"Saved index: {index_path}")

    print(f"Saved metadata: {metadata_path}")

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
    filename: str, question: str, session_id: str, background_tasks: BackgroundTasks
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

    index_metadata_task = asyncio.to_thread(
        load_index_and_metadata,
        index_path,
        metadata_path,
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
        memory_context,
        summary,
    ) = await asyncio.gather(
        index_metadata_task,
        memory_task,
        summary_task,
    )

    index, documents = index_metadata

    # =========================
    # VECTOR SEARCH
    # =========================

    print("\nRunning vector similarity search...")

    results = semantic_search(
        index=index,
        metadata=documents,
        query_embedding=query_embedding,
        top_k=TOP_K,
    )

    context = ""

    print("\n" + "=" * 80)
    print("VECTOR SEARCH RESULTS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        distance = result["score"]

        doc = result["data"]

        print(f"\nRESULT #{rank}")

        print("-" * 80)

        print(f"Page: {doc['page']}")

        print(f"Similarity Score: {distance}")

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
