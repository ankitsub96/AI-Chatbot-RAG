import os
import asyncio
import json
from typing import AsyncGenerator, List, Optional
from fastapi import HTTPException, BackgroundTasks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import Field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from sqlmodel import Session, select
from sqlalchemy import text

from app.services.database import engine
from app.models.document_chunk import DocumentChunk
from app.models.session_document import SessionDocument
from app.models.document import Document as DocumentModel
from app.services.memory_service import (
    retrieve_relevant_memories,
    save_conversation_turn,
)
from app.services.semantic_cache_service import (
    get_exact_cache,
    get_semantic_cache,
    set_exact_cache,
    set_semantic_cache,
)
from app.services.vector_search_service import (
    create_embedding,
    hybrid_search,
    hybrid_document_search_multiple,
    group_chunks_by_parent,
    select_top_parents,
    expand_parent_chunks,
)
from app.utils.helpers import format_sse, thinking, token_event, done_event
from app.utils.helpers import timer
from app.config.settings import TOP_K

# ── shared LLM instances ──────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

llmGroq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ── helpers ───────────────────────────────────────────────────────────────────


def log_step(step: str, data=None):
    print(f"\n✓ {step}")
    if data is not None:
        print(data)


def _resolve_document_ids(session_id: str) -> list[str]:
    """Resolve all document_ids linked to a session via SessionDocument."""
    with Session(engine) as db:
        rows = db.exec(
            select(SessionDocument.document_id).where(
                SessionDocument.session_id == session_id
            )
        ).all()
    return list(rows)


# ── retriever ─────────────────────────────────────────────────────────────────


class PostgresRetriever(BaseRetriever):
    session_id: str
    query_embedding: object | None = None
    query: str = ""
    trace: list = Field(default_factory=list)
    document_ids_filter: list[str] | None = None

    # -------------------------
    # logging
    # -------------------------

    def _log(self, step: str, data=None):
        msg = {
            "step": step,
            "data": data,
            "ts": time.time(),
        }

        self.trace.append(msg)

        print(f"\n✓ {step}")

        if data:
            print(data)

    # -------------------------
    # query expansion
    # -------------------------

    def _expand_queries(self, query: str) -> list[str]:
        try:
            expanded = expand_query_sync(
                question=query,
                n=10,
            )
        except Exception as e:
            self._log(
                "retriever.query_expansion.error",
                {"error": str(e)},
            )

            expanded = []

        queries = [query]

        for q in expanded:
            if q and q not in queries:
                queries.append(q)

        return queries

    # -------------------------
    # search one document
    # -------------------------

    def _search_document(
        self,
        query: str,
        query_embedding,
        document_id: str,
    ):
        try:
            with Session(engine) as session:
                return hybrid_search(
                    # session=session,
                    query=query,
                    query_embedding=query_embedding,
                    document_id=document_id,
                    top_k=TOP_K,
                )

        except Exception as e:
            self._log(
                "search.document.error",
                {
                    "document_id": document_id,
                    "error": str(e),
                },
            )

            return []

    # -------------------------
    # search one query
    # -------------------------

    def _retrieve_for_query(
        self,
        query: str,
        query_embedding,
        document_ids: list[str],
    ):
        results = []

        workers = min(
            max(1, len(document_ids)),
            16,
        )

        with ThreadPoolExecutor(max_workers=workers) as executor:

            futures = {
                executor.submit(
                    self._search_document,
                    query,
                    query_embedding,
                    doc_id,
                ): doc_id
                for doc_id in document_ids
            }

            for future in as_completed(futures):
                try:
                    results.extend(future.result())

                except Exception as e:
                    self._log(
                        "search.document.future_error",
                        {"error": str(e)},
                    )

        return results

    # -------------------------
    # main retrieval
    # -------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:

        start = time.time()

        self._log("retriever.start")
        self._log("retriever.query_received", query)

        self.query = query

        # =====================
        # resolve docs
        # =====================

        if self.document_ids_filter:
            document_ids = self.document_ids_filter

            self._log(
                "retriever.document_ids_from_filter",
                {"document_ids": document_ids},
            )

        else:
            document_ids = _resolve_document_ids(self.session_id)

            self._log(
                "retriever.document_ids_resolved",
                {
                    "count": len(document_ids),
                },
            )

        if not document_ids:
            self._log("retriever.no_documents")
            return []

        # =====================
        # query expansion
        # =====================

        expanded_queries = self._expand_queries(query)

        self._log(
            "retriever.query_expansion.done",
            {
                "count": len(expanded_queries),
                "queries": expanded_queries,
            },
        )

        # =====================
        # embeddings
        # =====================

        self._log(
            "retriever.embedding.start",
            {"queries": len(expanded_queries)},
        )

        with ThreadPoolExecutor(max_workers=min(len(expanded_queries), 8)) as executor:

            embeddings = list(
                executor.map(
                    create_embedding,
                    expanded_queries,
                )
            )

        self._log(
            "retriever.embedding.done",
            {
                "embeddings": len(embeddings),
            },
        )

        # =====================
        # retrieval
        # =====================

        self._log(
            "retriever.multi_query.start",
            {
                "queries": len(expanded_queries),
                "documents": len(document_ids),
            },
        )

        all_result_sets = []

        with ThreadPoolExecutor(max_workers=min(len(expanded_queries), 8)) as executor:

            futures = [
                executor.submit(
                    self._retrieve_for_query,
                    q,
                    emb,
                    document_ids,
                )
                for q, emb in zip(
                    expanded_queries,
                    embeddings,
                )
            ]

            for future in as_completed(futures):
                try:
                    all_result_sets.append(future.result())

                except Exception as e:
                    self._log(
                        "retriever.query_error",
                        {"error": str(e)},
                    )

        # =====================
        # RRF merge
        # =====================

        rrf_k = 60

        merged = {}

        for result_set in all_result_sets:

            ranked = sorted(
                result_set,
                key=lambda x: x.get(
                    "score",
                    0,
                ),
                reverse=True,
            )

            for rank, row in enumerate(ranked):

                chunk_id = row["id"]

                rrf_score = 1 / (rrf_k + rank + 1)

                if chunk_id not in merged:

                    merged[chunk_id] = {
                        **row,
                        "rrf_score": 0,
                    }

                merged[chunk_id]["rrf_score"] += rrf_score

        all_results = sorted(
            merged.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        self._log(
            "retriever.rrf.done",
            {
                "unique_chunks": len(all_results),
            },
        )

        # =====================
        # parent grouping
        # =====================

        grouped = group_chunks_by_parent(all_results)

        top_parents = select_top_parents(
            grouped,
            top_n=TOP_K,
        )

        self._log(
            "retriever.parents_selected",
            {
                "total_parents": len(grouped),
                "selected": len(top_parents),
            },
        )

        # =====================
        # parent expansion
        # =====================

        parent_ids = [p["parent_id"] for p in top_parents]

        with Session(engine) as session:
            expanded = expand_parent_chunks(
                session=session,
                parent_ids=parent_ids,
            )

        # =====================
        # build docs
        # =====================

        documents = []

        for parent in top_parents:

            parent_id = parent["parent_id"]

            children = expanded.get(
                parent_id,
                parent["chunks"],
            )

            children = sorted(
                children,
                key=lambda c: c.get("chunk_index", 0),
            )

            full_text = "\n\n".join(c["text"] for c in children)

            page = children[0].get("page", "?") if children else "?"

            documents.append(
                Document(
                    page_content=full_text,
                    metadata={
                        "parent_id": parent_id,
                        "page": page,
                        "document_id": (
                            children[0].get("document_id") if children else None
                        ),
                        "score": parent["score"],
                        "rrf_score": parent.get("rrf_score"),
                        "child_count": len(children),
                    },
                )
            )

        self._log(
            "retriever.complete",
            {
                "documents": len(documents),
                "execution_time_sec": round(
                    time.time() - start,
                    3,
                ),
            },
        )

        return documents


# ── streaming helpers ─────────────────────────────────────────────────────────


class _ListStream:
    def __init__(self, items: list[str]):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class _LLMStream:
    def __init__(
        self,
        retriever,
        prompt,
        question: str,
        schedule_saves,
        thinking_queue: asyncio.Queue,
    ):
        self._retriever = retriever
        self._prompt = prompt
        self._question = question
        self._schedule_saves = schedule_saves
        self._thinking_queue = thinking_queue
        self._chain_iter = None
        self._full_answer = ""
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self._thinking_queue.empty():
            return self._thinking_queue.get_nowait()

        if self._chain_iter is None:
            chain = (
                {"context": self._retriever, "question": lambda x: x}
                | self._prompt
                | llmGroq
            )
            self._chain_iter = chain.astream(self._question).__aiter__()

        if self._done:
            raise StopAsyncIteration

        try:
            chunk = await self._chain_iter.__anext__()
            if not self._thinking_queue.empty():
                return self._thinking_queue.get_nowait()
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                self._full_answer += token
                return token_event(token)
            return ""
        except StopAsyncIteration:
            self._done = True
            self._schedule_saves(self._full_answer)
            return done_event()


# ── service ───────────────────────────────────────────────────────────────────


async def ask_document_langchain(
    session_id: str,
    question: str,
    background_tasks: BackgroundTasks,
    stream: bool = False,
    document_ids: list[str] | None = None,  # optional subset filter
) -> str | AsyncGenerator[str, None]:

    print("\n" + "=" * 80)
    print("LANGCHAIN ASK")
    print("=" * 80)
    print({"session_id": session_id, "question": question})

    cache_key = session_id
    thinking_queue: asyncio.Queue = asyncio.Queue()

    def think(event: str, message: str, data: dict = None):
        if stream:
            thinking_queue.put_nowait(
                format_sse(
                    json.dumps({"thinking": event, "message": message, **(data or {})})
                )
            )
        else:
            print(f"[{event}] {message}", data or "")

    # =========================
    # EXACT CACHE
    # =========================
    exact_cached = get_exact_cache(cache_key, question)
    if exact_cached:
        print("\nEXACT CACHE HIT")
        if stream:
            return _ListStream(
                [
                    format_sse(json.dumps({"token": exact_cached})),
                    format_sse(json.dumps({"done": True}), event="done"),
                ]
            )
        return exact_cached

    # =========================
    # QUERY EMBEDDING
    # =========================
    think("embedding", "Creating query embedding", {"model": "sentence-transformers"})
    query_embedding = await asyncio.to_thread(create_embedding, question)
    think("embedding_done", "Query embedding created")

    # =========================
    # SEMANTIC CACHE
    # =========================
    think("cache", "Checking semantic cache")
    semantic_cached = get_semantic_cache(cache_key, query_embedding)
    if semantic_cached:
        print("\nSEMANTIC CACHE HIT")
        if stream:
            return _ListStream(
                [
                    format_sse(json.dumps({"token": semantic_cached})),
                    format_sse(json.dumps({"done": True}), event="done"),
                ]
            )
        return semantic_cached

    think(
        "cache_miss",
        "Cache miss — starting retrieval pipeline",
        {"session_id": session_id},
    )

    # =========================
    # MEMORY
    # =========================
    think("memory_search", "Searching conversation memory", {"session_id": session_id})
    memory_context = await asyncio.to_thread(
        retrieve_relevant_memories, session_id, question
    )
    think("memory_done", "Memory context loaded")

    # =========================
    # RETRIEVER
    # =========================
    think("retrieval", "Running hybrid retrieval (vector + BM25)")
    retriever = PostgresRetriever(
        session_id=session_id,
        query_embedding=query_embedding,
        query=question,
        document_ids_filter=document_ids,  # None = search all session docs
    )

    # =========================
    # PROMPT
    # =========================
    safe_memory_context = (
        (memory_context or "No previous conversations.")
        .replace("{", "{{")
        .replace("}", "}}")
    )

    prompt = PromptTemplate(
        input_variables=["context", "safe_memory_context", "question"],
        template=f"""
You are a helpful document assistant.

Use:
1. Retrieved document context
2. Relevant conversation memory
3. When citations are aksed for, try to format the metadata related to the chunk in human readable format whenever it is included in the response. Also try to include what text was returned from this page, only when citing. do this only in the separate citation section at the end. 
Don't return metadata json.

Example:
Citations:
- Document metadata={{{{'parent_id': 'parent_930_page_174', 'page': 174, ...}}}}

Return:
page 174 =(Exact Text from the chunk, which was taken as basis for answer)


If uncertain say: "I could not find that in the documents."

==================================================
RELEVANT PAST CONVERSATIONS
==================================================

{safe_memory_context}

==================================================
RETRIEVED DOCUMENTS
==================================================

{{context}}
==================================================
QUESTION
==================================================

{{question}}
""",
    )

    # =========================
    # BACKGROUND SAVE HELPER
    # =========================
    def schedule_saves(answer: str):
        background_tasks.add_task(set_exact_cache, cache_key, question, answer)
        background_tasks.add_task(
            set_semantic_cache, cache_key, question, query_embedding, answer
        )
        background_tasks.add_task(save_conversation_turn, session_id, question, answer)

    # =========================
    # STREAM RESPONSE
    # =========================
    think("llm", "Building prompt and generating response")

    if stream:
        return _LLMStream(retriever, prompt, question, schedule_saves, thinking_queue)

    # =========================
    # NORMAL RESPONSE
    # =========================
    chain = RetrievalQA.from_chain_type(
        llm=llmGroq,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
        output_key="result",
    )

    print("\nRunning LangChain chain...")
    result = await chain.ainvoke({"query": question})
    answer = result["result"]

    print("\nANSWER GENERATED")
    schedule_saves(answer)

    return answer


@timer
def expand_query_sync(
    question: str,
    memory_context: str = "",
    n: int = 4,
) -> list[str]:
    """
    Synchronous wrapper around expand_query().
    Safe to call from LangChain retrievers.
    """

    prompt = f"""
    You are a query rewriting assistant for a retrieval system.

    Your goal is to generate search queries that retrieve evidence relevant to the ORIGINAL QUESTION.

    IMPORTANT:
    - Preserve the original meaning and intent.
    - Do NOT broaden the topic.
    - Do NOT introduce new questions.
    - Do NOT introduce related themes that are not explicitly present.
    - Keep important named entities, characters, places, events, and concepts.
    - Focus on alternative wording, terminology, and phrasing.
    - Queries should retrieve overlapping evidence from different lexical angles.
    - Avoid generic or high-level summaries.

    Conversation history:
    {memory_context or "None"}

    Original question:
    {question}

    Generate exactly {n} rewritten search queries.

    Good examples:

    Question:
    Why did Murtagh remain loyal to Galbatorix?

    Good rewrites:
    [
    "Reasons for Murtagh's loyalty to Galbatorix",
    "What motivated Murtagh to serve Galbatorix",
    "Factors influencing Murtagh's allegiance to Galbatorix",
    "Murtagh's relationship with Galbatorix and loyalty"
    ]

    Bad rewrites:
    [
    "Dragon rider politics",
    "Power struggles in Alagaësia",
    "History of Galbatorix",
    "Murtagh character analysis"
    ]

    Return ONLY a JSON array of strings.
    """

    try:
        response = llmGroq.invoke([{"role": "user", "content": prompt}])

        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        queries = json.loads(raw)

        if isinstance(queries, list):
            all_queries = [question]

            for q in queries:
                if isinstance(q, str) and q.strip() and q not in all_queries:
                    all_queries.append(q)
            print({"all_queries": all_queries})
            return all_queries[: n + 1]

    except Exception as e:
        print(f"[expand_query_sync] failed: {e}")

    return [question]
