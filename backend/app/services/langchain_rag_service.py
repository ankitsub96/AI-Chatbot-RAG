import os
import asyncio
import json
from typing import AsyncGenerator
from fastapi import HTTPException, BackgroundTasks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from typing import List, Optional
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor
import time

from sqlmodel import Session, select
from sqlalchemy import text

from app.services.database import engine
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
    # load_index_and_metadata,
    # semantic_search,
    # build_bm25_index,
    # save_bm25_index,
    # load_bm25_index,
    hybrid_search,
)
from app.utils.file_utils import (
    get_index_path,
    get_metadata_path,
    get_bm25_path,
)

from app.utils.helpers import format_sse, thinking, token_event, done_event
from app.config.settings import TOP_K

# ── shared instances ─────────────────────────────────────────────────────────

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

# ── retriever ────────────────────────────────────────────────────────────────


# class FaissRetriever(BaseRetriever):

#     indexes: list
#     document_sets: list[list]

#     query_embedding: object
#     query: str = ""

#     bm25_indexes: Optional[list] = None

#     def _search_single_document(
#         self,
#         doc_idx: int,
#     ):

#         index = self.indexes[doc_idx]

#         documents = self.document_sets[doc_idx]

#         bm25_index = None

#         if self.bm25_indexes and len(self.bm25_indexes) > doc_idx:
#             bm25_index = self.bm25_indexes[doc_idx]

#         print(
#             {
#                 "document_set": doc_idx,
#                 "chunks": len(documents),
#                 "bm25_enabled": bm25_index is not None,
#             }
#         )

#         if bm25_index is not None:

#             return hybrid_search(
#                 faiss_index=index,
#                 bm25_index=bm25_index,
#                 metadata=documents,
#                 query_embedding=self.query_embedding,
#                 query=self.query,
#                 top_k=TOP_K,
#             )

#         return semantic_search(
#             index=index,
#             metadata=documents,
#             query_embedding=self.query_embedding,
#             top_k=TOP_K,
#         )

#     def _get_relevant_documents(
#         self,
#         query: str,
#         *,
#         run_manager: CallbackManagerForRetrieverRun,
#     ) -> List[Document]:

#         print("\n" + "=" * 80)
#         print("LANGCHAIN RETRIEVER")
#         print("=" * 80)

#         # =========================
#         # PARALLEL SEARCH
#         # =========================

#         worker_count = min(
#             len(self.indexes),
#             8,
#         )

#         with ThreadPoolExecutor(max_workers=worker_count) as executor:

#             search_results = list(
#                 executor.map(
#                     self._search_single_document,
#                     range(len(self.indexes)),
#                 )
#             )

#         # =========================
#         # MERGE
#         # =========================

#         all_results = []

#         for result_set in search_results:
#             all_results.extend(result_set)

#         print(
#             {
#                 "documents_searched": len(self.indexes),
#                 "total_candidates": len(all_results),
#             }
#         )

#         # =========================
#         # GLOBAL RERANK
#         # =========================

#         all_results.sort(
#             key=lambda x: x["score"],
#             reverse=True,
#         )

#         all_results = all_results[:TOP_K]

#         print(
#             {
#                 "final_results": len(all_results),
#             }
#         )

#         # =========================
#         # DEBUG
#         # =========================

#         for rank, result in enumerate(all_results, start=1):

#             print(
#                 {
#                     "rank": rank,
#                     "score": result["score"],
#                     "page": result["data"].get("page"),
#                     "source_file": result["data"].get("source_file"),
#                 }
#             )

#         # =========================
#         # LANGCHAIN DOCUMENTS
#         # =========================

#         return [
#             Document(
#                 page_content=result["data"]["text"],
#                 metadata={
#                     "page": result["data"].get("page"),
#                     "source_file": result["data"].get("source_file"),
#                     "score": result["score"],
#                 },
#             )
#             for result in all_results
#         ]


def log_step(step: str, data=None):
    print(f"\n✓ {step}")
    if data is not None:
        print(data)


class PostgresRetriever(BaseRetriever):

    filenames: list[str]
    query_embedding: object
    query: str = ""

    # optional debug trace collector
    trace: list = []

    def _log(self, step: str, data=None):
        msg = {"step": step, "data": data, "ts": time.time()}
        self.trace.append(msg)
        print(f"\n✓ {step}")
        if data:
            print(data)

    def _search_single_document(self, filename: str):

        self._log("search.single_document.start", {"filename": filename})

        with Session(engine) as session:

            results = hybrid_search(
                session=session,
                query=self.query,
                query_embedding=self.query_embedding,
                filename=filename,
                top_k=TOP_K,
            )

        self._log(
            "search.single_document.done",
            {
                "filename": filename,
                "results": len(results),
            },
        )

        return results

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:

        start = time.time()

        # =====================================================
        # START
        # =====================================================
        self._log("retriever.start")
        self._log("retriever.query_received", query)
        self._log("retriever.documents", self.filenames)

        self.query = query

        # =====================================================
        # SEARCH ALL FILES IN PARALLEL
        # =====================================================
        self._log(
            "retriever.searching_documents",
            {"count": len(self.filenames)},
        )

        worker_count = min(len(self.filenames), 8)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:

            search_results = list(
                executor.map(
                    self._search_single_document,
                    self.filenames,
                )
            )

        # =====================================================
        # RAW RESULTS DEBUG
        # =====================================================
        self._log("retriever.raw_results_collected")

        for filename, result_set in zip(self.filenames, search_results):

            self._log(
                "retriever.file_results",
                {
                    "filename": filename,
                    "count": len(result_set),
                },
            )

        # =====================================================
        # MERGE
        # =====================================================
        all_results = []

        for result_set in search_results:
            all_results.extend(result_set)

        self._log(
            "retriever.candidates",
            {"total": len(all_results)},
        )

        # =====================================================
        # RANKING STEP (THIS IS YOUR "THINKING")
        # =====================================================
        self._log(
            "retriever.ranking",
            {"top_k": TOP_K},
        )

        all_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        top_results = all_results[:TOP_K]

        # =====================================================
        # FINAL OUTPUT TRACE
        # =====================================================
        self._log(
            "retriever.final_chunks",
            {
                "returned": len(top_results),
                "execution_time_sec": round(time.time() - start, 3),
            },
        )

        for i, r in enumerate(top_results, 1):

            self._log(
                "retriever.chunk",
                {
                    "rank": i,
                    "page": r.get("page"),
                    "file": r.get("filename"),
                    "score": r["hybrid_score"],
                },
            )

        # =====================================================
        # RETURN LANGCHAIN DOCUMENTS
        # =====================================================
        return [
            Document(
                page_content=r["text"],
                metadata={
                    "page": r.get("page"),
                    "source_file": r.get("filename"),
                    "score": r["hybrid_score"],
                },
            )
            for r in top_results
        ]


# ── service ──────────────────────────────────────────────────────────────────


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
        # Drain thinking queue first
        if not self._thinking_queue.empty():
            return self._thinking_queue.get_nowait()

        # Initialize chain iterator once
        if self._chain_iter is None:
            chain = (
                {"context": self._retriever, "question": lambda x: x}
                | self._prompt
                | llmGroq
            )
            self._chain_iter = chain.astream(self._question).__aiter__()

        if self._done:
            raise StopAsyncIteration

        # Get next chunk
        try:
            chunk = await self._chain_iter.__anext__()
            # Drain thinking queue between chunks
            if not self._thinking_queue.empty():
                return self._thinking_queue.get_nowait()
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                self._full_answer += token
                return token_event(token)
            return ""  # empty chunk, caller will just get empty string
        except StopAsyncIteration:
            self._done = True
            self._schedule_saves(self._full_answer)
            return done_event()


async def ask_document_langchain(
    filenames: list[str],
    question: str,
    session_id: str,
    background_tasks: BackgroundTasks,
    stream: bool = False,
) -> str | AsyncGenerator[str, None]:

    print("\n" + "=" * 80)
    print("LANGCHAIN ASK")
    print("=" * 80)
    print({"filenames": filenames, "question": question, "session_id": session_id})

    cache_key = "|".join(sorted(filenames))
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
    print("\nGenerating query embedding...")
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
        {"filenames": filenames},
    )

    # =========================
    # MEMORY
    # =========================
    think("memory_search", "Searching conversation memory", {"session_id": session_id})
    print("\nLoading memory context...")
    memory_context = await asyncio.to_thread(
        retrieve_relevant_memories, session_id, question
    )
    think("memory_done", "Memory context loaded")

    # =========================
    # RETRIEVER
    # =========================
    think("retrieval", "Running hybrid retrieval (vector + BM25)")
    retriever = PostgresRetriever(
        filenames=filenames,
        query_embedding=query_embedding,
        query=question,
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
        input_variables=["context", "question"],
        template=f"""
You are a helpful document assistant.

Use:
1. Retrieved document context
2. Relevant conversation memory

If uncertain say: "I could not find that in the documents."

==================================================
RELEVANT PAST CONVERSATIONS
==================================================

{safe_memory_context}

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
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
        output_key="result",
    )

    print("\nRunning LangChain chain...")
    result = await chain.ainvoke({"query": question})
    answer = result["result"]

    print("\nANSWER GENERATED")
    schedule_saves(answer)

    return answer
