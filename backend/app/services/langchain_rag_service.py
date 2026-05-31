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
from app.utils.file_utils import get_index_path, get_metadata_path, get_bm25_path
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


class PostgresRetriever(BaseRetriever):

    filenames: list[str]

    query_embedding: object

    query: str = ""

    def _search_single_document(
        self,
        filename: str,
    ):
        with Session(engine) as session:

            return hybrid_search(
                session=session,
                query=self.query,
                query_embedding=self.query_embedding,
                filename=filename,
                top_k=TOP_K,
            )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        print("INSIDE RETRIEVER")
        print("\n" + "=" * 80)
        print("POSTGRES RETRIEVER")
        print("=" * 80)

        worker_count = min(
            len(self.filenames),
            8,
        )

        with ThreadPoolExecutor(
            max_workers=worker_count,
        ) as executor:

            search_results = list(
                executor.map(
                    self._search_single_document,
                    self.filenames,
                )
            )

        all_results = []

        for result_set in search_results:
            all_results.extend(result_set)

        print(
            {
                "documents_searched": len(self.filenames),
                "total_candidates": len(all_results),
            }
        )

        all_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        all_results = all_results[:TOP_K]

        print(
            {
                "final_results": len(all_results),
            }
        )

        for rank, result in enumerate(
            all_results,
            start=1,
        ):
            print(
                {
                    "rank": rank,
                    "score": result["hybrid_score"],
                    "page": result.get("page"),
                    "source_file": result.get("filename"),
                }
            )

        return [
            Document(
                page_content=result["text"],
                metadata={
                    "page": result.get("page"),
                    "source_file": result.get("filename"),
                    "score": result["hybrid_score"],
                },
            )
            for result in all_results
        ]


# ── service ──────────────────────────────────────────────────────────────────


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

        if stream:

            async def _exact_stream():
                yield format_sse(
                    json.dumps(
                        {
                            "token": exact_cached,
                        }
                    )
                )

                yield format_sse(
                    json.dumps(
                        {
                            "done": True,
                        }
                    ),
                    event="done",
                )

            return _exact_stream()

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

        if stream:

            async def _semantic_stream():
                yield format_sse(
                    json.dumps(
                        {
                            "token": semantic_cached,
                        }
                    )
                )

                yield format_sse(
                    json.dumps(
                        {
                            "done": True,
                        }
                    ),
                    event="done",
                )

            return _semantic_stream()

        return semantic_cached

    # =========================
    # MEMORY
    # =========================

    print("\nLoading memory context...")

    memory_context = await asyncio.to_thread(
        retrieve_relevant_memories,
        session_id,
        question,
    )

    # =========================
    # RETRIEVER
    # =========================

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
        input_variables=[
            "context",
            "question",
        ],
        template=f"""
You are a helpful document assistant.

You may receive context from multiple documents.

Use:
1. Retrieved document context
2. Relevant conversation memory

If uncertain say:

"I could not find that in the documents."

==================================================
RELEVANT PAST CONVERSATIONS
==================================================

{safe_memory_context}

==================================================
DOCUMENT CONTEXT
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

    # =========================
    # STREAM RESPONSE
    # =========================

    if stream:

        async def _stream():

            full_answer = ""

            chain = (
                {
                    "context": retriever,
                    "question": lambda x: x,
                }
                | prompt
                | llmGroq
            )

            async for chunk in chain.astream(question):

                token = chunk.content if hasattr(chunk, "content") else str(chunk)

                if token:

                    full_answer += token

                    yield format_sse(
                        json.dumps(
                            {
                                "token": token,
                            }
                        )
                    )

            yield format_sse(
                json.dumps(
                    {
                        "done": True,
                    }
                ),
                event="done",
            )

            schedule_saves(full_answer)

        return _stream()

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

    result = await chain.ainvoke(
        {
            "query": question,
        }
    )

    answer = result["result"]

    print("\nANSWER GENERATED")

    schedule_saves(answer)

    return answer


def format_sse(data: str, event: str = "message") -> str:
    return f"event: {event}\ndata: {data}\n\n"
