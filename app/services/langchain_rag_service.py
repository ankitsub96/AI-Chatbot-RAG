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
from typing import List
from langchain_groq import ChatGroq


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
    load_index_and_metadata,
    semantic_search,
)
from app.utils.file_utils import get_index_path, get_metadata_path
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


class FaissRetriever(BaseRetriever):
    index: object
    documents: list
    query_embedding: object

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        results = semantic_search(
            index=self.index,
            metadata=self.documents,
            query_embedding=self.query_embedding,
            top_k=TOP_K,
        )

        return [
            Document(
                page_content=r["data"]["text"],
                metadata={"page": r["data"]["page"], "score": r["score"]},
            )
            for r in results
        ]


# ── service ──────────────────────────────────────────────────────────────────


async def ask_document_langchain(
    filename: str,
    question: str,
    session_id: str,
    background_tasks: BackgroundTasks,
    stream: bool = False,
) -> str | AsyncGenerator[str, None]:

    print("\n" + "=" * 80)
    print("LANGCHAIN ASK")
    print("=" * 80)

    # =========================
    # EXACT CACHE
    # =========================

    exact_cached = get_exact_cache(filename, question)

    if exact_cached:
        print("\nEXACT CACHE HIT")
        if stream:

            async def _exact_stream():
                yield format_sse(json.dumps({"token": exact_cached}))
                yield format_sse(json.dumps({"done": True}), event="done")

            return _exact_stream()
        return exact_cached

    # =========================
    # QUERY EMBEDDING
    # =========================

    print("\nGenerating query embedding...")

    query_embedding = await asyncio.to_thread(create_embedding, question)

    # =========================
    # SEMANTIC CACHE
    # =========================

    semantic_cached = get_semantic_cache(filename, query_embedding)

    if semantic_cached:
        print("\nSEMANTIC CACHE HIT")
        if stream:

            async def _semantic_stream():
                yield format_sse(json.dumps({"token": semantic_cached}))
                yield format_sse(json.dumps({"done": True}), event="done")

            return _semantic_stream()
        return semantic_cached

    # =========================
    # FILE VALIDATION
    # =========================

    index_path = get_index_path(filename)
    metadata_path = get_metadata_path(filename)

    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' is still processing",
        )

    # =========================
    # PARALLEL LOAD
    # =========================

    print("\nLoading index and memory in parallel...")

    (index, documents), memory_context = await asyncio.gather(
        asyncio.to_thread(load_index_and_metadata, index_path, metadata_path),
        asyncio.to_thread(retrieve_relevant_memories, session_id, question),
    )

    # =========================
    # RETRIEVER
    # =========================

    retriever = FaissRetriever(
        index=index,
        documents=documents,
        query_embedding=query_embedding,
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
        You are a helpful PDF assistant.

        Use the context below to answer the question.
        If uncertain, say: "I could not find that in the document."

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
        background_tasks.add_task(set_exact_cache, filename, question, answer)
        background_tasks.add_task(
            set_semantic_cache, filename, question, query_embedding, answer
        )
        background_tasks.add_task(save_conversation_turn, session_id, question, answer)

    # =========================
    # STREAM
    # =========================

    if stream:

        async def _stream() -> AsyncGenerator[str, None]:

            full_answer = ""

            chain = {"context": retriever, "question": lambda x: x} | prompt | llmGroq

            async for chunk in chain.astream(question):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield format_sse(json.dumps({"token": token}))

            yield format_sse(json.dumps({"done": True}), event="done")

            schedule_saves(full_answer)

        return _stream()

    # =========================
    # JSON — RetrievalQA single call
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


def format_sse(data: str, event: str = "message") -> str:
    return f"event: {event}\ndata: {data}\n\n"
