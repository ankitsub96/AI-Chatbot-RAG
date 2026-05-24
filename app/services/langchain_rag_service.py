import os
import asyncio

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

TOP_K = 50

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
) -> str:

    print("\n" + "=" * 80)
    print("LANGCHAIN ASK")
    print("=" * 80)

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

    query_embedding = await asyncio.to_thread(create_embedding, question)

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

    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' is still processing",
        )

    # =========================
    # PARALLEL LOAD
    # =========================

    print("\nLoading index and memory in parallel...")

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

    (index, documents), memory_context = await asyncio.gather(
        index_metadata_task,
        memory_task,
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
    # CHAIN
    # =========================
    safe_memory_context = (
        (memory_context or "No previous conversations.")
        .replace("{", "{{")
        .replace("}", "}}")
    )

    prompt = PromptTemplate(
        input_variables=["safe_memory_context", "context", "question"],
        template=f"""
        You are a helpful PDF assistant.

        Use the context below to answer the question.
        If uncertain, say: "I could not find that in the document."

        ==================================================
        RELEVANT PAST CONVERSATIONS
        ==================================================
        {safe_memory_context or "No previous conversations."}

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

    chain = RetrievalQA.from_chain_type(
        # llm=llm,
        llm=llmGroq,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
        output_key="result",
    )

    # =========================
    # RUN
    # =========================

    print("\nRunning LangChain chain...")

    result = await chain.ainvoke(
        {
            "query": question,
        }
    )

    answer = result["result"]

    print("\nANSWER GENERATED")

    # =========================
    # BACKGROUND TASKS
    # =========================

    background_tasks.add_task(set_exact_cache, filename, question, answer)
    background_tasks.add_task(
        set_semantic_cache, filename, question, query_embedding, answer
    )
    background_tasks.add_task(save_conversation_turn, session_id, question, answer)

    return answer
