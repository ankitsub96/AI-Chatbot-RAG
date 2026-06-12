import json
import time
import asyncio
from typing import Annotated, AsyncGenerator
import operator
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.services.llm_service import generate_response, expand_query_sync
from app.services.vector_search_service import (
    create_embedding,
    hybrid_search,
    group_chunks_by_parent,
    select_top_parents,
    expand_parent_chunks,
    build_parent_context_blocks,
)
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
from app.utils.helpers import format_sse, token_event, done_event, timer
from app.config.settings import TOP_K, MAX_RETRIES, CONFIDENCE_THRESHOLD
from app.services.database import engine
from app.models.document_chunk import DocumentChunk
from sqlmodel import Session, select

# =====================================================
# STATE
# =====================================================


class AgentState(TypedDict):
    session_id: str
    question: str
    document_ids: list[str]
    current_question: str
    query_embedding: list[float]
    memory_context: str
    memory_messages: list[dict]
    expanded_queries: list[str]
    child_results: list[dict]
    context: str
    full_answer: str
    eval_result: dict | None
    attempt: int
    trace: Annotated[list[dict], operator.add]
    last_thought: dict
    cache_hit: bool
    cache_answer: str | None


# =====================================================
# HELPERS
# =====================================================


def _thought(node: str, message: str, data: dict | None = None) -> dict:
    return {"node": node, "message": message, "data": data, "ts": time.time()}


def _parse_memory_to_messages(memory_context: str) -> list[dict]:
    if not memory_context.strip():
        return []

    messages = []
    blocks = memory_context.split("USER:")
    for block in blocks:
        if not block.strip():
            continue
        if "A:" in block:
            parts = block.split("A:")
            human_text = parts[0].strip()
            ai_text = parts[1].strip() if len(parts) > 1 else ""
            if human_text:
                messages.append({"type": "human", "content": human_text})
            if ai_text:
                messages.append({"type": "ai", "content": ai_text})
        else:
            messages.append({"type": "human", "content": block.strip()})
    return messages


# =====================================================
# NODES
# =====================================================


def node_cache_check(state: AgentState) -> dict:
    question = state["question"]
    session_id = state["session_id"]
    document_ids = state["document_ids"]

    cache_key = session_id + "|" + "|".join(sorted(document_ids))

    exact = get_exact_cache(cache_key, question)
    if exact:
        return {
            "cache_hit": True,
            "cache_answer": exact,
            "last_thought": _thought("cache_check", "Exact cache hit"),
            "trace": [_thought("cache_check", "Exact cache hit")],
        }

    embedding = create_embedding(question)

    semantic = get_semantic_cache(cache_key, embedding[0].tolist())
    if semantic:
        return {
            "cache_hit": True,
            "cache_answer": semantic,
            "query_embedding": embedding[0].tolist(),
            "last_thought": _thought("cache_check", "Semantic cache hit"),
            "trace": [_thought("cache_check", "Semantic cache hit")],
        }

    return {
        "cache_hit": False,
        "query_embedding": embedding[0].tolist(),
        "current_question": question,
        "last_thought": _thought(
            "cache_check", "Cache miss — starting agentic pipeline"
        ),
        "trace": [_thought("cache_check", "Cache miss — starting agentic pipeline")],
    }


def node_save_cache_hit(state: AgentState) -> dict:
    thought = _thought("cache_check", "Returning cached answer")
    return {"last_thought": thought, "trace": [thought]}


def node_memory_retrieval(state: AgentState) -> dict:
    session_id = state["session_id"]
    question = state["question"]

    memory_context = retrieve_relevant_memories(session_id, question)
    memory_messages = _parse_memory_to_messages(memory_context)

    thought = _thought(
        "memory_retrieval",
        "Memory context loaded",
        {"has_memory": bool(memory_context.strip())},
    )
    return {
        "memory_context": memory_context,
        "memory_messages": memory_messages,
        "last_thought": thought,
        "trace": [thought],
    }


def node_query_expansion(state: AgentState) -> dict:
    current_question = state["current_question"]
    memory_context = state["memory_context"]

    expanded = expand_query_sync(
        question=current_question,
        memory_context=memory_context,
        n=4,
    )

    thought = _thought(
        "query_expansion",
        f"Generated {len(expanded)} query variants",
        {"queries": expanded},
    )
    return {
        "expanded_queries": expanded,
        "last_thought": thought,
        "trace": [thought],
    }


def node_hybrid_retrieval(state: AgentState) -> dict:
    expanded_queries = state["expanded_queries"]
    document_ids = state["document_ids"]
    attempt = state["attempt"]

    embeddings = list(
        ThreadPoolExecutor(max_workers=min(len(expanded_queries), 8)).map(
            create_embedding, expanded_queries
        )
    )

    def search_one(args):
        query, emb = args
        return hybrid_search(
            query=query,
            query_embedding=emb,
            document_ids=document_ids,
            top_k=TOP_K,
        )

    combos = list(zip(expanded_queries, embeddings))

    all_result_sets = []
    with ThreadPoolExecutor(max_workers=min(len(combos), 8)) as executor:
        futures = {executor.submit(search_one, combo): combo for combo in combos}
        for future in as_completed(futures):
            try:
                all_result_sets.append(future.result())
            except Exception as e:
                print(f"[node_hybrid_retrieval] search error: {e}")

    rrf_k = 60
    merged = {}
    for result_set in all_result_sets:
        ranked = sorted(
            result_set, key=lambda x: x.get("hybrid_score", 0), reverse=True
        )
        for rank, row in enumerate(ranked):
            chunk_id = row["id"]
            rrf_score = 1 / (rrf_k + rank + 1)
            if chunk_id not in merged:
                merged[chunk_id] = {**row, "rrf_score": 0}
            merged[chunk_id]["rrf_score"] += rrf_score

    child_results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)

    thought = _thought(
        "hybrid_retrieval",
        f"Retrieved {len(child_results)} unique chunks (attempt {attempt + 1})",
        {"chunks": len(child_results), "queries": len(expanded_queries)},
    )
    return {
        "child_results": child_results,
        "last_thought": thought,
        "trace": [thought],
    }


def node_parent_expansion(state: AgentState) -> dict:
    child_results = state["child_results"]
    document_ids = state["document_ids"]

    grouped = group_chunks_by_parent(child_results)
    top_parents = select_top_parents(grouped, top_n=TOP_K)
    parent_ids = [p["parent_id"] for p in top_parents]

    with Session(engine) as db:
        expanded = expand_parent_chunks(
            session=db,
            parent_ids=parent_ids,
            document_ids=document_ids,
        )

    context = build_parent_context_blocks(top_parents, expanded)

    thought = _thought(
        "parent_expansion",
        f"Expanded {len(top_parents)} parent context blocks",
        {"parents": len(top_parents)},
    )
    return {
        "context": context,
        "last_thought": thought,
        "trace": [thought],
    }


def node_answer_generation(state: AgentState) -> dict:
    question = state["current_question"]
    context = state["context"]
    memory_messages_dicts = state["memory_messages"]
    attempt = state["attempt"]

    memory_messages = [
        (
            HumanMessage(content=m["content"])
            if m["type"] == "human"
            else AIMessage(content=m["content"])
        )
        for m in memory_messages_dicts
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful document assistant.\n"
                    "Use the retrieved document context and conversation memory to answer.\n"
                    "Only answer when supported by context.\n"
                    'If uncertain say: "I could not find that in the documents."\n'
                    "When citing, include page number and only relevant text excerpt in a Citations section. Do not generate answers with some excerpts, and provide some other excerpts in the citations section. Make sure the right excerpts/citations are returned."
                )
            ),
            MessagesPlaceholder(variable_name="memory_turns"),
            HumanMessage(
                content=(
                    f"==================================================\n"
                    f"DOCUMENT CONTEXT\n"
                    f"==================================================\n"
                    f"{context}\n\n"
                    f"==================================================\n"
                    f"QUESTION\n"
                    f"==================================================\n"
                    f"{question}"
                )
            ),
        ]
    )

    formatted = prompt.format_messages(memory_turns=memory_messages)

    messages = [
        {
            "role": (
                "system"
                if isinstance(m, SystemMessage)
                else "user" if isinstance(m, HumanMessage) else "assistant"
            ),
            "content": m.content,
        }
        for m in formatted
    ]

    response = generate_response(messages=messages, temperature=0)
    full_answer = response.choices[0].message.content

    thought = _thought(
        "answer_generation",
        f"Answer generated (attempt {attempt + 1})",
        {"length": len(full_answer)},
    )
    return {
        "full_answer": full_answer,
        "last_thought": thought,
        "trace": [thought],
    }


def node_evaluation(state: AgentState) -> dict:
    question = state["current_question"]
    context = state["context"]
    full_answer = state["full_answer"]
    attempt = state["attempt"]

    eval_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are evaluating a RAG answer for groundedness and quality. "
                    "Return ONLY valid JSON, no markdown fences."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Retrieved Context (truncated):\n{context[:4000]}\n\n"
                    f"Answer:\n{full_answer}\n\n"
                    "Return JSON:\n"
                    "{\n"
                    '  "supported": true | false,\n'
                    '  "reason": "one sentence",\n'
                    '  "retry": true | false,\n'
                    '  "suggested_query": "specific gap-targeting query if retry=true, else null",\n'
                    '  "confidence": 0-10\n'
                    "}\n\n"
                    "Guidelines:\n"
                    "- supported=true if answer is directly grounded in context\n"
                    "- retry=true if retrieval appears insufficient\n"
                    "- suggested_query must target the specific gap, not restate the original\n"
                    "- confidence: how completely the answer addresses the question"
                )
            ),
        ]
    )

    formatted = eval_prompt.format_messages()
    messages = [
        {
            "role": "system" if isinstance(m, SystemMessage) else "user",
            "content": m.content,
        }
        for m in formatted
    ]

    response = generate_response(messages=messages, temperature=0)
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    print({"raw": raw})
    try:
        evaluation = json.loads(raw)
    except Exception:
        evaluation = {
            "supported": True,
            "reason": "Evaluation parse failed — accepting answer",
            "retry": False,
            "suggested_query": None,
            "confidence": 6,
        }

    thought = _thought(
        "evaluation",
        f"Confidence {evaluation.get('confidence', '?')}/10 — "
        f"{'supported' if evaluation.get('supported') else 'unsupported'}",
        {"eval_result": evaluation, "attempt": attempt + 1},
    )
    return {
        "eval_result": evaluation,
        "attempt": attempt + 1,
        "last_thought": thought,
        "trace": [thought],
    }


def node_prepare_retry(state: AgentState) -> dict:
    suggested = state["eval_result"]["suggested_query"]
    thought = _thought(
        "prepare_retry",
        f"Retrying with refined query (attempt {state['attempt'] + 1})",
        {"suggested_query": suggested},
    )
    return {
        "current_question": suggested,
        "last_thought": thought,
        "trace": [thought],
    }


def node_save_and_return(state: AgentState) -> dict:
    session_id = state["session_id"]
    question = state["question"]
    full_answer = state["full_answer"]
    query_embedding = state["query_embedding"]
    document_ids = state["document_ids"]
    trace = state["trace"]

    cache_key = session_id + "|" + "|".join(sorted(document_ids))

    set_exact_cache(cache_key, question, full_answer)
    set_semantic_cache(cache_key, question, [query_embedding], full_answer)

    asyncio.run(
        save_conversation_turn(
            session_id=session_id,
            question=question,
            answer=full_answer,
            thoughts=trace,
        )
    )

    thought = _thought("save_and_return", "Turn saved to memory")
    return {
        "last_thought": thought,
        "trace": [thought],
    }


# =====================================================
# EDGES
# =====================================================


def route_after_cache(state: AgentState) -> str:
    if state.get("cache_hit"):
        return "save_cache_hit"
    return "memory_retrieval"


def route_after_evaluation(state: AgentState) -> str:
    evaluation = state["eval_result"] or {}
    attempt = state["attempt"]

    should_retry = (
        evaluation.get("retry", False)
        and evaluation.get("confidence", 10) < CONFIDENCE_THRESHOLD
        and attempt < MAX_RETRIES
        and evaluation.get("suggested_query")
    )

    return "prepare_retry" if should_retry else "save_and_return"


# =====================================================
# GRAPH
# =====================================================


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("cache_check", RunnableLambda(timer(node_cache_check)))
    graph.add_node("save_cache_hit", RunnableLambda(timer(node_save_cache_hit)))
    graph.add_node("memory_retrieval", RunnableLambda(timer(node_memory_retrieval)))
    graph.add_node("query_expansion", RunnableLambda(timer(node_query_expansion)))
    graph.add_node("hybrid_retrieval", RunnableLambda(timer(node_hybrid_retrieval)))
    graph.add_node("parent_expansion", RunnableLambda(timer(node_parent_expansion)))
    graph.add_node("answer_generation", RunnableLambda(timer(node_answer_generation)))
    graph.add_node("evaluation", RunnableLambda(timer(node_evaluation)))
    graph.add_node("prepare_retry", RunnableLambda(timer(node_prepare_retry)))
    graph.add_node("save_and_return", RunnableLambda(timer(node_save_and_return)))

    graph.set_entry_point("cache_check")

    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "save_cache_hit": "save_cache_hit",
            "memory_retrieval": "memory_retrieval",
        },
    )

    graph.add_edge("save_cache_hit", END)
    graph.add_edge("memory_retrieval", "query_expansion")
    graph.add_edge("query_expansion", "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval", "parent_expansion")
    graph.add_edge("parent_expansion", "answer_generation")
    graph.add_edge("answer_generation", "evaluation")

    graph.add_conditional_edges(
        "evaluation",
        route_after_evaluation,
        {
            "prepare_retry": "prepare_retry",
            "save_and_return": "save_and_return",
        },
    )

    graph.add_edge("prepare_retry", "query_expansion")
    graph.add_edge("save_and_return", END)

    return graph.compile()


agent_graph = build_graph()


# =====================================================
# ENTRY POINT
# =====================================================


@timer
async def run_agent(
    session_id: str,
    question: str,
    document_ids: list[str] | None,
    stream: bool,
) -> str | AsyncGenerator[str, None]:
    if not document_ids:
        from app.models.session_document import SessionDocument

        with Session(engine) as db:
            rows = db.exec(
                select(SessionDocument.document_id).where(
                    SessionDocument.session_id == session_id
                )
            ).all()
            document_ids = list(rows)

    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "current_question": question,
        "document_ids": document_ids,
        "query_embedding": [],
        "memory_context": "",
        "memory_messages": [],
        "expanded_queries": [],
        "child_results": [],
        "context": "",
        "full_answer": "",
        "eval_result": None,
        "attempt": 0,
        "trace": [],
        "last_thought": {},
        "cache_hit": False,
        "cache_answer": None,
    }

    if stream:
        return _stream_agent(initial_state)
    return await _run_agent_sync(initial_state)


async def _run_agent_sync(initial_state: AgentState) -> str:
    final_state = await asyncio.to_thread(lambda: agent_graph.invoke(initial_state))
    if final_state.get("cache_hit"):
        return final_state["cache_answer"]
    return final_state["full_answer"]


async def _stream_agent(initial_state: AgentState) -> AsyncGenerator[str, None]:
    async for snapshot in agent_graph.astream(initial_state):
        for node_name, state in snapshot.items():
            if node_name == "__end__":
                yield done_event()
                return

            thought = state.get("last_thought", {})
            if thought:
                yield format_sse(
                    json.dumps(
                        {
                            "thinking": thought.get("node", node_name),
                            "message": thought.get("message", ""),
                            **(thought.get("data") or {}),
                        }
                    )
                )

            if node_name == "answer_generation":
                answer = state.get("full_answer", "")
                if answer:
                    yield token_event(answer)

            if node_name == "save_cache_hit":
                cached = state.get("cache_answer", "")
                if cached:
                    yield token_event(cached)
                yield done_event()
                return

    yield done_event()
