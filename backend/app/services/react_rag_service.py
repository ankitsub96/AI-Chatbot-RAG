import json
import time
import asyncio
from typing import Annotated, AsyncGenerator
import operator

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

from app.utils.helpers import thinking, token_event, done_event, timer
from app.services.shared_rag_nodes import (
    check_cache_agent,
    serve_cache_agent,
    retrieve_memory_agent,
    save_turn_agent,
    route_cache_agent,
    reason_agent,
    act_agent,
    respond_agent,
    route_loop_agent,
    evaluate_agent,
)
from app.models.chunk_types import StrictnessLevel, AnyToolResult, NodeOutput
from app.models.agent_states import ReactState, ReactCacheCheckOutput, ReactNodeOutput
from app.config.settings import TOP_K, REACT_MAX_ITERATIONS, CONFIDENCE_THRESHOLD
from app.services.vector_search_service import create_embedding
from app.services.database import engine
from sqlmodel import Session, select

# =====================================================
# STATE
# =====================================================


# =====================================================
# NODES
# =====================================================


# =====================================================
# ROUTING
# =====================================================


def route_after_evaluation(state: ReactState) -> NodeOutput:
    evaluation = state["eval_result"] or {}
    iteration = state["iteration"]

    should_retry = (
        evaluation.get("retry", False)
        and evaluation.get("confidence", 10) < CONFIDENCE_THRESHOLD
        and iteration < REACT_MAX_ITERATIONS
        and evaluation.get("suggested_query")
    )
    return "reason" if should_retry else "save_turn"


# =====================================================
# GRAPH
# =====================================================


def build_graph() -> StateGraph:
    graph = StateGraph(ReactState)

    graph.add_node("check_cache", RunnableLambda(timer(check_cache_agent)))
    graph.add_node("save_cache_hit", RunnableLambda(timer(serve_cache_agent)))
    graph.add_node("retrieve_memory", RunnableLambda(timer(retrieve_memory_agent)))
    graph.add_node("reason", RunnableLambda(timer(reason_agent)))
    graph.add_node("act", RunnableLambda(timer(act_agent)))
    graph.add_node("respond", RunnableLambda(timer(respond_agent)))
    graph.add_node("evaluate", RunnableLambda(timer(evaluate_agent)))
    graph.add_node("save_turn", RunnableLambda(timer(save_turn_agent)))

    graph.set_entry_point("check_cache")

    graph.add_conditional_edges(
        "check_cache",
        route_cache_agent,
        {
            "save_cache_hit": "save_cache_hit",
            "retrieve_memory": "retrieve_memory",
        },
    )

    graph.add_edge("save_cache_hit", END)
    graph.add_edge("retrieve_memory", "reason")

    graph.add_conditional_edges(
        "reason",
        route_loop_agent,
        {
            "act": "act",
            "respond": "respond",
        },
    )

    graph.add_edge("act", "reason")
    graph.add_edge("respond", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "reason": "reason",
            "save_turn": "save_turn",
        },
    )

    graph.add_edge("save_turn", END)

    return graph.compile()


react_graph = build_graph()


# =====================================================
# ENTRY POINT
# =====================================================


@timer
async def run_react_agent(
    session_id: str,
    question: str,
    document_ids: list[str] | None,
    stream: bool,
    use_web: bool = False,
    strictness: StrictnessLevel = "balanced",
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

    initial_state: ReactState = {
        "session_id": session_id,
        "question": question,
        "document_ids": document_ids,
        "use_web": use_web,
        "strictness": strictness,
        "memory_context": "",
        "memory_messages": [],
        "query_embedding": [],
        "step_contexts": [],
        "tool_history": [],
        "collected_context": "",
        "full_answer": "",
        "eval_result": None,
        "iteration": 0,
        "trace": [],
        "last_thought": {},
        "cache_hit": False,
        "cache_answer": None,
    }

    if stream:
        return _stream_react(initial_state)
    return await _run_react_sync(initial_state)


async def _run_react_sync(initial_state: ReactState) -> str:
    final_state = await asyncio.to_thread(lambda: react_graph.invoke(initial_state))
    if final_state.get("cache_hit"):
        return final_state["cache_answer"]
    return final_state["full_answer"]


async def _stream_react(initial_state: ReactState) -> AsyncGenerator[str, None]:
    async for snapshot in react_graph.astream(initial_state):
        for node_name, state in snapshot.items():
            if node_name == "__end__":
                yield done_event()
                return

            thought = state.get("last_thought", {})
            if thought:
                yield thinking(
                    stage=thought.get("node", node_name),
                    message=thought.get("message", ""),
                    data=thought.get("data") or {},
                )

            if node_name == "answer_generation":
                answer = state.get("full_answer", "")
                if answer:
                    yield token_event(answer)
            if node_name == "respond":
                answer = state.get("full_answer", "")
                if answer:
                    yield token_event(answer)
            if node_name == "save_cache_hit":
                cached = state.get("cache_answer", "")
                if cached:
                    yield token_event(cached)
                yield done_event()
                return
            if node_name == "serve_cache":
                cached = state.get("cache_answer", "")
                if cached:
                    yield token_event(cached)
                yield done_event()
                return

    yield done_event()
