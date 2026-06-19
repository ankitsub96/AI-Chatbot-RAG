import json
import time
import asyncio
from typing import Annotated, AsyncGenerator
import operator

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from sqlmodel import Session, select

from app.services.research_tools import (
    tool_question_decomposer,
    tool_document_search,
    tool_web_search,
    tool_answer_generator,
    tool_answer_synthesizer,
    tool_answer_evaluator,
    run_tools_parallel,
)
from app.services.memory_service import save_conversation_turn
from app.services.semantic_cache_service import (
    get_exact_cache,
    get_semantic_cache,
    set_exact_cache,
    set_semantic_cache,
)
from app.utils.helpers import thinking, token_event, done_event, timer
from app.models.chunk_types import StrictnessLevel, NodeOutput, AnyToolResult
from app.models.agent_states import PlannerState, PlannerNodeOutput
from app.config.settings import PLANNER_MAX_STEPS, CONFIDENCE_THRESHOLD
from app.services.vector_search_service import create_embedding
from app.services.database import engine
from app.services.shared_rag_nodes import (
    check_cache_agent,
    serve_cache_agent,
    retrieve_memory_agent,
    save_turn_agent,
    route_cache_agent,
    create_plan_agent,
    search_parallel_agent,
    answer_parallel_agent,
    synthesize_agent,
    evaluate_agent,
)

# =====================================================
# GRAPH
# =====================================================


def build_graph() -> StateGraph:
    graph = StateGraph(PlannerState)

    graph.add_node("check_cache", RunnableLambda(timer(check_cache_agent)))
    graph.add_node("save_cache_hit", RunnableLambda(timer(serve_cache_agent)))
    graph.add_node("retrieve_memory", RunnableLambda(timer(retrieve_memory_agent)))
    graph.add_node("create_plan", RunnableLambda(timer(create_plan_agent)))
    graph.add_node("search_parallel", RunnableLambda(timer(search_parallel_agent)))
    graph.add_node("answer_parallel", RunnableLambda(timer(answer_parallel_agent)))
    graph.add_node("synthesize", RunnableLambda(timer(synthesize_agent)))
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
    graph.add_edge("retrieve_memory", "create_plan")
    graph.add_edge("create_plan", "search_parallel")
    graph.add_edge("search_parallel", "answer_parallel")
    graph.add_edge("answer_parallel", "synthesize")
    graph.add_edge("synthesize", "evaluate")
    graph.add_edge("evaluate", "save_turn")
    graph.add_edge("save_turn", END)

    return graph.compile()


planner_graph = build_graph()


# =====================================================
# ENTRY POINT
# =====================================================


@timer
async def run_planner_agent(
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

    initial_state: PlannerState = {
        "session_id": session_id,
        "question": question,
        "document_ids": document_ids,
        "use_web": use_web,
        "strictness": strictness,
        "memory_context": "",
        "memory_messages": [],
        "query_embedding": [],
        "plan": [],
        "step_contexts": [],
        "sub_answers": [],
        "full_answer": "",
        "eval_result": None,
        "trace": [],
        "last_thought": {},
        "cache_hit": False,
        "cache_answer": None,
    }

    if stream:
        return _stream_planner(initial_state)
    return await _run_planner_sync(initial_state)


async def _run_planner_sync(initial_state: PlannerState) -> str:
    final_state = await asyncio.to_thread(lambda: planner_graph.invoke(initial_state))
    if final_state.get("cache_hit"):
        return final_state["cache_answer"]
    return final_state["full_answer"]


async def _stream_planner(initial_state: PlannerState) -> AsyncGenerator[str, None]:
    async for snapshot in planner_graph.astream(initial_state):
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

            if node_name == "answer_synthesis":
                answer = state.get("full_answer", "")
                if answer:
                    yield token_event(answer)

            if node_name == "save_cache_hit":
                cached = state.get("cache_answer", "")
                if cached:
                    yield token_event(cached)
                yield done_event()
                return
            if node_name == "synthesize":  # was "answer_synthesis"
                answer = state.get("full_answer", "")
                if answer:
                    yield token_event(answer)

            if node_name == "serve_cache":  # was "save_cache_hit"
                cached = state.get("cache_answer", "")
                if cached:
                    yield token_event(cached)
                yield done_event()
                return

    yield done_event()
