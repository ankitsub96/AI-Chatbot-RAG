import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from sqlmodel import Session, select
from app.services.semantic_cache_service import (
    get_exact_cache,
    get_semantic_cache,
    set_exact_cache,
    set_semantic_cache,
)
from app.services.memory_service import save_conversation_turn
from app.services.vector_search_service import create_embedding
from app.utils.helpers import (
    rag_thought,
    rag_cache_key,
    _parse_memory_to_messages,
    timer,
    thinking,
    token_event,
    done_event,
)
from app.models.chunk_types import (
    CacheCheckOutput,
    CacheHitOutput,
    MemoryRetrievalOutput,
    SaveAndReturnOutput,
    StrictnessLevel,
    AnyToolResult,
    NodeOutput,
)
from app.models.agent_states import (
    ReactState,
    PlannerState,
    HybridState,
    StepState,
    ReactNodeOutput,
    PlannerNodeOutput,
    StepReactOutput,
    StepToolOutput,
    StepAnswerOutput,
    PlanCreationOutput,
    StepExecutionOutput,
    SynthesisOutput,
    EvaluationOutput,
)
from app.services.research_tools import (
    tool_memory_search,
    tool_document_search,
    tool_web_search,
    tool_query_expander,
    tool_question_decomposer,
    tool_answer_generator,
    tool_answer_synthesizer,
    tool_answer_evaluator,
    run_tools_parallel,
    tool_batch_answer_generator,
)
from app.config.settings import (
    REACT_MAX_ITERATIONS,
    PLANNER_MAX_STEPS,
    CONFIDENCE_THRESHOLD,
)
from app.services.llm_service import generate_response


def check_cache_agent(state: dict) -> CacheCheckOutput:
    question = state["question"]
    session_id = state["session_id"]
    document_ids = state["document_ids"]
    key = rag_cache_key(session_id, document_ids)

    exact = get_exact_cache(key, question)
    if exact:
        return {
            "cache_hit": True,
            "cache_answer": exact,
            "last_thought": rag_thought("cache_check", "Exact cache hit"),
            "trace": [rag_thought("cache_check", "Exact cache hit")],
        }

    embedding = create_embedding(question)
    semantic = get_semantic_cache(key, embedding[0].tolist())
    if semantic:
        return {
            "cache_hit": True,
            "cache_answer": semantic,
            "query_embedding": embedding[0].tolist(),
            "last_thought": rag_thought("cache_check", "Semantic cache hit"),
            "trace": [rag_thought("cache_check", "Semantic cache hit")],
        }

    return {
        "cache_hit": False,
        "query_embedding": embedding[0].tolist(),
        "last_thought": rag_thought("cache_check", "Cache miss"),
        "trace": [rag_thought("cache_check", "Cache miss")],
    }


def serve_cache_agent(state: dict) -> CacheHitOutput:
    thought = rag_thought("cache_check", "Returning cached answer")
    return {
        "cache_answer": state.get("cache_answer", ""),
        "last_thought": thought,
        "trace": [thought],
    }


def retrieve_memory_agent(state: dict) -> MemoryRetrievalOutput:
    from app.services.research_tools import tool_memory_search

    result = tool_memory_search(state["session_id"], state["question"])
    memory_messages = _parse_memory_to_messages(result["full_context"])
    thought = rag_thought("memory_retrieval", result["summary"])
    return {
        "memory_context": result["full_context"],
        "memory_messages": memory_messages,
        "last_thought": thought,
        "trace": [thought],
    }


def save_turn_agent(state: dict, pipeline_name: str = "") -> SaveAndReturnOutput:
    session_id = state["session_id"]
    question = state["question"]
    full_answer = state["full_answer"]
    query_embedding = state["query_embedding"]
    document_ids = state["document_ids"]
    trace = state["trace"]

    key = rag_cache_key(session_id, document_ids)
    set_exact_cache(key, question, full_answer)
    set_semantic_cache(key, question, [query_embedding], full_answer)

    asyncio.run(
        save_conversation_turn(
            session_id=session_id,
            question=question,
            answer=full_answer,
            thoughts=trace,
        )
    )

    thought = rag_thought(
        "save_and_return",
        f"Turn saved to memory{' — ' + pipeline_name if pipeline_name else ''}",
    )
    return {"last_thought": thought, "trace": [thought]}


def route_cache_agent(state: dict) -> str:
    return "save_cache_hit" if state.get("cache_hit") else "retrieve_memory"


def reason_agent(state: ReactState) -> NodeOutput:
    question = state["question"]
    memory_context = state["memory_context"]
    tool_history = state["tool_history"]
    collected_context = state["collected_context"]
    iteration = state["iteration"]
    use_web = state["use_web"]
    strictness = state["strictness"]

    tool_history_str = (
        json.dumps(tool_history[-10:], indent=2) if tool_history else "None"
    )

    strictness_instruction = {
        "strict": "Use ONLY document context. Do not infer beyond what is retrieved.",
        "balanced": "Prefer document context. You may connect ideas using general knowledge.",
        "creative": "Use document context and general knowledge freely.",
    }.get(strictness, "Prefer document context.")

    available_tools = ["document_search", "query_expander"]
    if use_web:
        available_tools.append("web_search")

    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a ReAct research agent.\n{strictness_instruction}\n"
                "Decide the next action based on tool history and collected context.\n"
                f"Available tools: {available_tools}\n"
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "action": "document_search" | "query_expander" | "web_search" | "finish",\n'
                '  "args": {{"query": "..."}},\n'
                '  "reason": "one sentence"\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Memory:\n{memory_context or 'None'}\n\n"
                f"Tool history:\n{tool_history_str}\n\n"
                f"Collected context length: {len(collected_context)} chars\n"
                f"Iteration: {iteration + 1}/{REACT_MAX_ITERATIONS}"
            ),
        },
    ]

    try:
        response = generate_response(messages=prompt, temperature=0)
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        decision = json.loads(raw)
    except Exception:
        decision = {"action": "finish", "args": {}, "reason": "Parse failed"}

    thought = rag_thought(
        "react",
        f"Iteration {iteration + 1} — action: {decision.get('action')}",
        {"decision": decision},
    )
    return {
        "last_thought": thought,
        "trace": [thought],
        "tool_history": [{"iteration": iteration, "decision": decision}],
    }


def act_agent(state: ReactState) -> NodeOutput:
    tool_history = state["tool_history"]
    document_ids = state["document_ids"]
    use_web = state["use_web"]

    last = tool_history[-1]["decision"] if tool_history else {}
    action = last.get("action", "finish")
    args = last.get("args", {})

    result: AnyToolResult | None = None

    if action == "document_search":
        query = args.get("query", state["question"])
        result = tool_document_search(query=query, document_ids=document_ids)

    elif action == "query_expander":
        query = args.get("query", state["question"])
        exp_result = tool_query_expander(
            question=query, memory_context=state["memory_context"]
        )
        queries = exp_result["data"].get("queries", [query])
        tasks = [
            (tool_document_search, (), {"query": q, "document_ids": document_ids})
            for q in queries
        ]
        search_results = run_tools_parallel(tasks)
        combined_context = "\n\n".join(
            r["full_context"] for r in search_results if r["success"]
        )
        result = exp_result
        result = {**result, "full_context": combined_context}

    elif action == "web_search" and use_web:
        query = args.get("query", state["question"])
        result = tool_web_search(query=query)

    new_context = state["collected_context"]
    tool_record = {"action": action, "args": args, "result": "no result"}

    if result:
        new_context = state["collected_context"] + "\n\n" + result["full_context"]
        tool_record["result"] = result["truncated_result"]

    thought = rag_thought(
        "tool_executor",
        f"Executed {action}",
        {"truncated_result": tool_record["result"][:200]},
    )
    return {
        "collected_context": new_context,
        "tool_history": [tool_record],
        "last_thought": thought,
        "trace": [thought],
    }


def respond_agent(state: ReactState) -> NodeOutput:
    from app.services.research_tools import tool_answer_generator

    result = tool_answer_generator(
        question=state["question"],
        context=state["collected_context"],
        memory_context=state["memory_context"],
        strictness=state["strictness"],
    )
    thought = rag_thought(
        "answer_generation", "Answer generated", {"length": len(result["full_context"])}
    )
    return {
        "full_answer": result["full_context"],
        "last_thought": thought,
        "trace": [thought],
    }


def route_loop_agent(state: ReactState) -> NodeOutput:
    tool_history = state["tool_history"]
    iteration = state["iteration"]
    last = tool_history[-1]["decision"] if tool_history else {}
    action = last.get("action", "finish")

    if action == "finish" or iteration >= REACT_MAX_ITERATIONS:
        return "respond"
    return "act"


def reason_step_agent(state: StepState) -> StepReactOutput:
    sub_question = state["sub_question"]
    memory_context = state["memory_context"]
    tool_history = state["tool_history"]
    iteration = state["iteration"]
    use_web = state["use_web"]
    strictness = state["strictness"]

    tool_history_str = (
        json.dumps(tool_history[-5:], indent=2) if tool_history else "None"
    )

    strictness_instruction = {
        "strict": "Use ONLY document context. Do not infer beyond what is retrieved.",
        "balanced": "Prefer document context. You may connect ideas using general knowledge.",
        "creative": "Use document context and general knowledge freely.",
    }.get(strictness, "Prefer document context.")

    available_tools = ["document_search"]
    if use_web:
        available_tools.append("web_search")

    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a ReAct research agent handling one sub-question.\n"
                f"{strictness_instruction}\n"
                f"Available tools: {available_tools}\n"
                "Return ONLY valid JSON:\n"
                "{\n"
                '  "action": "document_search" | "web_search" | "finish",\n'
                '  "args": {"query": "..."},\n'
                '  "reason": "one sentence"\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Sub-question: {sub_question}\n\n"
                f"Memory:\n{memory_context or 'None'}\n\n"
                f"Tool history:\n{tool_history_str}\n\n"
                f"Collected context length: {len(state['collected_context'])} chars\n"
                f"Iteration: {iteration + 1}/{REACT_MAX_ITERATIONS}"
            ),
        },
    ]

    try:
        response = generate_response(messages=prompt, temperature=0)
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        decision = json.loads(raw)
    except Exception:
        decision = {"action": "finish", "args": {}, "reason": "Parse failed"}

    thought = rag_thought(
        "step_react",
        f"Step iteration {iteration + 1} — action: {decision.get('action')}",
        {"decision": decision},
    )
    return {
        "tool_history": [{"iteration": iteration, "decision": decision}],
        "last_thought": thought,
        "trace": [thought],
    }


def act_step_agent(state: StepState) -> StepToolOutput:
    tool_history = state["tool_history"]
    document_ids = state["document_ids"]
    use_web = state["use_web"]
    # 1. Get and increment the active iteration counter
    current_iteration = state.get("iteration", 0)
    next_iteration = current_iteration + 1

    last = tool_history[-1]["decision"] if tool_history else {}
    action = last.get("action", "finish")
    args = last.get("args", {})

    result: AnyToolResult | None = None

    if action == "document_search":
        query = args.get("query", state["sub_question"])
        result = tool_document_search(query=query, document_ids=document_ids)

    elif action == "web_search" and use_web:
        query = args.get("query", state["sub_question"])
        result = tool_web_search(query=query)

    new_context = state["collected_context"]
    # 2. Keep the 'decision' key structure intact so the next iteration can read it
    tool_record = {"decision": {"action": action, "args": args}, "result": "no result"}

    if result:
        new_context = state["collected_context"] + "\n\n" + result["full_context"]
        tool_record["result"] = result["truncated_result"]

    thought = rag_thought(
        "step_tool",
        f"Step executed {action} (Iteration {next_iteration})",
        {"truncated_result": tool_record["result"][:200]},
    )

    # 3. CRITICAL: You MUST return the incremented iteration so LangGraph can save it!
    return {
        "collected_context": new_context,
        "tool_history": [tool_record],  # Appended seamlessly if using a list reducer
        "iteration": next_iteration,  # <-- Save state update
        "last_thought": thought,
        "trace": [thought],
    }


def respond_step_agent(state: StepState) -> StepAnswerOutput:
    result = tool_answer_generator(
        question=state["sub_question"],
        context=state["collected_context"],
        memory_context=state["memory_context"],
        strictness=state["strictness"],
    )
    thought = rag_thought(
        "step_answer",
        f"Step answer generated",
        {"length": len(result["full_context"])},
    )
    return {
        "sub_answer": result["full_context"],
        "last_thought": thought,
    }


def route_step_agent(state: StepState) -> str:
    tool_history = state["tool_history"]
    iteration = state.get("iteration", 0)
    last = tool_history[-1]["decision"] if tool_history else {}
    action = last.get("action", "finish")

    if action == "finish" or iteration >= REACT_MAX_ITERATIONS:
        return "respond"
    return "act"


def create_plan_agent(state: PlannerState) -> NodeOutput:
    result = tool_question_decomposer(
        question=state["question"],
        max_steps=PLANNER_MAX_STEPS,
    )
    steps = result["data"].get("steps", [state["question"]])
    thought = rag_thought(
        "plan_creation",
        f"Plan created — {len(steps)} steps",
        {"plan": steps},
    )
    return {
        "plan": steps,
        "last_thought": thought,
        "trace": [thought],
    }


def search_parallel_agent(state: PlannerState) -> NodeOutput:
    plan = state["plan"]
    document_ids = state["document_ids"]
    use_web = state["use_web"]

    search_tasks = [
        (tool_document_search, (), {"query": step, "document_ids": document_ids})
        for step in plan
    ]

    if use_web:
        search_tasks += [(tool_web_search, (), {"query": step}) for step in plan]

    search_results = run_tools_parallel(search_tasks)

    # pair each step with its doc search result context
    step_contexts = []
    for i, step in enumerate(plan):
        doc_result = search_results[i]
        ctx = doc_result["full_context"] if doc_result["success"] else ""
        if use_web:
            web_result = search_results[len(plan) + i]
            if web_result["success"]:
                ctx += "\n\n" + web_result["full_context"]
        step_contexts.append(ctx)

    thought = rag_thought(
        "parallel_execution",
        f"Ran {len(search_tasks)} search tools in parallel",
        {"tools": len(search_tasks)},
    )
    return {
        "step_contexts": step_contexts,
        "last_thought": thought,
        "trace": [thought],
    }


def answer_parallel_agent(state: PlannerState) -> NodeOutput:
    plan = state["plan"]
    step_contexts = state["step_contexts"]
    memory_context = state["memory_context"]
    strictness = state["strictness"]

    batch_results = tool_batch_answer_generator(
        steps=plan,
        step_contexts=step_contexts,
        memory_context=memory_context,
        strictness=strictness,
    )

    missing_indices = [i for i in range(len(plan)) if i not in batch_results]

    fallback_results = {}
    if missing_indices:
        tasks = [
            (
                tool_answer_generator,
                (),
                {
                    "question": plan[i],
                    "context": step_contexts[i] if i < len(step_contexts) else "",
                    "memory_context": memory_context,
                    "strictness": strictness,
                },
            )
            for i in missing_indices
        ]
        results = run_tools_parallel(tasks)
        for idx, r in zip(missing_indices, results):
            if r["success"]:
                fallback_results[idx] = r["full_context"]

    sub_answers = [
        batch_results.get(i) or fallback_results.get(i, "") for i in range(len(plan))
    ]
    sub_answers = [a for a in sub_answers if a]

    message = f"Generated {len(batch_results)} via batch call"
    if missing_indices:
        message += f", {len(fallback_results)} via fallback"

    thought = rag_thought(
        "parallel_answer_gen",
        message,
        {
            "count": len(sub_answers),
            "batched": len(batch_results),
            "fallback": len(fallback_results),
        },
    )
    return {
        "sub_answers": sub_answers,
        "last_thought": thought,
        "trace": [thought],
    }


def synthesize_agent(state: PlannerState) -> NodeOutput:
    # If using tool_batch_answer_generator, state['sub_answers'] will contain
    # a list of the dictionary structures returned by your batch generator
    sub_answers_data = state.get("sub_answers", [])
    question = state["question"]

    flat_answers = []
    master_sources = {}

    # 1. Deduplicate source chunks and compile answers using native iteration loops
    for entry in sub_answers_data:
        if isinstance(entry, dict) and "answer" in entry:
            flat_answers.append(entry["answer"])
            # Merge sources mapping dictionary safely
            step_sources = entry.get("sources", {})
            for page, snippet in step_sources.items():
                if page and snippet:
                    master_sources[page] = snippet
        elif isinstance(entry, str):
            # Fallback wrapper string compatibility handling
            flat_answers.append(entry)

    # 2. Invoke the token-minimized LLM text synthesizer tool
    result = tool_answer_synthesizer(
        original_question=question,
        sub_answers=flat_answers,
        sources_catalog=master_sources,
        strictness=state["strictness"],
    )

    thought = rag_thought(
        "answer_synthesis",
        f"Synthesized compiled pipeline output utilizing {len(master_sources)} target source quotes.",
        {"length": len(result["full_context"])},
    )
    return {
        "full_answer": result["full_context"],
        "last_thought": thought,
        "trace": [thought],
    }


def evaluate_agent(state: PlannerState) -> NodeOutput:
    combined_context = "\n\n".join(state.get("step_contexts", []))
    result = tool_answer_evaluator(
        question=state["question"],
        context=combined_context,
        answer=state["full_answer"],
    )
    thought = rag_thought(
        "evaluation",
        result["summary"],
        {"eval_result": result["data"]},
    )
    return {
        "eval_result": result["data"],
        "last_thought": thought,
        "trace": [thought],
    }


def execute_steps_agent(state: HybridState) -> StepExecutionOutput:
    plan = state["plan"]
    document_ids = state["document_ids"]
    use_web = state["use_web"]
    strictness = state["strictness"]
    memory_context = state["memory_context"]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    step_results = [None] * len(plan)

    with ThreadPoolExecutor(max_workers=min(len(plan), 8)) as executor:
        futures = {
            executor.submit(
                _run_step,
                step,
                document_ids,
                use_web,
                strictness,
                memory_context,
            ): idx
            for idx, step in enumerate(plan)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                step_results[idx] = future.result()
            except Exception as e:
                step_results[idx] = {
                    "sub_answer": f"Step failed: {e}",
                    "collected_context": "",
                }

    sub_answers = [r["sub_answer"] for r in step_results if r]
    step_contexts = [r["collected_context"] for r in step_results if r]

    thought = rag_thought(
        "step_execution",
        f"Completed {len(plan)} parallel ReAct steps",
        {"steps": len(plan)},
    )
    return {
        "sub_answers": sub_answers,
        "step_contexts": step_contexts,
        "last_thought": thought,
        "trace": [thought],
    }


# =====================================================
# STEP SUBGRAPH
# =====================================================


def build_step_subgraph() -> StateGraph:
    graph = StateGraph(StepState)

    graph.add_node("reason", RunnableLambda(timer(reason_step_agent)))
    graph.add_node("act", RunnableLambda(timer(act_step_agent)))
    graph.add_node("respond", RunnableLambda(timer(respond_step_agent)))

    graph.set_entry_point("reason")

    graph.add_conditional_edges(
        "reason",
        route_step_agent,
        {
            "act": "act",
            "respond": "respond",
        },
    )

    graph.add_edge("act", "reason")
    graph.add_edge("respond", END)

    return graph.compile()


step_subgraph = build_step_subgraph()


# =====================================================
# STEP RUNNER
# =====================================================


def _run_step(
    sub_question: str,
    document_ids: list[str],
    use_web: bool,
    strictness: StrictnessLevel,
    memory_context: str,
) -> StepState:
    initial: StepState = {
        "sub_question": sub_question,
        "document_ids": document_ids,
        "use_web": use_web,
        "strictness": strictness,
        "memory_context": memory_context,
        "tool_history": [],
        "collected_context": "",
        "sub_answer": "",
        "iteration": 0,
    }
    return step_subgraph.invoke(initial)
