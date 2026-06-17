import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from duckduckgo_search import DDGS

from app.services.vector_search_service import (
    create_embedding,
    hybrid_search,
    group_chunks_by_parent,
    select_top_parents,
    expand_parent_chunks,
    build_parent_context_blocks,
)
from app.services.llm_service import generate_response, expand_query_sync
from app.services.memory_service import retrieve_relevant_memories
from app.services.database import engine
from app.models.document_chunk import DocumentChunk
from app.models.chunk_types import (
    ChunkDict,
    HybridResultDict,
    GroupedParentDict,
    StrictnessLevel,
    ToolResult,
)
from app.config.settings import TOP_K, WEB_SEARCH_MAX_RESULTS
from sqlmodel import Session, select

# =====================================================
# TOOL RESULT
# =====================================================

# class ToolResult:
#     def __init__(
#         self,
#         name: str,
#         success: bool,
#         summary: str,
#         top_chunks: list[ChunkDict],
#         full_context: str,
#         data: dict | None,
#         truncated_result: str,
#     ):
#         self.name = name
#         self.success = success
#         self.summary = summary
#         self.top_chunks = top_chunks
#         self.full_context = full_context
#         self.data = data or {}
#         self.truncated_result = truncated_result[:500]
#         self.ts = time.time()


# =====================================================
# TOOL 1 — document_search
# =====================================================


def tool_document_search(
    query: str,
    document_ids: list[str],
    top_k: int = TOP_K,
) -> ToolResult:
    embedding = create_embedding(query)
    results = hybrid_search(
        query=query,
        query_embedding=embedding,
        document_ids=document_ids,
        top_k=top_k,
    )
    grouped = group_chunks_by_parent(results)
    top_parents = select_top_parents(grouped, top_n=top_k)
    parent_ids = [p["parent_id"] for p in top_parents]
    pages_found = list(
        {c.get("page") for p in top_parents for c in p["chunks"] if c.get("page")}
    )

    with Session(engine) as db:
        expanded = expand_parent_chunks(
            session=db,
            parent_ids=parent_ids,
            document_ids=document_ids,
        )

    full_context = build_parent_context_blocks(top_parents, expanded)
    top_chunks = [c for p in top_parents[:3] for c in p["chunks"]][:3]
    summary = f"Found {len(results)} chunks across pages {sorted(pages_found)} for query: '{query}'"
    truncated = (
        summary
        + " | top chunk: "
        + (top_chunks[0].get("text", "")[:300] if top_chunks else "none")
    )

    return ToolResult(
        name="document_search",
        success=bool(results),
        summary=summary,
        top_chunks=top_chunks,
        full_context=full_context,
        data={"pages_found": pages_found, "total_chunks": len(results)},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 2 — page_lookup
# =====================================================


def tool_page_lookup(
    page: int,
    document_ids: list[str],
) -> ToolResult:
    with Session(engine) as db:
        stmt = select(DocumentChunk).where(
            DocumentChunk.page == page,
            DocumentChunk.document_id.in_(document_ids),
        )
        rows = db.exec(stmt).all()

    chunks: list[ChunkDict] = [
        {
            "id": r.id,
            "document_id": r.document_id,
            "page": r.page,
            "section": r.section,
            "chunk_type": r.chunk_type,
            "parent_id": r.parent_id,
            "child_id": r.child_id,
            "chunk_index": r.chunk_index,
            "text": r.text,
            "chunk_metadata": r.chunk_metadata,
        }
        for r in rows
    ]

    full_context = "\n\n".join(c["text"] for c in chunks)
    top_chunks = chunks[:3]
    summary = f"Page {page}: found {len(chunks)} chunks"
    truncated = summary + " | " + (chunks[0]["text"][:300] if chunks else "none")

    return ToolResult(
        name="page_lookup",
        success=bool(chunks),
        summary=summary,
        top_chunks=top_chunks,
        full_context=full_context,
        data={"page": page, "total_chunks": len(chunks)},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 3 — web_search
# =====================================================


def tool_web_search(
    query: str,
) -> ToolResult:
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=WEB_SEARCH_MAX_RESULTS))
    except Exception as e:
        return ToolResult(
            name="web_search",
            success=False,
            summary=f"Web search failed: {e}",
            top_chunks=[],
            full_context="",
            data={},
            truncated_result=f"Web search failed: {e}"[:500],
        )

    snippets = [r.get("body", "") for r in raw if r.get("body")]
    urls = [r.get("href", "") for r in raw if r.get("href")]
    full_context = "\n\n".join(snippets)
    summary = f"Web search for '{query}' returned {len(snippets)} results"
    truncated = summary + " | " + (snippets[0][:300] if snippets else "none")

    return ToolResult(
        name="web_search",
        success=bool(snippets),
        summary=summary,
        top_chunks=[],
        full_context=full_context,
        data={"urls": urls},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 4 — query_expander
# =====================================================


def tool_query_expander(
    question: str,
    memory_context: str = "",
) -> ToolResult:
    expanded = expand_query_sync(question=question, memory_context=memory_context, n=4)
    summary = f"Expanded into {len(expanded)} queries"
    truncated = summary + " | " + str(expanded)[:300]

    return ToolResult(
        name="query_expander",
        success=bool(expanded),
        summary=summary,
        top_chunks=[],
        full_context="",
        data={"queries": expanded},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 5 — question_decomposer
# =====================================================


def tool_question_decomposer(
    question: str,
    max_steps: int = 5,
) -> ToolResult:
    prompt = f"""
You are a research planner. Decompose the question into at most {max_steps} independent sub-questions.
Each sub-question should be answerable on its own.
Return ONLY a JSON array of strings. No markdown, no preamble.

Question: {question}
"""
    try:
        response = generate_response(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        steps = json.loads(raw)
        if not isinstance(steps, list):
            steps = [question]
    except Exception:
        steps = [question]

    steps = steps[:max_steps]
    summary = f"Decomposed into {len(steps)} sub-questions"
    truncated = summary + " | " + str(steps)[:300]

    return ToolResult(
        name="question_decomposer",
        success=True,
        summary=summary,
        top_chunks=[],
        full_context="",
        data={"steps": steps},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 6 — answer_generator (per sub-question)
# =====================================================


def tool_answer_generator(
    question: str,
    context: str,
    memory_context: str = "",
    strictness: StrictnessLevel = "balanced",
) -> ToolResult:
    strictness_instruction = {
        "strict": "Answer ONLY from the document context. If not found, say so.",
        "balanced": "Prefer document context. You may use general knowledge to connect ideas.",
        "creative": "Use document context and your knowledge freely to give a complete answer.",
    }.get(strictness, "Prefer document context.")

    prompt = [
        {
            "role": "system",
            "content": (
                f"You are a document research assistant.\n{strictness_instruction}\n"
                f"Memory context:\n{memory_context or 'None'}"
            ),
        },
        {"role": "user", "content": (f"Context:\n{context}\n\nQuestion:\n{question}")},
    ]

    try:
        response = generate_response(messages=prompt, temperature=0)
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Answer generation failed: {e}"

    summary = f"Answer generated for: '{question[:80]}'"
    truncated = summary + " | " + answer[:300]

    return ToolResult(
        name="answer_generator",
        success=bool(answer),
        summary=summary,
        top_chunks=[],
        full_context=answer,
        data={"question": question},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 7 — answer_synthesizer
# =====================================================


def tool_answer_synthesizer(
    original_question: str,
    sub_answers: list[str],
    strictness: StrictnessLevel = "balanced",
) -> ToolResult:
    combined = "\n\n".join(f"Sub-answer {i+1}:\n{a}" for i, a in enumerate(sub_answers))

    strictness_instruction = {
        "strict": "Synthesize only from the sub-answers provided. Do not add external knowledge.",
        "balanced": "Synthesize from sub-answers. You may connect ideas using general knowledge.",
        "creative": "Synthesize freely, enriching with your knowledge where helpful.",
    }.get(strictness, "Synthesize from sub-answers.")

    prompt = [
        {
            "role": "system",
            "content": (f"You are a research synthesizer.\n{strictness_instruction}"),
        },
        {
            "role": "user",
            "content": (
                f"Original question:\n{original_question}\n\n"
                f"Sub-answers:\n{combined}\n\n"
                "Synthesize a single coherent final answer."
            ),
        },
    ]

    try:
        response = generate_response(messages=prompt, temperature=0)
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = sub_answers[0] if sub_answers else f"Synthesis failed: {e}"

    summary = f"Synthesized {len(sub_answers)} sub-answers"
    truncated = summary + " | " + answer[:300]

    return ToolResult(
        name="answer_synthesizer",
        success=bool(answer),
        summary=summary,
        top_chunks=[],
        full_context=answer,
        data={"sub_answer_count": len(sub_answers)},
        truncated_result=truncated,
    )


# =====================================================
# TOOL 8 — answer_evaluator
# =====================================================


def tool_answer_evaluator(
    question: str,
    context: str,
    answer: str,
) -> ToolResult:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are evaluating a RAG answer for groundedness and quality. "
                "Return ONLY valid JSON, no markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Context (truncated):\n{context[:4000]}\n\n"
                f"Answer:\n{answer}\n\n"
                "Return JSON:\n"
                "{\n"
                '  "supported": true | false,\n'
                '  "reason": "one sentence",\n'
                '  "retry": true | false,\n'
                '  "suggested_query": "gap-targeting query if retry=true, else null",\n'
                '  "confidence": 0-10\n'
                "}\n"
            ),
        },
    ]

    try:
        response = generate_response(messages=prompt, temperature=0)
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        evaluation = json.loads(raw)
    except Exception:
        evaluation = {
            "supported": True,
            "reason": "Evaluation parse failed — accepting answer",
            "retry": False,
            "suggested_query": None,
            "confidence": 6,
        }

    summary = f"Confidence {evaluation.get('confidence', '?')}/10 — {'supported' if evaluation.get('supported') else 'unsupported'}"
    truncated = summary + " | " + evaluation.get("reason", "")[:200]

    return ToolResult(
        name="answer_evaluator",
        success=True,
        summary=summary,
        top_chunks=[],
        full_context="",
        data=evaluation,
        truncated_result=truncated,
    )


# =====================================================
# TOOL 9 — memory_search
# =====================================================


def tool_memory_search(
    session_id: str,
    question: str,
) -> ToolResult:
    context = retrieve_relevant_memories(session_id, question)
    summary = f"Memory retrieved — {'has context' if context.strip() else 'empty'}"
    truncated = summary + " | " + context[:300]

    return ToolResult(
        name="memory_search",
        success=bool(context.strip()),
        summary=summary,
        top_chunks=[],
        full_context=context,
        data={"has_memory": bool(context.strip())},
        truncated_result=truncated,
    )


# =====================================================
# PARALLEL EXECUTOR
# =====================================================


def run_tools_parallel(
    tasks: list[tuple[callable, tuple, dict]],
    max_workers: int = 8,
) -> list[ToolResult]:
    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=min(len(tasks), max_workers)) as executor:
        futures = {
            executor.submit(fn, *args, **kwargs): idx
            for idx, (fn, args, kwargs) in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                fn_name = tasks[idx][0].__name__
                results[idx] = ToolResult(
                    name=fn_name,
                    success=False,
                    summary=f"{fn_name} failed: {e}",
                    top_chunks=[],
                    full_context="",
                    data={},
                    truncated_result=f"{fn_name} failed: {e}"[:500],
                )

    return results
