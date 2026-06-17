import hashlib
import time
import asyncio
import functools
from collections import defaultdict
import threading
import json

from app.models.rag_model import MemoryMessage

_lock = threading.Lock()
_active_counts = defaultdict(int)


def create_cache_key(text: str):

    return hashlib.md5(text.encode()).hexdigest()


def timer(func):

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):

        with _lock:
            _active_counts[func.__name__] += 1
            instance = _active_counts[func.__name__]

        print(f"[{func.__name__}] instance {instance} started")

        start = time.perf_counter()

        try:
            result = await func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[{func.__name__}] instance {instance} finished in {elapsed:.3f}s")

            with _lock:
                _active_counts[func.__name__] -= 1

        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):

        with _lock:
            _active_counts[func.__name__] += 1
            instance = _active_counts[func.__name__]

        print(f"[{func.__name__}] instance {instance} started")

        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"[{func.__name__}] instance {instance} finished in {elapsed:.3f}s")

            with _lock:
                _active_counts[func.__name__] -= 1

        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper

    return sync_wrapper


def format_sse(data: str, event: str = "message") -> str:
    return f"event: {event}\ndata: {data}\n\n"


def thinking(stage: str, message: str, data: dict = None)-> str:
    return format_sse(
        json.dumps(
            {"type": "thinking", "stage": stage, "message": message, "data": data or {}}
        ),
        event="thinking",
    )


def token_event(token: str)-> str:
    return format_sse(
        json.dumps({"type": "response", "token": token}), event="response"
    )


def done_event()-> str:
    return format_sse(json.dumps({"type": "done"}), event="done")


def _thought(node: str, message: str, data: dict | None = None) -> dict:
    return {"node": node, "message": message, "data": data, "ts": time.time()}


def _parse_memory_to_messages(memory_context: str) -> list[MemoryMessage]:
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
# SHARED RAG HELPERS
# =====================================================

def rag_thought(node: str, message: str, data: dict | None = None) -> dict:
    return {"node": node, "message": message, "data": data, "ts": time.time()}


def rag_cache_key(session_id: str, document_ids: list[str]) -> str:
    return session_id + "|" + "|".join(sorted(document_ids))