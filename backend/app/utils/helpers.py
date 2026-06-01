import hashlib
import time
import asyncio
import functools
from collections import defaultdict
import threading
import json

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


def thinking(stage: str, message: str, data: dict = None):
    return format_sse(
        json.dumps(
            {"type": "thinking", "stage": stage, "message": message, "data": data or {}}
        ),
        event="thinking",
    )


def token_event(token: str):
    return format_sse(
        json.dumps({"type": "response", "token": token}), event="response"
    )


def done_event():
    return format_sse(json.dumps({"type": "done"}), event="done")
