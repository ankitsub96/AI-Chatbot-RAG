import hashlib
import time
import functools
from collections import defaultdict
import threading

_lock = threading.Lock()
_active_counts = defaultdict(int)


def create_cache_key(text: str):

    return hashlib.md5(text.encode()).hexdigest()


def timer(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

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

    return wrapper
