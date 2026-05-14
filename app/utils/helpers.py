import hashlib


def create_cache_key(text: str):

    return hashlib.md5(
        text.encode()
    ).hexdigest()