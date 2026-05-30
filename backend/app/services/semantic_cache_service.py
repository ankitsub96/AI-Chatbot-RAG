import json
import hashlib
import numpy as np

from app.services.cache_service import (
    redis_client
)

SEMANTIC_CACHE_THRESHOLD = 0.90

CACHE_TTL = 3600


def normalize_question(question: str):

    return (
        question
        .lower()
        .strip()
    )


def get_exact_cache_key(
    filename: str,
    question: str
):

    return hashlib.md5(
        f"{filename}::{question}".encode()
    ).hexdigest()


def get_semantic_cache_key(
    filename: str,
    question: str
):

    question_hash = hashlib.md5(
        question.encode()
    ).hexdigest()

    return (
        f"semantic_cache::{filename}::"
        f"{question_hash}"
    )


def get_exact_cache(
    filename: str,
    question: str
):

    normalized_question = normalize_question(
        question
    )

    cache_key = get_exact_cache_key(
        filename,
        normalized_question
    )

    return redis_client.get(cache_key)


def set_exact_cache(
    filename: str,
    question: str,
    answer: str
):

    normalized_question = normalize_question(
        question
    )

    cache_key = get_exact_cache_key(
        filename,
        normalized_question
    )

    redis_client.setex(
        cache_key,
        CACHE_TTL,
        answer
    )


def get_semantic_cache(
    filename: str,
    query_embedding
):

    semantic_cache_keys = redis_client.keys(
        f"semantic_cache::{filename}::*"
    )

    best_similarity = -1

    best_answer = None

    for key in semantic_cache_keys:

        cached_item = redis_client.get(key)

        if not cached_item:
            continue

        cached_item = json.loads(
            cached_item
        )

        cached_embedding = np.array(
            cached_item["embedding"],
            dtype="float32"
        ).reshape(1, -1)

        similarity = np.dot(
            query_embedding,
            cached_embedding.T
        )[0][0]

        if similarity > best_similarity:

            best_similarity = similarity

            best_answer = cached_item["answer"]

    print(f"Semantic similarity: {best_similarity}")

    if best_similarity >= SEMANTIC_CACHE_THRESHOLD:

        return best_answer

    return None


def set_semantic_cache(
    filename: str,
    question: str,
    embedding,
    answer: str
):

    normalized_question = normalize_question(
        question
    )

    cache_key = get_semantic_cache_key(
        filename,
        normalized_question
    )

    payload = {
        "question": normalized_question,
        "embedding": embedding[0].tolist(),
        "answer": answer
    }

    redis_client.setex(
        cache_key,
        CACHE_TTL,
        json.dumps(payload)
    )