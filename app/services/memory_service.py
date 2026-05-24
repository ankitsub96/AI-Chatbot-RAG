import os
import json
import asyncio

from app.services.llm_service import generate_response
from app.services.vector_search_service import (
    create_embedding,
    create_embeddings,
    load_index_and_metadata,
    semantic_search,
    save_faiss_index,
    create_flat_index,
    load_faiss_index,
)


from app.utils.file_utils import save_json_file, load_json_file, write_text_file

# =========================
# CONFIG
# =========================

MEMORY_DIR = "app/memory_store"

MEMORY_TOP_K = 5

MAX_HISTORY = 50

SUMMARY_TRIGGER = 12

RECENT_HISTORY = 10

os.makedirs(MEMORY_DIR, exist_ok=True)

# =========================
# PATHS
# =========================


def get_memory_index_path(session_id: str):

    return os.path.join(MEMORY_DIR, f"{session_id}.index")


def get_memory_metadata_path(session_id: str):

    return os.path.join(MEMORY_DIR, f"{session_id}.json")


def get_summary_path(session_id: str):

    return os.path.join(MEMORY_DIR, f"{session_id}_summary.txt")


# =========================
# SUMMARY
# =========================


def save_summary(session_id: str, summary: str):

    path = get_summary_path(session_id)

    with open(path, "w", encoding="utf-8") as f:

        f.write(summary)


def load_summary(session_id: str):

    path = get_summary_path(session_id)

    if not os.path.exists(path):

        return ""

    with open(path, "r", encoding="utf-8") as f:

        return f.read()


# =========================
# SUMMARIZATION
# =========================


def summarize_conversation(history):

    if not history:

        return ""

    text = ""

    for item in history:

        text += f"""

USER:
{item['question']}

ASSISTANT:
{item['answer']}
"""

    prompt = f"""
Summarize this conversation memory.

Preserve:
- important facts
- technical discussions
- ongoing tasks
- user preferences
- decisions
- context continuity

Conversation:
{text}
"""

    response = generate_response(
        messages=[{"role": "user", "content": prompt}], temperature=0
    )

    return response.choices[0].message.content


# =========================
# SAVE MEMORY
# =========================


async def save_conversation_turn(session_id: str, question: str, answer: str):

    print("\n" + "=" * 80)
    print("SAVING CONVERSATION MEMORY")
    print("=" * 80)

    index_path = get_memory_index_path(session_id)

    metadata_path = get_memory_metadata_path(session_id)

    txt_path = os.path.join(MEMORY_DIR, f"{session_id}.txt")

    memory_text = f"""
USER:
{question}

ASSISTANT:
{answer}
"""

    # =========================
    # LOAD MEMORY
    # =========================

    memories = load_json_file(metadata_path, default=[])

    memories.append({"question": question, "answer": answer, "text": memory_text})

    # =========================
    # SUMMARIZATION
    # =========================

    if len(memories) > SUMMARY_TRIGGER:

        old_memories = memories[:-RECENT_HISTORY]

        previous_summary = await asyncio.to_thread(load_summary, session_id)

        new_summary = await asyncio.to_thread(summarize_conversation, old_memories)

        combined_summary = f"""
PREVIOUS SUMMARY:
{previous_summary}

NEW SUMMARY:
{new_summary}
"""

        final_summary = await asyncio.to_thread(
            summarize_conversation,
            [{"question": "Conversation Summary", "answer": combined_summary}],
        )

        await asyncio.to_thread(save_summary, session_id, final_summary)

        memories = memories[-RECENT_HISTORY:]

    # =========================
    # HARD LIMIT
    # =========================

    memories = memories[-MAX_HISTORY:]

    texts = [memory["text"] for memory in memories]

    # =========================
    # PARALLEL TASKS
    # =========================

    embeddings_task = asyncio.to_thread(create_embeddings, texts)

    txt_content_task = asyncio.to_thread(build_memory_text_file, session_id, memories)

    embeddings, txt_content = await asyncio.gather(embeddings_task, txt_content_task)

    # =========================
    # BUILD INDEX
    # =========================

    index = create_flat_index(embeddings)

    # =========================
    # SAVE ALL IN PARALLEL
    # =========================

    await asyncio.gather(
        asyncio.to_thread(save_faiss_index, index, index_path),
        asyncio.to_thread(save_json_file, metadata_path, memories),
        asyncio.to_thread(write_text_file, txt_path, txt_content),
    )

    print("\nMEMORY SAVE COMPLETE")


def build_memory_text_file(session_id: str, memories: list):

    combined_text = ""

    summary_path = get_summary_path(session_id)

    if os.path.exists(summary_path):

        summary = load_summary(session_id)

        combined_text += f"""
================================================================================
SESSION SUMMARY
================================================================================

{summary}

"""

    combined_text += """
================================================================================
RECENT CONVERSATIONS
================================================================================
"""

    for index_num, memory in enumerate(memories, start=1):

        combined_text += f"""

[{index_num}]

USER:
{memory['question']}

ASSISTANT:
{memory['answer']}

--------------------------------------------------------------------------------
"""

    return combined_text


# =========================
# MEMORY RETRIEVAL
# =========================


def retrieve_relevant_memories(session_id: str, question: str):

    index_path = get_memory_index_path(session_id)

    metadata_path = get_memory_metadata_path(session_id)

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):

        return ""

    index, memories = load_index_and_metadata(index_path, metadata_path)

    query_embedding = create_embedding(question)

    results = semantic_search(
        index=index,
        metadata=memories,
        query_embedding=query_embedding,
        top_k=MEMORY_TOP_K,
    )

    context = ""

    print("\nMEMORY RETRIEVAL")

    for result in results:

        memory = result["data"]

        print({"score": result["score"], "memory": memory["text"]})

        context += f"""

{memory['text']}
"""

    return context


# =========================
# SESSION APIs
# =========================


def get_all_sessions():

    sessions = []

    for file in os.listdir(MEMORY_DIR):

        if file.endswith(".json") and "_summary" not in file:

            sessions.append(file.replace(".json", ""))

    return sessions


def get_session_history(session_id: str, page: int = 1, page_size: int = 20):

    path = get_memory_metadata_path(session_id)

    if not os.path.exists(path):

        return {"total": 0, "items": []}

    with open(path, "r", encoding="utf-8") as f:

        history = json.load(f)

    total = len(history)

    start = (page - 1) * page_size

    end = start + page_size

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": history[start:end],
    }


def delete_session_memory(session_id: str):

    files = [
        get_memory_index_path(session_id),
        get_memory_metadata_path(session_id),
        get_summary_path(session_id),
    ]

    for path in files:

        if os.path.exists(path):

            os.remove(path)


def semantic_search_session(session_id: str, query: str, top_k: int = 5):

    index_path = get_memory_index_path(session_id)

    metadata_path = get_memory_metadata_path(session_id)

    if not os.path.exists(index_path):

        return []

    index, history = load_index_and_metadata(index_path, metadata_path)

    query_embedding = create_embedding(query)

    results = semantic_search(
        index=index,
        metadata=history,
        query_embedding=query_embedding,
        top_k=top_k,
    )

    return [result["data"] for result in results]
