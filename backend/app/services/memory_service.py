import os
import json
import asyncio
from sqlmodel import Session, select
from sqlalchemy import text
from concurrent.futures import ThreadPoolExecutor

from app.services.llm_service import generate_response
from app.services.vector_search_service import (
    create_embedding,
    create_embeddings,
    # load_index_and_metadata,
    # semantic_search,
    # save_faiss_index,
    # create_flat_index,
    # load_faiss_index,
)

from app.services.database import engine
from app.models.conversation_memory import ConversationMemory
from app.models.conversation_summary import (
    ConversationSummary,
)
from app.models.session import Session as SessionModel

from app.utils.file_utils import save_json_file, load_json_file, write_text_file

# =========================
# CONFIG
# =========================

MEMORY_DIR = "app/memory_store"

MEMORY_TOP_K = 5
SUMMARY_TOP_K = 5
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


async def save_conversation_turn(
    session_id: str,
    question: str,
    answer: str,
    thoughts: list[dict] | None = None,
):
    print("\n" + "=" * 80)
    print("SAVING CONVERSATION MEMORY")
    print("=" * 80)

    memory_text = f"""
USER:
{question}

ASSISTANT:
{answer}
"""

    # =========================
    # SAVE CURRENT MEMORY
    # =========================

    embedding = await asyncio.to_thread(
        create_embedding,
        memory_text,
    )

    with Session(engine) as session:

        session.add(
            ConversationMemory(
                session_id=session_id,
                question=question,
                answer=answer,
                text=memory_text,
                embedding=embedding[0].tolist(),
                thoughts=thoughts,
            )
        )

        session.commit()

    # =========================
    # LOAD SESSION MEMORIES
    # =========================

    with Session(engine) as session:

        memories = session.exec(
            select(ConversationMemory)
            .where(ConversationMemory.session_id == session_id)
            .order_by(ConversationMemory.created_at)
        ).all()

    # =========================
    # SUMMARIZATION
    # =========================

    if len(memories) > SUMMARY_TRIGGER:

        old_memories = memories[:-RECENT_HISTORY]

        summary_input = []

        for memory in old_memories:

            summary_input.append(
                {
                    "question": memory.question,
                    "answer": memory.answer,
                }
            )

        new_summary = await asyncio.to_thread(
            summarize_conversation,
            summary_input,
        )

        summary_embedding = await asyncio.to_thread(
            create_embedding,
            new_summary,
        )

        with Session(engine) as session:

            session.add(
                ConversationSummary(
                    session_id=session_id,
                    summary=new_summary,
                    embedding=summary_embedding[0].tolist(),
                )
            )

            session.commit()

        # =========================
        # DELETE SUMMARIZED MEMORIES
        # KEEP RECENT ONLY
        # =========================

        # old_ids = [memory.id for memory in old_memories]

        # with Session(engine) as session:

        #     rows = session.exec(
        #         select(ConversationMemory).where(ConversationMemory.id.in_(old_ids))
        #     ).all()

        #     for row in rows:
        #         session.delete(row)

        #     session.commit()

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


def retrieve_relevant_memories(
    session_id: str,
    question: str,
):
    print("\nMEMORY RETRIEVAL")

    query_embedding = create_embedding(question)
    embedding_str = str(query_embedding[0].tolist())

    context = ""

    def get_summaries():
        with Session(engine) as session:
            return list(
                session.execute(
                    text("""
                        SELECT
                            summary,
                            embedding <=> CAST(:embedding AS vector) AS distance
                        FROM conversation_summaries
                        WHERE session_id = :session_id
                        ORDER BY embedding <=> CAST(:embedding AS vector)
                        LIMIT :limit
                    """),
                    {
                        "session_id": session_id,
                        "embedding": embedding_str,
                        "limit": SUMMARY_TOP_K,
                    },
                ).mappings()
            )

    def get_memories():
        with Session(engine) as session:
            return list(
                session.execute(
                    text("""
                        SELECT
                            text,
                            embedding <=> CAST(:embedding AS vector) AS distance
                        FROM conversation_memories
                        WHERE session_id = :session_id
                        ORDER BY embedding <=> CAST(:embedding AS vector)
                        LIMIT :limit
                    """),
                    {
                        "session_id": session_id,
                        "embedding": embedding_str,
                        "limit": MEMORY_TOP_K,
                    },
                ).mappings()
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = executor.submit(get_summaries)
        memory_future = executor.submit(get_memories)

        summary_rows = summary_future.result()
        memory_rows = memory_future.result()

    for row in summary_rows:
        print(
            {
                "type": "summary",
                "distance": row["distance"],
            }
        )

        context += f"""

LONG TERM MEMORY

{row["summary"]}

"""

    for row in memory_rows:
        print(
            {
                "type": "memory",
                "distance": row["distance"],
            }
        )

        context += f"""

RECENT MEMORY

{row["text"]}

"""

    return context


# =========================
# SESSION APIs
# =========================


def get_all_sessions():

    with Session(engine) as session:

        sessions = session.exec(select(SessionModel)).all()

    return sessions


def get_session_history(
    session_id: str,
    page: int = 1,
    page_size: int = 20,
):
    with Session(engine) as session:

        rows = session.exec(
            select(ConversationMemory)
            .where(ConversationMemory.session_id == session_id)
            .order_by(ConversationMemory.created_at.asc())
        ).all()

    total = len(rows)

    start = (page - 1) * page_size
    end = start + page_size
    items = []

    for row in rows[start:end]:
        items.append(
            {
                "id": row.id,
                "question": row.question,
                "answer": row.answer,
                "created_at": row.created_at,
            }
        )
    return {"total": total, "page": page, "page_size": page_size, "items": items}


def delete_session_memory(
    session_id: str,
):
    with Session(engine) as session:

        memories = session.exec(
            select(ConversationMemory).where(
                ConversationMemory.session_id == session_id
            )
        ).all()

        for row in memories:
            session.delete(row)

        summaries = session.exec(
            select(ConversationSummary).where(
                ConversationSummary.session_id == session_id
            )
        ).all()

        for row in summaries:
            session.delete(row)

        session.commit()


def semantic_search_session(
    session_id: str,
    query: str,
    top_k: int = 5,
):
    query_embedding = create_embedding(query)

    embedding_str = str(query_embedding[0].tolist())

    with Session(engine) as session:

        rows = session.execute(
            text("""
                SELECT
                    id,
                    session_id,
                    question,
                    answer,
                    text,
                    created_at,
                    embedding <=> CAST(:embedding AS vector) AS distance
                FROM conversation_memories
                WHERE session_id = :session_id
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
                """),
            {
                "session_id": session_id,
                "embedding": embedding_str,
                "top_k": top_k,
            },
        ).mappings()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "question": row["question"],
                "answer": row["answer"],
                "text": row["text"],
                "created_at": row["created_at"],
                "distance": row["distance"],
            }
            for row in rows
        ]
