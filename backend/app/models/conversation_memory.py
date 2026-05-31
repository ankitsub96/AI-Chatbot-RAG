# app/models/conversation_memory.py

from datetime import datetime

from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from sqlmodel import SQLModel, Field


class ConversationMemory(SQLModel, table=True):
    __tablename__ = "conversation_memories"

    id: int | None = Field(default=None, primary_key=True)

    session_id: str = Field(index=True)

    question: str
    answer: str
    text: str

    embedding: list[float] = Field(sa_column=Column(Vector(768)))

    created_at: datetime = Field(default_factory=datetime.utcnow)
