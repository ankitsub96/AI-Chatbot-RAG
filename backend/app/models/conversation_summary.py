# app/models/conversation_summary.py

from datetime import datetime

from sqlalchemy import Column
from pgvector.sqlalchemy import Vector
from sqlmodel import SQLModel, Field


class ConversationSummary(SQLModel, table=True):
    __tablename__ = "conversation_summaries"

    id: int | None = Field(default=None, primary_key=True)

    session_id: str = Field(index=True)

    summary: str

    embedding: list[float] = Field(sa_column=Column(Vector(768)))

    created_at: datetime = Field(default_factory=datetime.utcnow)
