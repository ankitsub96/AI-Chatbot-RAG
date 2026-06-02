import uuid
from datetime import datetime
from sqlmodel import SQLModel, Field


class Session(SQLModel, table=True):
    __tablename__ = "sessions"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    title: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
