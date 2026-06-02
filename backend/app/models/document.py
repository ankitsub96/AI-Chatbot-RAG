import uuid
from datetime import datetime
from sqlmodel import SQLModel, Field


class Document(SQLModel, table=True):
    __tablename__ = "documents"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    checksum: str = Field(unique=True, index=True)

    original_filename: str
    stored_filename: str

    status: str = Field(default="pending")  # pending | ready | failed

    page_count: int | None = None
    chunk_count: int | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
