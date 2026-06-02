import uuid
from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    document_id: str = Field(foreign_key="documents.id", index=True)

    page: int | None = None
    section: str | None = None
    chunk_type: str | None = None

    # hierarchy
    parent_id: str | None = Field(default=None, index=True)
    child_id: str | None = Field(default=None, index=True)
    chunk_index: int | None = (
        None  # position within parent, useful for sibling retrieval
    )

    text: str

    chunk_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )

    embedding: list[float] = Field(sa_column=Column(Vector(768)))

    tsv: str | None = Field(
        default=None,
        sa_column=Column(TSVECTOR),
    )
