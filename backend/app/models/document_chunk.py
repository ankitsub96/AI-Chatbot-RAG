from sqlalchemy import Column, JSON
from sqlmodel import SQLModel, Field
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR


class DocumentChunk(SQLModel, table=True):
    __tablename__ = "document_chunks"

    id: int | None = Field(default=None, primary_key=True)

    filename: str
    page: int | None = None
    section: str | None = None
    chunk_type: str | None = None

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
