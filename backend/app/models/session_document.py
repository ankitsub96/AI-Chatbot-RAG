import uuid
from datetime import datetime
from sqlmodel import SQLModel, Field
from sqlalchemy import UniqueConstraint


class SessionDocument(SQLModel, table=True):
    __tablename__ = "session_documents"

    __table_args__ = (
        UniqueConstraint("session_id", "document_id", name="uq_session_document"),
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
    )

    session_id: str = Field(foreign_key="sessions.id", index=True)
    document_id: str = Field(foreign_key="documents.id", index=True)

    created_at: datetime = Field(default_factory=datetime.utcnow)
