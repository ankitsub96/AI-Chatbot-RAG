from typing import List, Optional
from pydantic import BaseModel, Field


class AskDocumentRequest(BaseModel):
    session_id: str
    question: str
    document_ids: list[str] | None = (
        None  # optional subset — None means all session docs
    )
