from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from app.models.chunk_types import StrictnessLevel


class AskDocumentRequest(BaseModel):
    session_id: str
    question: str
    document_ids: list[str] | None = (
        None  # optional subset — None means all session docs
    )


class AgentAskRequest(BaseModel):
    session_id: str
    question: str
    document_ids: list[str] | None = None
    stream: bool = True

class ResearchAskRequest(BaseModel):
    session_id: str
    question: str
    document_ids: list[str] | None = None
    stream: bool = True
    use_web: bool = False
    strictness: StrictnessLevel = "balanced"


class MemoryMessage(TypedDict):
    type: Literal["human", "ai"]
    content: str