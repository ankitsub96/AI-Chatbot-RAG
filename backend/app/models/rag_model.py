from typing import List
from pydantic import BaseModel, Field


class AskDocumentRequest(BaseModel):

    filenames: List[str] = Field(
        ...,
        example=["resume.pdf", "experience_letter.pdf"],
        description="List of indexed documents to search",
    )

    question: str = Field(
        ...,
        example="What backend technologies does he know?",
    )

    session_id: str = Field(
        ...,
        example="user-123",
    )
