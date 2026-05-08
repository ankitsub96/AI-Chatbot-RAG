from pydantic import BaseModel
from pydantic import Field


class AskDocumentRequest(BaseModel):

    filename: str = Field(..., example="resume.pdf")

    question: str = Field(..., example="What backend technologies does he know?")

    session_id: str = Field(..., example="user-123")
