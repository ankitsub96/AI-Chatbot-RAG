from pydantic import BaseModel
from pydantic import Field


class SearchMemoryRequest(BaseModel):

    query: str
