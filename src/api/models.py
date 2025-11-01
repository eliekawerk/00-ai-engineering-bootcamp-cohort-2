from typing import List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the rag pipeline")
    thread_id: str = Field(..., description="Pesistent memory")


class RAGUsedContextResponse(BaseModel):
    image_url: str = Field(..., description="The image URL of the item")
    price: Optional[float] = Field(default=None, description="price of the item")
    description: str = Field(..., description="Description of the item")


class AgentResponse(BaseModel):
    request_id: str = Field(..., description="Request id")
    answer: str = Field(..., description="The answer to the query")
    used_context: List[RAGUsedContextResponse] = Field(
        ..., description="Information about items used to answer the query"
    )
