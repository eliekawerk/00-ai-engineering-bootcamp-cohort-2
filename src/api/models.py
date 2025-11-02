from typing import List, Optional, Union

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


class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(
        ..., description="1 if the feedback is positive, 0 if the feedback is negative"
    )
    feedback_text: str = Field(..., description="The feedback text")
    trace_id: str = Field(..., description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")
    feedback_source_type: str = Field(
        ..., description="The type of feedback. Human or API"
    )


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")
