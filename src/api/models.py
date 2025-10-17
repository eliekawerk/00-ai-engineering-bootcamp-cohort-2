from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the rag pipeline")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="Request id")
    answer: str = Field(..., description="The answer to the query")
