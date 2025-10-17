from pydantic import BaseModel, Field


class RAGGenerationResponse(BaseModel):
    answer: str = Field("Answer to the question")
