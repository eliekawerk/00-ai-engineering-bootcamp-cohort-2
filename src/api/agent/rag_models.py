from pydantic import BaseModel, Field


class RAGGenerationResponse(BaseModel):
    answer: str = Field("Answer to the question")


class RAGUsedContext(BaseModel):
    id: str = Field(description="Product id of items used to answer the question")
    description: str = Field(
        description="Short description of the item used to answer the question"
    )


class RAGGenerationResponseWithReferences(BaseModel):
    answer: str = Field("Answer to the question")
    references: list[RAGUsedContext] = Field(
        ..., description="list of items used to answer the question."
    )
