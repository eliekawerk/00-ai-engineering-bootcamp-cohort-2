import logging

from fastapi import APIRouter, Request

from src.api.models import RAGRequest, RAGResponse, RAGUsedContextResponse
from src.api.rag.retrieval_generation import rag_pipeline_wrapper

from pydantic import ValidationError

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:
    answer = rag_pipeline_wrapper(payload.query)
    for context in answer["used_context"]:
        print(context)
    try:
        return RAGResponse(
            request_id=request.state.request_id,
            answer=answer["answer"],
            used_context=[
                RAGUsedContextResponse(**used_context)
                for used_context in answer["used_context"]
            ],
        )
    except ValidationError as exc:
        print(repr(exc.errors()[0]["type"]))


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
