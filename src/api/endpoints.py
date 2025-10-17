import logging

from fastapi import APIRouter, Request

from src.api.models import RAGRequest, RAGResponse
from src.api.rag.retrieval_generation import rag_pipeline

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: RAGRequest) -> RAGResponse:
    answer = rag_pipeline(payload.query)
    return RAGResponse(request_id=request.state.request_id, answer=answer["answer"])


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
