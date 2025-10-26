import logging

from fastapi import APIRouter, Request
from pydantic import ValidationError

from src.api.models import AgentRequest, AgentResponse, RAGUsedContextResponse
from src.api.agent.graph import run_agent_wrapper

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: AgentRequest) -> AgentResponse:
    answer = run_agent_wrapper(payload.query)
    for context in answer["used_context"]:
        print(context)
    try:
        return AgentResponse(
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
