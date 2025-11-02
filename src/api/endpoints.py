import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.models import (
    AgentRequest,
    AgentResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.api.agent.graph import run_agent_streaming_wrapper
from src.api.processors.submit_feedback import submit_feedback

logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/")
def rag(request: Request, payload: AgentRequest) -> AgentResponse:
    return StreamingResponse(
        run_agent_streaming_wrapper(payload.query, payload.thread_id),
        media_type="text/event-stream",
    )
    # for context in answer["used_context"]:
    #     print(context)
    # try:
    #     return AgentResponse(
    #         request_id=request.state.request_id,
    #         answer=answer["answer"],
    #         used_context=[
    #             RAGUsedContextResponse(**used_context)
    #             for used_context in answer["used_context"]
    #         ],
    #     )
    # except ValidationError as exc:
    #     print(repr(exc.errors()[0]["type"]))


@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    submit_feedback(
        payload.trace_id,
        payload.feedback_score,
        payload.feedback_text,
        payload.feedback_source_type,
    )
    return FeedbackResponse(request_id=request.state.request_id, status="success")


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["feedback"])
