import numpy as np
from qdrant_client import QdrantClient


from pydantic import BaseModel
from typing import Annotated, List, Any, Dict
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

from src.api.agent.agents import ToolCall, RAGUsedContext
from src.api.agent.utils.utils import get_tool_descriptions
from src.api.agent.tools import get_formatted_context
from src.api.agent.agents import agent_node, intent_router_node
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.api.agent.retrieval_generation import COLLECTION


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []


def tool_router(state: State) -> str:
    """Decide whether to continue or to end"""
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def intent_rounter_conditional_edges(state: State) -> str:
    return "agent_node" if state.question_relevant else "end"


workflow = StateGraph(State)

tools = [get_formatted_context]
tool_node = ToolNode(tools)
tools_descriptions = get_tool_descriptions(tools)

workflow.add_node("agent_node", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")
workflow.add_conditional_edges(
    "intent_router_node",
    intent_rounter_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END,
    },
)
workflow.add_conditional_edges(
    "agent_node", tool_router, {"tools": "tool_node", "end": END}
)
workflow.add_edge("tool_node", "agent_node")


def run_agent(question: str, thread_id: str) -> str:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iteration": 0,
        "available_tools": tools_descriptions,
    }
    config = {"configurable": {"thread_id": thread_id}}
    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        return graph.invoke(initial_state, config=config)


def run_agent_wrapper(question: str, thread_id: str):
    qdrant_client_ = QdrantClient(url="http://qdrant:6333")
    result = run_agent(question, thread_id)
    used_context = []
    dummy_vector = np.zeros(1536).tolist()
    for item in result.get("references", []):
        payload = (
            qdrant_client_.query_points(
                collection_name=COLLECTION,
                query=dummy_vector,
                using="text-embedding-3-small",
                limit=1,
                with_payload=True,
                # with_vectors=False,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin", match=MatchValue(value=item.id)
                        )
                    ]
                ),
            )
            .points[0]
            .payload
        )
        image_url = payload.get("image")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.description,
                }
            )
    return {"answer": result.get("answer"), "used_context": used_context}
