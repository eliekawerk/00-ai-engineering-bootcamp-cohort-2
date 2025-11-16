import numpy as np
from qdrant_client import QdrantClient
import json

from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Dict
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver

from src.api.agent.agents import (
    ToolCall,
    RAGUsedContext,
    product_qa_agent,
    shopping_cart_agent,
    coordinator_agent,
    Delegation,
)
from src.api.agent.utils.utils import get_tool_descriptions
from src.api.agent.tools import (
    get_formatted_context,
    get_formatted_reviews_context,
    add_to_shopping_cart,
    get_shopping_cart,
    remove_from_cart,
)
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from src.api.agent.retrieval_generation import COLLECTION


class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    available_tools: List[Dict[str, Any]] = []
    tool_calls: List[ToolCall] = []


class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qa_agent: AgentProperties = Field(default_factory=AgentProperties)
    shopping_cart_agent: AgentProperties = Field(default_factory=AgentProperties)
    coordinator_agent: CoordinatorAgentProperties = Field(
        default_factory=AgentProperties
    )
    answer: str = ""
    references: Annotated[List[RAGUsedContext], add] = []
    user_id: str = ""
    cart_id: str = ""


def product_qa_agent_tool_router(state: "State") -> str:
    """Decide whether to continue or to end"""
    if state.product_qa_agent.final_answer:
        return "end"
    elif state.product_qa_agent.iteration > 4:
        return "end"
    elif len(state.product_qa_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def shopping_cart_agent_tool_router(state: "State") -> str:
    """Decide whether to continue or to end"""
    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 2:
        return "end"
    elif len(state.shopping_cart_agent.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def user_intent_router(state):
    """Decide whether to continue or to end"""
    if state.user_intent == "product_qa":
        return "product_qa_agent"
    elif state.user_intent == "shopping_cart":
        return "shopping_cart_agent"
    else:
        return "end"


def coordinator_router(state):
    """Decide whether to continue or to end"""
    if state.coordinator_agent.iteration > 3:
        return "end"
    elif (
        state.coordinator_agent.final_answer and len(state.coordinator_agent.plan) == 0
    ):
        return "end"
    elif state.coordinator_agent.next_agent == "product_qa_agent":
        return "product_qa_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    else:
        return "end"


workflow = StateGraph(State)

product_qa_tools = [
    get_formatted_context,
    get_formatted_reviews_context,
]
product_qa_tool_node = ToolNode(product_qa_tools)
product_qa_tools_descriptions = get_tool_descriptions(product_qa_tools)

shopping_cart_tools = [add_to_shopping_cart, get_shopping_cart, remove_from_cart]
shopping_cart_tool_node = ToolNode(shopping_cart_tools)
shopping_cart_tool_descriptions = get_tool_descriptions(shopping_cart_tools)

workflow.add_node("product_qa_agent", product_qa_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("coordinator_agent", coordinator_agent)

workflow.add_node("product_qa_agent_tool_node", product_qa_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_tool_node)

workflow.add_edge(START, "coordinator_agent")
workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_router,
    {
        "product_qa_agent": "product_qa_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "end": END,
    },
)
workflow.add_conditional_edges(
    "product_qa_agent",
    product_qa_agent_tool_router,
    {
        "tools": "product_qa_agent_tool_node",
        "end": "coordinator_agent",
    },
)
workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {"tools": "shopping_cart_agent_tool_node", "end": "coordinator_agent"},
)

workflow.add_edge("product_qa_agent_tool_node", "product_qa_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")


def run_agent(question: str, thread_id: str) -> str:
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "user_intent": "",
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_tools_descriptions,
            "tool_calls": [],
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_tool_descriptions,
            "tool_calls": [],
        },
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


def _string_for_sse(message: str) -> str:
    return f"data: {message}\n\n"


def _process_graph_event(chunk):
    def _is_node_start(chunk):
        return chunk[1].get("type") == "task"

    def _is_node_end(chunk):
        return chunk[0] == "updates"

    def _tool_to_text(tool_call):
        if tool_call.name == "get_formatted_context":
            return f"Looking for items: {tool_call.arguments.get('query', '')}."
        elif tool_call.name == "get_formatted_reviews_context":
            return "Fetching user reviews..."
        else:
            return "Unknown"

    if _is_node_start(chunk):
        if chunk[1].get("payload", {}).get("name") == "intent_router_node":
            return "Analysing the question..."
        if chunk[1].get("payload", {}).get("name") == "agent_node":
            return "Planning..."
        if chunk[1].get("payload", {}).get("name") == "tool_node":
            message = " ".join(
                [
                    _tool_to_text(tool_call)
                    for tool_call in chunk[1]
                    .get("payload", {})
                    .get("input", {})
                    .tool_calls
                ]
            )
            return message
    else:
        return False


def run_agent_streaming_wrapper(question: str, thread_id: str):
    qdrant_client_ = QdrantClient(url="http://qdrant:6333")

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "product_qa_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": product_qa_tools_descriptions,
            "tool_calls": [],
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False,
            "available_tools": shopping_cart_tool_descriptions,
            "tool_calls": [],
        },
        "coordinator_agent": {
            "iteration": 0,
            "final_answer": False,
            "plan": [],
            "next_agent": "",
        },
        "user_id": thread_id,
        "cart_id": thread_id,
    }

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
        for chunk in graph.stream(
            initial_state, config=config, stream_mode=["debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk)
            if processed_chunk:
                yield _string_for_sse(processed_chunk)

            if chunk[0] == "values":
                result = chunk[1]

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
    shopping_cart = get_shopping_cart(thread_id, thread_id)
    shopping_cart_items = [
        {
            col: item.get(col)
            for col in [
                "price",
                "quantity",
                "currency",
                "product_image_url",
                "total_price",
            ]
        }
        for item in shopping_cart
    ]
    yield _string_for_sse(
        json.dumps(
            {
                "type": "final_result",
                "data": {
                    "answer": result.get("answer"),
                    "used_context": used_context,
                    "trace_id": result.get("trace_id"),
                    "shopping_cart": shopping_cart_items,
                },
            },
            default=float,
        )
    )
