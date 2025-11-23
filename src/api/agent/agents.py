from typing import List
import instructor


from langchain_core.messages import convert_to_openai_messages
from langchain_core.messages import AIMessage
from langsmith import traceable, get_current_run_tree

from pydantic import BaseModel, Field

from litellm import completion

from src.api.agent.utils.utils import format_ai_message
from src.api.agent.utils.prompt_management import prompt_template_config


class ToolCall(BaseModel):
    name: str
    arguments: dict


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer")


class ProductQAAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(
        description="List of items used to answer the question."
    )
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


class ShoppingCartAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


class WarehouseManagerAgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


class IntentRouterResponse(BaseModel):
    user_intent: str
    answer: str


class Delegation(BaseModel):
    agent: str
    task: str


class CoordinatorAgentResponse(BaseModel):
    next_agent: str
    plan: List[Delegation]
    final_answer: bool
    answer: str


@traceable(
    name="product_qa_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"},
)
def product_qa_agent(state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]) -> dict:
    prompts = dict()
    for model in models:
        template = prompt_template_config("src/api/agent/prompts/qa_agent.yaml", model)
        prompts[model] = template.render(
            available_tools=state.product_qa_agent.available_tools
        )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ProductQAAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0.0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "product_qa_agent": {
            "tool_calls": response.tool_calls,
            "iteration": state.product_qa_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.product_qa_agent.available_tools,
        },
        "answer": response.answer,
        "references": response.references,
    }


@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"},
)
def shopping_cart_agent(
    state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]
) -> dict:
    prompts = dict()
    for model in models:
        template = prompt_template_config(
            "src/api/agent/prompts/shopping_cart_agent.yaml", model
        )
        prompts[model] = template.render(
            available_tools=state.shopping_cart_agent.available_tools,
            user_id=state.user_id,
            cart_id=state.cart_id,
        )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=ShoppingCartAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0.0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "shopping_cart_agent": {
            "tool_calls": response.tool_calls,
            "iteration": state.shopping_cart_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.shopping_cart_agent.available_tools,
        },
        "answer": response.answer,
    }


@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"},
)
def warehouse_manager_agent(
    state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]
) -> dict:
    prompts = dict()
    for model in models:
        template = prompt_template_config(
            "src/api/agent/prompts/warehouse_manager_agent.yaml", model
        )
        prompts[model] = template.render(
            available_tools=state.warehouse_manager_agent.available_tools,
        )

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=WarehouseManagerAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0.0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "warehouse_manager_agent": {
            "tool_calls": response.tool_calls,
            "iteration": state.warehouse_manager_agent.iteration + 1,
            "final_answer": response.final_answer,
            "available_tools": state.warehouse_manager_agent.available_tools,
        },
        "answer": response.answer,
    }


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"},
)
def intent_router_node(
    state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]
) -> dict:
    prompts = dict()
    for model in models:
        template = prompt_template_config(
            "src/api/agent/prompts/intent_router_agent.yaml", model
        )
        prompts[model] = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=IntentRouterResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0.0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    if response.user_intent == "product_qa":
        ai_message = []
    else:
        ai_message = [AIMessage(content=response.answer)]

    return {
        "messages": ai_message,
        "answer": response.answer,
        "user_intent": response.user_intent,
        "trace_id": trace_id,
    }


@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1"},
)
def coordinator_agent(
    state, models=["gpt-4.1", "groq/llama-3.3-70b-versatile"]
) -> dict:
    prompts = dict()
    for model in models:
        template = prompt_template_config(
            "src/api/agent/prompts/coordinator_agent.yaml", model
        )
        prompts[model] = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)

    for model in models:
        try:
            response, raw_response = client.chat.completions.create_with_completion(
                model=model,
                response_model=CoordinatorAgentResponse,
                messages=[{"role": "system", "content": prompts[model]}, *conversation],
                temperature=0.0,
            )
            break
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
        trace_id = str(getattr(current_run, "trace_id", current_run.id))

    if response.final_answer:
        ai_message = []
    else:
        ai_message = [AIMessage(content=response.answer)]

    return {
        "messages": ai_message,
        "answer": response.answer,
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration + 1,
            "final_answer": response.final_answer,
            "next_agent": response.next_agent,
            "plan": [data.model_dump() for data in response.plan],
        },
        "trace_id": trace_id,
    }
