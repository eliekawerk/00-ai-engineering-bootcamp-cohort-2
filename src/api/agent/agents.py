from typing import List
import instructor

from openai import OpenAI

from langchain_core.messages import convert_to_openai_messages
from langsmith import traceable, get_current_run_tree

from pydantic import BaseModel, Field

from src.api.agent.utils.utils import format_ai_message
from src.api.agent.utils.prompt_management import prompt_template_config


class ToolCall(BaseModel):
    name: str
    arguments: dict


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer")


class AgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    references: list[RAGUsedContext] = Field(
        description="List of items used to answer the question."
    )
    final_answer: bool = False
    tool_calls: List[ToolCall] = []


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"},
)
def agent_node(state) -> dict:
    template = prompt_template_config("src/api/agent/prompts/qa_agent.yaml", "qa_agent")
    prompt = template.render(available_tools=state.available_tools)

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=AgentResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    ai_message = format_ai_message(response)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "references": response.references,
    }


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-mini"},
)
def intent_router_node(state):
    template = prompt_template_config(
        "src/api/agent/prompts/intent_router_agent.yaml", "intent_router_agent"
    )

    prompt = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_openai(OpenAI())

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        response_model=IntentRouterResponse,
        messages=[{"role": "system", "content": prompt}, *conversation],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    return {"question_relevant": response.question_relevant, "answer": response.answer}
