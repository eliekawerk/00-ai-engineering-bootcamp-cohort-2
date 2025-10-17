from typing import Optional

import instructor
import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree

from src.api.rag.rag_models import RAGGenerationResponse


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=text, model=model)
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, qdrant_client_, k=5):
    query_embedding = get_embeddings(query)
    results = qdrant_client_.query_points(
        collection_name="Amazon-items-collection-00",
        query=query_embedding,
        limit=k,
    )
    retrieved_context_ids = []
    retrieved_contexts = []
    similarity_scores = []
    retrieved_context_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_contexts.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "context_ids": retrieved_context_ids,
        "context": retrieved_contexts,
        "similarity_scores": similarity_scores,
        "context_ratings": retrieved_context_ratings,
    }


@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context: dict):
    formatted_context = ""
    for id, chunk in zip(context["context_ids"], context["context"]):
        formatted_context += f"- {id}: {chunk}\n"

    return formatted_context


@traceable(name="build_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    prompt = f"""
    You are a shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - You need to answer the question based on the provided context only.
    - Never use word context and refer to it as the available products.

    Context:
    {preprocessed_context}

    Question:
    {question}
    """

    return prompt


@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def generate_answer(prompt):
    client = instructor.from_openai(openai.OpenAI())
    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.5,
        response_model=RAGGenerationResponse,
    )
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            # "output_tokens": response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response


@traceable(name="rag-pipeline")
def rag_pipeline(
    question: str, qdrant_client_: Optional[QdrantClient] = None, topk: int = 5
) -> dict:
    qdrant_client_ = qdrant_client_ or QdrantClient(url="http://qdrant:6333")
    retrieved_context = retrieve_data(question, qdrant_client_, topk)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)
    final_output = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["context_ids"],
        "retrieved_context": retrieved_context["context"],
        "similarity_score": retrieved_context["similarity_scores"],
        "retrieved_context_ratings": retrieved_context["context_ratings"],
    }
    return final_output
