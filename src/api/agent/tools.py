from typing import List

from langsmith import traceable, get_current_run_tree
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    FusionQuery,
    Prefetch,
    Filter,
    FieldCondition,
    MatchAny,
)


@traceable(name="format_retrieved_context", run_type="prompt")
def process_context(context):
    formatted_context = ""

    for id, chunk, rating in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
        context["retrieved_context_ratings"],
    ):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

    return formatted_context


@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(name="retrieve_data", run_type="retriever")
def retrieve_data(query, k=5):
    query_embedding = get_embedding(query)

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid-search",
        prefetch=[
            Prefetch(query=query_embedding, using="text-embedding-3-small", limit=20),
            Prefetch(
                query=Document(text=query, model="qdrant/bm25"), using="bm25", limit=20
            ),
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )

    retrieved_context_ids = []
    retrieved_context = []
    retrieved_context_ratings = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["description"])
        retrieved_context_ratings.append(result.payload["average_rating"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "retrieved_context_ratings": retrieved_context_ratings,
        "similarity_scores": similarity_scores,
    }


def get_formatted_context(query: str, top_k: int = 5) -> str:
    """Get the top k context, each representing an inventory item for a given query.

    Args:
        query: The query to get the top k context for
        top_k: The number of context chunks to retrieve, works best with 5 or more

    Returns:
        A string of the top k context chunks with IDs and average ratings prepending each chunk, each representing an inventory item for a given query.
    """

    context = retrieve_data(query, top_k)
    formatted_context = process_context(context)

    return formatted_context


### Reviews Retrieval Tool
COLLECTION_REVIEWS = "Amazon-items-collection-01-reviews"


@traceable(
    name="retrive_reviews_data",
    run_type="retriever",
)
def retrieve_reviews_data(query: str, parent_asins: list[str], k=5):
    query_embedding = get_embedding(query)
    qdrant_client = QdrantClient("http://qdrant:6333")
    results = qdrant_client.query_points(
        collection_name=COLLECTION_REVIEWS,
        prefetch=[
            Prefetch(
                query=query_embedding,
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin", match=MatchAny(any=parent_asins)
                        )
                    ]
                ),
                limit=20,
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=k,
    )
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload["text"])
        similarity_scores.append(result.score)
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
    }


@traceable(
    name="format_retrieved_reviews_context",
    run_type="prompt",
)
def process_reviews_context(context):
    formatted_context = ""
    for id, context in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
    ):
        formatted_context += f"- ID: {id}, Review: {context}"
    return formatted_context


def get_formatted_reviews_context(
    query: str, items_list: List[str], topk: int = 5
) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query. This should be a python list containing the actual parent_sin (as strings) of retrieved items
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """
    context = retrieve_reviews_data(query, items_list, topk)
    formatted_context = process_reviews_context(context)

    return formatted_context
