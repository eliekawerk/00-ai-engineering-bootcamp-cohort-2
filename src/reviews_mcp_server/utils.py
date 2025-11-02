import openai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery,
    Prefetch,
    Filter,
    FieldCondition,
    MatchAny,
)

COLLECTION_REVIEWS = "Amazon-items-collection-01-reviews"


def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


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


def process_reviews_context(context):
    formatted_context = ""
    for id, context in zip(
        context["retrieved_context_ids"],
        context["retrieved_context"],
    ):
        formatted_context += f"- ID: {id}, Review: {context}"
    return formatted_context
