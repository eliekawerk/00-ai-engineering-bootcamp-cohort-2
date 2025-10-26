import openai
from langsmith import get_current_run_tree, traceable
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    FusionQuery,
    Prefetch,
)


# from src.api.agent.rag_models import RAGGenerationResponseWithReferences
from src.api.agent.utils.prompt_management import prompt_template_config


COLLECTION = "Amazon-items-collection-01-hybrid-search"


# @traceable(
#     name="embed_query",
#     run_type="embedding",
#     metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
# )
# def get_embeddings(text, model="text-embedding-3-small"):
#     response = openai.embeddings.create(input=text, model=model)
#     current_run = get_current_run_tree()
#     if current_run:
#         current_run.metadata["usage_metadata"] = {
#             "input_tokens": response.usage.prompt_tokens,
#             "total_tokens": response.usage.total_tokens,
#         }
#     return response.data[0].embedding


# @traceable(name="retrieve_data", run_type="retriever")
# def retrieve_data(query, qdrant_client, k=5):
#     query_embedding = get_embeddings(query)
#     results = qdrant_client.query_points(
#         collection_name=COLLECTION,
#         prefetch=[
#             Prefetch(query=query_embedding, limit=20, using="text-embedding-3-small"),
#             Prefetch(
#                 query=Document(text=query, model="qdrant/bm25"),
#                 using="bm25",
#                 limit=20,
#             ),
#         ],
#         query=FusionQuery(fusion="rrf"),
#         limit=k,
#     )
#     retrieved_context_ids = []
#     retrieved_context = []
#     retrieved_context_ratings = []
#     similarity_scores = []

#     for result in results.points:
#         retrieved_context_ids.append(result.payload["parent_asin"])
#         retrieved_context.append(result.payload["description"])
#         retrieved_context_ratings.append(result.payload["average_rating"])
#         similarity_scores.append(result.score)

#     return {
#         "context_ids": retrieved_context_ids,
#         "context": retrieved_context,
#         "context_ratings": retrieved_context_ratings,
#         "similarity_scores": similarity_scores,
#     }


# @traceable(name="format_retrieved_context", run_type="prompt")
# def process_context(context: dict):
#     formatted_context = ""
#     for id, chunk, rating in zip(
#         context["context_ids"], context["context"], context["context_ratings"]
#     ):
#         formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"

#     return formatted_context


@traceable(name="build_prompt", run_type="prompt")
def build_prompt(preprocessed_context, question):
    template = prompt_template_config(
        "src/api/rag/prompts/retrieval_generation.yaml", "retrieval_generation"
    )
    rendered_prompt = template.render(
        preprocessed_context=preprocessed_context, question=question
    )
    return rendered_prompt


# @traceable(
#     name="generate_answer",
#     run_type="llm",
#     metadata={"ls_provider": "openai", "ls_model_name": "text-embedding-3-small"},
# )
# def generate_answer(prompt):
#     client = instructor.from_openai(openai.OpenAI())
#     response, raw_response = client.chat.completions.create_with_completion(
#         model="gpt-4.1-mini",
#         messages=[{"role": "system", "content": prompt}],
#         temperature=0.5,
#         response_model=RAGGenerationResponseWithReferences,
#     )
#     current_run = get_current_run_tree()
#     if current_run:
#         current_run.metadata["usage_metadata"] = {
#             "input_tokens": raw_response.usage.prompt_tokens,
#             "output_tokens": raw_response.usage.completion_tokens,
#             "total_tokens": raw_response.usage.total_tokens,
#         }
#     return response


# @traceable(name="rag-pipeline")
# def rag_pipeline(
#     question: str, qdrant_client_: Optional[QdrantClient] = None, topK: int = 5
# ) -> dict:
#     retrieved_context = retrieve_data(question, qdrant_client_, topK)
#     preprocessed_context = process_context(retrieved_context)
#     prompt = build_prompt(preprocessed_context, question)
#     answer = generate_answer(prompt)
#     final_output = {
#         "answer": answer.answer,
#         "references": answer.references,
#         "question": question,
#         "retrieved_context_ids": retrieved_context["context_ids"],
#         "retrieved_context": retrieved_context["context"],
#         "similarity_score": retrieved_context["similarity_scores"],
#         "retrieved_context_ratings": retrieved_context["context_ratings"],
#     }
#     return final_output


# def rag_pipeline_wrapper(question: str, topK: int = 5):
#     qdrant_client_ = QdrantClient(url="http://qdrant:6333")
#     result = rag_pipeline(question, qdrant_client_, topK)
#     used_context = []
#     dummy_vector = np.zeros(1536).tolist()
#     for item in result.get("references", []):
#         payload = (
#             qdrant_client_.query_points(
#                 collection_name=COLLECTION,
#                 query=dummy_vector,
#                 using="text-embedding-3-small",
#                 limit=1,
#                 with_payload=True,
#                 # with_vectors=False,
#                 query_filter=Filter(
#                     must=[
#                         FieldCondition(
#                             key="parent_asin", match=MatchValue(value=item.id)
#                         )
#                     ]
#                 ),
#             )
#             .points[0]
#             .payload
#         )
#         image_url = payload.get("image")
#         price = payload.get("price")
#         if image_url:
#             used_context.append(
#                 {
#                     "image_url": image_url,
#                     "price": price,
#                     "description": item.description,
#                 }
#             )
#     return {"answer": result["answer"], "used_context": used_context}


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
