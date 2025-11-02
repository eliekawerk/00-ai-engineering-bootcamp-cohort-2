from fastmcp import FastMCP
from typing import List


from src.reviews_mcp_server.utils import retrieve_reviews_data, process_reviews_context

mcp = FastMCP("reviews_mcp_server")


@mcp.tool()
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


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
