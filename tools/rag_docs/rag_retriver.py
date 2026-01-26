# ===============================
# Retriever Function (load vector store and exract relevant content)
# ===============================
import os
import sys
import weaviate
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# logging
import logging
logger = logging.getLogger(__name__)

# langfuse tracing
from observability.langfuse_tracing import langfuse

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate and set up the retriever once
client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="Document",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

base_retriever = index.as_retriever(similarity_top_k=50)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=False)

# ===============================
# Function to query with logging and langfuse tracing
# ===============================
def query_documents(query: str):
    """
    Query the existing vector store and return relevant nodes.

    Args:
        query (str): The natural language question/query.

    Returns:
        list: Nodes containing full text, node ID, and similarity score.
    """
    logger.info("Starting document retrieval")
    logger.debug(f"Query: {query}")

    if trace is None:
        # If no trace is passed, start a temporary one
        trace = langfuse.trace(name="Vector Retrieval (no session)", input={"query": query})

    # Wrap retrieval in a span
    with trace.span(name="Vector Retrieval", input={"query": query}) as span:
        try:
            nodes = retriever.retrieve(query)
            logger.info(f"Retrieved {len(nodes)} nodes")

            results = []
            for node in nodes:
                results.append({
                    "node_id": node.node_id,
                    "similarity": getattr(node, "score", None),
                    "text": node.get_text(),
                })

            # End the span with structured output
            span.end(output={
                "num_nodes": len(results),
                "documents": [
                    {"node_id": r["node_id"], "similarity": r["similarity"]}
                    for r in results
                ]
            })

            return results

        except Exception as e:
            logger.exception("Retriever failed")
            span.end(output={"error": str(e)})
            raise
