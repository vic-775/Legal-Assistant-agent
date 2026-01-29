# ===============================
# Retriever Function (with metadata filters)
# ===============================
import os
import sys
import weaviate
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from dotenv import load_dotenv
import atexit

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Logging
import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# ===============================
# Connect to Weaviate
# ===============================
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# Close client safely at exit
atexit.register(lambda: client.close())

# ===============================
# Vector store & storage context
# ===============================
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="InternationalLawDocument",
    text_key="content",
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

# ===============================
# Retriever Function (clean version)
# ===============================
def query_documents(query: str, metadata_filters: dict = None, top_k: int = 50):
    """
    Query the Weaviate vector store.

    Args:
        query (str): The text query.
        metadata_filters (dict): Optional filter dict, e.g. {"document_type": "article"}
        top_k (int): Number of top similar results to retrieve.

    Returns:
        List[dict]: Each dict contains text, similarity_score, and metadata.
    """
    # Convert metadata_filters to LlamaIndex MetadataFilters format
    filters = None
    if metadata_filters:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key=k, value=v) for k, v in metadata_filters.items()]
        )
    
    # Create retriever
    base_retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=False)
    
    # Retrieve nodes
    nodes = retriever.retrieve(query)

    # Build simple results
    results = []
    for node in nodes:
        results.append({
            "text": node.get_text(),
            "similarity_score": getattr(node, "score", None),
            "metadata": node.metadata
        })

    return results
