# ===============================
# Retriever Function (single-argument)
# ===============================
import os
import weaviate
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core import StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.indices.vector_store import VectorStoreIndex
from dotenv import load_dotenv
import atexit

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
# Retriever Function
# ===============================
def retrieve_nodes(question, similarity_threshold=0.5, top_k=6, verbose=False):
    """
    Retrieve nodes from Weaviate + LlamaIndex using a question.

    Args:
        question (str): The query string.
        similarity_threshold (float): Minimum similarity score to keep nodes.
        top_k (int): Number of top nodes to retrieve.
        verbose (bool): Print debug info.

    Returns:
        List[NodeWithScore]: Filtered nodes with similarity scores above threshold.
    """
    base_retriever = index.as_retriever(similarity_top_k=top_k)
    retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=verbose)

    base_nodes = base_retriever.retrieve(question)
    filtered_nodes = [node for node in base_nodes if node.score > similarity_threshold]

    return filtered_nodes

