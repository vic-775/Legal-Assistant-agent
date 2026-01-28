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
import atexit

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# logging
import logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Connect to Weaviate and set up the retriever once
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# Close client safely at exit
atexit.register(lambda: client.close())

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
# Function to query with logging 
# ===============================
def query_documents(query: str, trace=None):
    logger.info("Starting document retrieval")
    logger.debug(f"Query: {query}")

    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        results.append({
            "node_id": node.node_id,
            "similarity": getattr(node, "score", None),
            "text": node.get_text(),
        })

    return results