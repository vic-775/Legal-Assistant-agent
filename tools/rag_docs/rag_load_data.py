# ===============================
# Imports
# ===============================
import os
import weaviate
from dotenv import load_dotenv

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding

# ===============================
# Load environment variables
# ===============================
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===============================
# Load PDF
# ===============================
loader = PyMuPDFReader()
data_path = r"C:\Users\USER\Desktop\Projects\rag_llamaindex\tools\rag_docs\legal_docs\international law handook.pdf"

docs_per_page = loader.load(file_path=data_path)
print(f"Pages loaded: {len(docs_per_page)}")

# Combine pages into one document
full_text = "\n\n".join([d.get_content() for d in docs_per_page])
documents = [Document(text=full_text)]

# ===============================
# Create hierarchical nodes
# ===============================
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)

leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

print(f"Total nodes: {len(nodes)}")
print(f"Leaf nodes (indexed): {len(leaf_nodes)}")
print(f"Root nodes (context): {len(root_nodes)}")

# ===============================
# Connect to Weaviate Cloud
# ===============================
client = weaviate.connect_to_wcs(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# ===============================
# Create Weaviate vector store
# ===============================
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="Document",  # class name in Weaviate
)

# ===============================
# Create storage context
# ===============================
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)  # store full hierarchy

storage_context = StorageContext.from_defaults(
    docstore=docstore,
    vector_store=vector_store,
)

# ===============================
# Create OpenAI embedding model
embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# ===============================
# Index leaf nodes into Weaviate
# ===============================
index = VectorStoreIndex.from_documents(
    leaf_nodes,
    storage_context=storage_context,
    embedding=embedding_model,
)

print("Leaf nodes successfully embedded and stored in Weaviate")
print(f"Vectors stored: {len(leaf_nodes)}")
