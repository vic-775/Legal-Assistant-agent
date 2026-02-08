#### Import Dependencies
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.vector_stores import MetadataFilter
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_child_nodes,
    get_root_nodes,
    get_leaf_nodes
)
from llama_index.llms.openai import OpenAI
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from IPython.display import Markdown, display
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import AutoMergingRetriever

import re
from tqdm import tqdm
import os
import openai
import weaviate

# environment variable for OpenAI API key
import os
from dotenv import load_dotenv
load_dotenv()

#### Data Loading
pdf_path = r"C:\Users\USER\Desktop\Projects\rag_llamaindex\tools\rag_docs\legal_docs\international law handook.pdf"
docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()

#### Chunk the data
chunks = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2000, 1000, 500],  
    chunk_overlap=50
)

### metafilters
qa_extractor = QuestionsAnsweredExtractor(questions=3)

#### Set Up LLM for embeddings
llm = OpenAI(model="gpt-4o") 
embed_model = OpenAIEmbedding(model="text-embedding-3-small") 

transformations = [
    chunks,
    qa_extractor
]

pipeline = IngestionPipeline(
    transformations=transformations
)
nodes = pipeline.run(documents=docs)
print(f"Total nodes created: {len(nodes)}")

first_node = nodes[0]
first_node.extra_info

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
)

class_name = "InternationalLawDocument"

if client.collections.exists(class_name):
    client.collections.delete(class_name)
    print("Deleted existing collection")

vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name=class_name,
    text_key="content",
)

docstore = SimpleDocumentStore()
# insert nodes into docstore
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore,)

# Create OpenAI embedding model
embedding_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")

base_index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embedding=embedding_model,
)