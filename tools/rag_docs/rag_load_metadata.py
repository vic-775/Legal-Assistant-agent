# ===============================
# Imports
# ===============================
import os
import re
import weaviate
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple

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
from llama_index.core.schema import TextNode, NodeRelationship
from llama_index.embeddings.openai import OpenAIEmbedding

# ===============================
# Metadata Extraction Functions
# ===============================

def extract_metadata_from_text(text: str, page_num: int) -> Dict:
    """Extract metadata from text based on document structure."""
    metadata = {
        "book": None,
        "chapter": None,
        "chapter_number": None,
        "section": None,
        "instrument": None,
        "page_start": page_num,
        "page_end": page_num,
        "article_start": None,
        "article_end": None,
    }
    
    # Extract Book information (e.g., "Book One", "BOOK ONE")
    book_match = re.search(r'BOOK (ONE|TWO|THREE|FOUR)', text[:500], re.IGNORECASE)
    if book_match:
        metadata["book"] = f"Book {book_match.group(1).title()}"
    
    # Extract Chapter (e.g., "CHAPTER I. CHARTER OF THE UNITED NATIONS")
    chapter_match = re.search(r'CHAPTER ([IVXLCDM]+)\.\s*([^\n]+)', text[:1000])
    if chapter_match:
        roman_numeral = chapter_match.group(1)
        chapter_title = chapter_match.group(2).strip()
        metadata["chapter"] = f"CHAPTER {roman_numeral}. {chapter_title}"
        
        # Convert roman numeral to number
        metadata["chapter_number"] = roman_to_int(roman_numeral)
    
    # Extract section headers (e.g., "CHAPTER I: PURPOSES AND PRINCIPLES")
    section_match = re.search(r'CHAPTER [IVXLCDM]+:\s*([^\n]+)', text[:2000])
    if section_match:
        metadata["section"] = section_match.group(1).strip()
    
    # Extract instrument names (e.g., "Charter of the United Nations")
    instrument_patterns = [
        r'Charter of the United Nations',
        r'Vienna Convention on the Law of Treaties',
        r'Convention on rights and duties of States',
        r'Universal Declaration of Human Rights',
        r'International Covenant on',
        r'Convention relating to the status of refugees',
    ]
    
    for pattern in instrument_patterns:
        if re.search(pattern, text[:2000], re.IGNORECASE):
            metadata["instrument"] = pattern
            break
    
    # Extract article ranges for legal documents
    article_matches = re.findall(r'Article (\d+)', text[:3000])
    if article_matches:
        try:
            article_nums = [int(a) for a in article_matches if a.isdigit()]
            metadata["article_start"] = min(article_nums)
            metadata["article_end"] = max(article_nums)
        except:
            pass
    
    return metadata

def roman_to_int(roman: str) -> int:
    """Convert Roman numeral to integer."""
    roman_dict = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    result = 0
    prev_value = 0
    
    for char in reversed(roman.upper()):
        value = roman_dict.get(char, 0)
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    
    return result

ARTICLE_PATTERN = re.compile(r"Article\s+(\d+)", re.IGNORECASE)
chapter_article_counter = {}

def add_metadata_to_nodes(nodes: List[TextNode]) -> List[TextNode]:
    chapter_article_counter = {}

    for node in nodes:
        page_num = node.metadata.get("page_num", 0)
        metadata = extract_metadata_from_text(node.text, page_num)

        # Merge metadata
        node.metadata = {**node.metadata, **metadata}

        # ---- ARTICLE NUMBER PRESERVATION ----
        chapter_num = node.metadata.get("chapter_number")

        if chapter_num:
            # initialize counter per chapter
            if chapter_num not in chapter_article_counter:
                chapter_article_counter[chapter_num] = 0

            article_match = ARTICLE_PATTERN.search(node.text[:300])

            if article_match:
                chapter_article_counter[chapter_num] += 1
                node.metadata["article_number"] = chapter_article_counter[chapter_num]
            else:
                # inherit article number from parent if exists
                parent = node.relationships.get(NodeRelationship.PARENT)
                if parent and "article_number" in parent.metadata:
                    node.metadata["article_number"] = parent.metadata["article_number"]

        # ---- DOCUMENT TYPE ----
        if "article_number" in node.metadata:
            node.metadata["document_type"] = "article"
        elif node.metadata.get("section"):
            node.metadata["document_type"] = "sub_chapter"
        elif node.metadata.get("chapter"):
            node.metadata["document_type"] = "chapter"
        else:
            node.metadata["document_type"] = "text"

    return nodes

def get_or_create_weaviate_class(client):
    """Simpler version for Weaviate v4+."""
    class_name = "InternationalLawDocument"
    
    try:
        # Check if class exists
        existing_classes = client.collections.list_all()
        if class_name in existing_classes:
            print(f"Class '{class_name}' already exists — deleting it first...")
            try:
                client.collections.delete(class_name)
                print(f"Deleted existing class '{class_name}' successfully.")
            except Exception as e_del:
                print(f"Error deleting class '{class_name}': {e_del}")
    except Exception as e_check:
        print(f"Error checking existing classes: {e_check}")

    # Create new class
    try:
        # Try using the old schema API
        client.schema.create_class({
            "class": class_name,
            "description": "International law documents with hierarchical metadata",
            "vectorizer": "none",
            "properties": [
                {"name": "book", "dataType": ["text"]},
                {"name": "chapter", "dataType": ["text"]},
                {"name": "chapter_number", "dataType": ["int"]},
                {"name": "section", "dataType": ["text"]},
                {"name": "instrument", "dataType": ["text"]},
                {"name": "page_start", "dataType": ["int"]},
                {"name": "page_end", "dataType": ["int"]},
                {"name": "article_start", "dataType": ["int"]},
                {"name": "article_end", "dataType": ["int"]},
                {"name": "document_type", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]}
            ]
        })
        print(f"Created new class: {class_name}")
        return class_name
    except Exception as e:
        print(f"Error creating class with old API: {e}")
        
        # Try creating with direct HTTP request as last resort
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {WEAVIATE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            schema_data = {
                "class": class_name,
                "vectorizer": "none",
                "properties": [
                    {"name": "book", "dataType": ["text"]},
                    {"name": "chapter", "dataType": ["text"]},
                    {"name": "chapter_number", "dataType": ["int"]},
                    {"name": "section", "dataType": ["text"]},
                    {"name": "instrument", "dataType": ["text"]},
                    {"name": "page_start", "dataType": ["int"]},
                    {"name": "page_end", "dataType": ["int"]},
                    {"name": "article_start", "dataType": ["int"]},
                    {"name": "article_end", "dataType": ["int"]},
                    {"name": "document_type", "dataType": ["text"]},
                    {"name": "content", "dataType": ["text"]}
                ]
            }
            
            response = requests.post(
                f"{WEAVIATE_URL}/v1/schema",
                headers=headers,
                json=schema_data
            )
            
            if response.status_code in [200, 201]:
                print(f"Created new class via HTTP: {class_name}")
                return class_name
            else:
                print(f"Failed to create class: {response.text}")
                # Return class name anyway - might already exist
                return class_name
                
        except Exception as e2:
            print(f"Error creating class via HTTP: {e2}")
            # Return class name anyway - might already exist
            return class_name

# ===============================
# Weaviate Cleanup Functions
# ===============================
def delete_all_objects(client, class_name: str):
    """
    Delete all objects in a Weaviate class using the current v5+ client.
    """
    try:
        with client.batch as batch:
            batch.delete_objects(class_name=class_name)
        print(f"All objects deleted in class '{class_name}'")
    except Exception as e:
        print(f"Error deleting objects: {e}")

# ===============================
# Load environment variables
# ===============================
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===============================
# Load PDF with metadata
# ===============================
loader = PyMuPDFReader()
data_path = r"C:\Users\USER\Desktop\Projects\rag_llamaindex\tools\rag_docs\legal_docs\international law handook.pdf"

print("Loading PDF...")
docs_per_page = loader.load(file_path=data_path)
print(f"Pages loaded: {len(docs_per_page)}")

# Add page numbers to documents
for i, doc in enumerate(docs_per_page):
    doc.metadata["page_num"] = i + 1

# Combine into documents with metadata
documents = []
for doc in docs_per_page:
    documents.append(Document(
        text=doc.get_content(),
        metadata=doc.metadata
    ))

# ===============================
# Create hierarchical nodes with metadata
# ===============================
print("\nCreating hierarchical nodes...")
node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(documents)

# Add metadata to nodes
print("Extracting metadata from nodes...")
nodes = add_metadata_to_nodes(nodes)  # FIXED: Removed extra (nodes)

leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

# ===============================
# Smart chunking function
# ===============================
def smart_chunk_text(node: TextNode, max_chars: int = 10000, chunk_size: int = 8000, overlap: int = 1000) -> List[TextNode]:
    """
    Chunk article nodes intelligently:
      - <= max_chars: keep as-is
      - > max_chars: split into overlapping chunks
    Each chunk preserves metadata and prepends article number and heading.
    """
    text = node.text
    metadata = node.metadata.copy()

    # Prepend article number and heading if present
    prefix = ""
    if "article_number" in metadata:
        prefix += f"Article {metadata['article_number']}"
        if metadata.get("section"):
            prefix += f": {metadata['section']}"
        prefix += "\n\n"
    
    full_text = prefix + text
    if len(full_text) <= max_chars:
        node.text = full_text
        return [node]
    
    # Split into chunks with overlap
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk_text = full_text[start:end]
        chunk_node = TextNode(
            text=chunk_text,
            metadata=metadata.copy(),
            relationships=node.relationships.copy()
        )
        chunks.append(chunk_node)
        start += chunk_size - overlap  # move with overlap
    
    return chunks

# ===============================
# Apply smart chunking to leaf nodes
# ===============================
print("\nApplying smart chunking to leaf nodes...")
chunked_leaf_nodes = []
for node in leaf_nodes:
    if node.metadata.get("document_type") == "article":
        chunked_leaf_nodes.extend(smart_chunk_text(node))
    else:
        chunked_leaf_nodes.append(node)  # keep other nodes as-is

print(f"Total nodes after smart chunking: {len(chunked_leaf_nodes)}")

print(f"\nNode Statistics:")
print(f"Total nodes: {len(nodes)}")
print(f"Leaf nodes (indexed): {len(leaf_nodes)}")
print(f"Root nodes (context): {len(root_nodes)}")

# ===============================
# Connect to Weaviate Cloud
# ===============================
print("\nConnecting to Weaviate...")
client = weaviate.connect_to_weaviate_cloud(  
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# Get or create Weaviate class
class_name = get_or_create_weaviate_class(client)

# ===============================
# Create Weaviate vector store
# ===============================
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name=class_name,
    text_key="content",  # Field to store text content
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
# ===============================
embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# ===============================
# Index leaf nodes into Weaviate with metadata
# ===============================
print("Indexing documents with metadata into Weaviate...")

# Get a reference to the class collection
collection = client.collections.get(class_name)

for node in leaf_nodes:
    obj_props = {
        "content": node.text,
        "book": node.metadata.get("book"),
        "chapter": node.metadata.get("chapter"),
        "chapter_number": node.metadata.get("chapter_number"),
        "section": node.metadata.get("section"),
        "instrument": node.metadata.get("instrument"),
        "page_start": node.metadata.get("page_start"),
        "page_end": node.metadata.get("page_end"),
        "article_start": node.metadata.get("article_start"),
        "article_end": node.metadata.get("article_end"),
        "document_type": node.metadata.get("document_type"),
    }
    # Insert the object into Weaviate
    collection.data.insert(properties=obj_props)

print(f"Indexed {len(leaf_nodes)} nodes into Weaviate.")

# ===============================
# Verification
# ===============================
print("\nVerifying indexed data...")
# Count objects in Weaviate
try:
    # Simple count query
    response = client.query.aggregate(class_name).with_meta_count().do()
    count = response['data']['Aggregate'][class_name][0]['meta']['count']
    print(f"Total vectors in Weaviate: {count}")
    
    # Sample metadata from first few objects
    print("\nSample metadata from indexed documents:")
    response = client.query.get(
        class_name=class_name,
        properties=["book", "chapter", "instrument"],
        limit=3
    ).do()
    
    if 'data' in response and 'Get' in response['data']:
        objects = response['data']['Get'][class_name]
        for i, obj in enumerate(objects):
            print(f"\nDocument {i+1}:")
            print(f"  Book: {obj.get('book', 'N/A')}")
            print(f"  Chapter: {str(obj.get('chapter', 'N/A'))[:50]}...")
            print(f"  Instrument: {obj.get('instrument', 'N/A')}")
        
except Exception as e:
    print(f"Error verifying data: {e}")

print(f"\nSuccessfully indexed {len(leaf_nodes)} documents with metadata")
print(f"Weaviate class: {class_name}")
print(f"Metadata fields available for filtering: book, chapter, chapter_number, section, instrument, article_start, article_end, document_type")

# Close Weaviate connection
client.close()