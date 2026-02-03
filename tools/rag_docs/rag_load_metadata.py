# ===============================
# Imports
# ===============================
import os
import re
import weaviate
from dotenv import load_dotenv
from typing import List, Dict, Optional

from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document, StorageContext
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
import tiktoken

# ===============================
# Token Counting Helper
# ===============================

def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def smart_chunk_by_tokens(text: str, max_tokens: int = 7000, overlap_tokens: int = 200) -> List[str]:
    """
    Chunk text by token count instead of character count.
    Keeps chunks under max_tokens with overlap for context.
    """
    try:
        encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if end >= len(tokens):
            break
        
        start = end - overlap_tokens
    
    return chunks

# ===============================
# Enhanced Metadata Extraction
# ===============================

def extract_main_chapter(text: str) -> Optional[str]:
    """Extract main chapter (e.g., 'BOOK ONE', 'BOOK TWO')"""
    match = re.search(r'\bBOOK\s+(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|[IVXLCDM]+)\b', text, re.IGNORECASE)
    if match:
        return f"BOOK {match.group(1).upper()}"
    return None

def extract_subchapter(text: str) -> Optional[Dict]:
    """Extract subchapter (CHAPTER or PART with numbers)"""
    # Try CHAPTER first
    chapter_match = re.search(r'\b(CHAPTER)\s+([IVXLCDM]+|[0-9]+)[\s\.\:]+([^\n]+)', text, re.IGNORECASE)
    if chapter_match:
        return {
            "type": "CHAPTER",
            "number": chapter_match.group(2),
            "title": chapter_match.group(3).strip()
        }
    
    # Try PART
    part_match = re.search(r'\b(PART)\s+([IVXLCDM]+|[0-9]+)[\s\.\:]+([^\n]+)', text, re.IGNORECASE)
    if part_match:
        return {
            "type": "PART",
            "number": part_match.group(2),
            "title": part_match.group(3).strip()
        }
    
    return None

def extract_article_number(text: str) -> Optional[int]:
    """Extract article number from beginning of text"""
    match = re.search(r'^\s*Article\s+(\d+)', text, re.IGNORECASE | re.MULTILINE)
    if match:
        return int(match.group(1))
    return None

def roman_to_int(s: str) -> int:
    """Convert Roman numeral to integer"""
    if s.isdigit():
        return int(s)
    
    roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    result = 0
    prev_value = 0
    
    for char in reversed(s.upper()):
        value = roman_dict.get(char, 0)
        if value < prev_value:
            result -= value
        else:
            result += value
        prev_value = value
    
    return result

# ===============================
# Custom Document Parser
# ===============================

def parse_documents_with_structure(documents: List[Document]) -> List[TextNode]:
    """
    Parse documents preserving hierarchical structure:
    - Track main chapters (BOOK X)
    - Track subchapters (CHAPTER/PART X)
    - Identify articles and chunk if needed (by TOKENS, not chars)
    """
    nodes = []
    
    # State tracking across pages
    current_main_chapter = None
    current_subchapter = None
    current_subchapter_number = None
    article_counter = 0
    
    # Combine all text to process sequentially
    full_text = ""
    page_map = []
    
    for doc in documents:
        text = doc.get_content()
        start_pos = len(full_text)
        full_text += text + "\n\n"
        end_pos = len(full_text)
        page_map.append({
            "start": start_pos,
            "end": end_pos,
            "page_num": doc.metadata.get("page_num", 0),
            "text": text
        })
    
    # Split by articles
    article_pattern = re.compile(r'(^|\n)\s*Article\s+\d+', re.IGNORECASE | re.MULTILINE)
    article_splits = article_pattern.split(full_text)
    
    current_pos = 0
    
    for i, segment in enumerate(article_splits):
        if not segment.strip():
            continue
        
        # Find which page this segment belongs to
        segment_start = full_text.find(segment, current_pos)
        segment_end = segment_start + len(segment)
        current_pos = segment_end
        
        page_num = None
        for page_info in page_map:
            if page_info["start"] <= segment_start < page_info["end"]:
                page_num = page_info["page_num"]
                break
        
        # Check for main chapter change
        main_chapter = extract_main_chapter(segment)
        if main_chapter:
            current_main_chapter = main_chapter
            article_counter = 0
        
        # Check for subchapter change
        subchapter = extract_subchapter(segment)
        if subchapter:
            current_subchapter = f"{subchapter['type']} {subchapter['number']}: {subchapter['title']}"
            try:
                current_subchapter_number = roman_to_int(subchapter['number'])
            except:
                current_subchapter_number = None
        
        # Check if this is an article
        article_num = extract_article_number(segment)
        if article_num:
            article_counter += 1
            is_article = True
        else:
            is_article = False
        
        # Prepare metadata
        metadata = {
            "page_num": page_num,
            "main_chapter": current_main_chapter,
            "subchapter": current_subchapter,
            "subchapter_number": current_subchapter_number,
            "article_number": article_counter if is_article else None,
            "document_type": "article" if is_article else "section"
        }
        
        # ===== TOKEN-AWARE CHUNKING =====
        # Check token count instead of character count
        token_count = count_tokens(segment)
        
        # If too large (>7000 tokens or >10000 chars as fallback), chunk it
        if token_count > 7000 or len(segment) > 10000:
            print(f"  Chunking segment (Article {article_counter if is_article else 'N/A'}): {token_count} tokens")
            
            # Prepend article info for context
            article_prefix = ""
            if is_article:
                article_prefix = f"Article {article_counter}"
                if current_subchapter:
                    article_prefix += f" ({current_subchapter})"
                article_prefix += "\n\n"
            
            # Smart chunk by tokens
            text_chunks = smart_chunk_by_tokens(segment, max_tokens=7000, overlap_tokens=200)
            
            for chunk_idx, chunk_text in enumerate(text_chunks):
                if chunk_idx == 0:
                    final_text = article_prefix + chunk_text
                else:
                    final_text = f"{article_prefix}(continued)\n\n{chunk_text}"
                
                node = TextNode(
                    text=final_text,
                    metadata=metadata.copy()
                )
                nodes.append(node)
        else:
            # Keep as single node
            node = TextNode(
                text=segment,
                metadata=metadata
            )
            nodes.append(node)
    
    return nodes

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

print("Loading PDF...")
docs_per_page = loader.load(file_path=data_path)
print(f"Pages loaded: {len(docs_per_page)}")

# Add page numbers
for i, doc in enumerate(docs_per_page):
    doc.metadata["page_num"] = i + 1

# ===============================
# Parse with structure
# ===============================
print("\nParsing document structure...")
nodes = parse_documents_with_structure(docs_per_page)

# Filter out empty nodes
nodes = [n for n in nodes if n.text.strip()]

print(f"\nCreated {len(nodes)} nodes")
print(f"Articles: {sum(1 for n in nodes if n.metadata.get('document_type') == 'article')}")
print(f"Sections: {sum(1 for n in nodes if n.metadata.get('document_type') == 'section')}")

# Check token counts
print("\nToken count statistics:")
token_counts = [count_tokens(n.text) for n in nodes]
print(f"  Max tokens: {max(token_counts)}")
print(f"  Avg tokens: {sum(token_counts) / len(token_counts):.0f}")
print(f"  Nodes > 7000 tokens: {sum(1 for t in token_counts if t > 7000)}")

# Show sample metadata
print("\n" + "="*50)
print("SAMPLE METADATA FROM FIRST 5 NODES:")
print("="*50)
for i, node in enumerate(nodes[:5]):
    tokens = count_tokens(node.text)
    print(f"\nNode {i+1} ({tokens} tokens):")
    print(f"  Main Chapter: {node.metadata.get('main_chapter')}")
    print(f"  Subchapter: {node.metadata.get('subchapter')}")
    print(f"  Article #: {node.metadata.get('article_number')}")
    print(f"  Type: {node.metadata.get('document_type')}")
    print(f"  Text preview: {node.text[:100]}...")

# ===============================
# Connect to Weaviate
# ===============================
print("\nConnecting to Weaviate...")
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
)

# ===============================
# Delete old class if exists
# ===============================
class_name = "InternationalLawDocument"
try:
    existing_classes = client.collections.list_all()
    if class_name in existing_classes:
        print(f"Deleting existing class '{class_name}'...")
        client.collections.delete(class_name)
        print(f"✓ Deleted existing class")
except Exception as e:
    print(f"Note: {e}")

# ===============================
# Create vector store
# ===============================
print(f"\nCreating vector store...")
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name=class_name,
    text_key="content",
)

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(
    docstore=docstore,
    vector_store=vector_store,
)

embedding_model = OpenAIEmbedding(
    api_key=OPENAI_API_KEY,
    model="text-embedding-ada-002"
)

# ===============================
# Index with embeddings
# ===============================
print("\nIndexing documents...")
print("This may take a while for large documents...")

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embedding_model,
    show_progress=True
)

print(f"\n✓ Indexed {len(nodes)} nodes")

# ===============================
# Save docstore
# ===============================
print("\nSaving docstore...")
storage_context.persist(persist_dir="./storage")
print("✓ Saved docstore to ./storage")

# ===============================
# Verification
# ===============================
print("\n" + "="*50)
print("VERIFICATION - Checking indexed data in Weaviate")
print("="*50)

try:
    collection = client.collections.get(class_name)
    response = collection.query.fetch_objects(limit=5)

    for i, obj in enumerate(response.objects):
        print(f"\n--- Object {i+1} ---")
        print(f"Main Chapter: {obj.properties.get('main_chapter')}")
        print(f"Subchapter: {obj.properties.get('subchapter')}")
        print(f"Article #: {obj.properties.get('article_number')}")
        print(f"Type: {obj.properties.get('document_type')}")
        print(f"Has vector: {obj.vector is not None}")
        content = obj.properties.get('content', '')
        print(f"Content preview: {content[:150]}...")
        print(f"Tokens: {count_tokens(content)}")
except Exception as e:
    print(f"Error during verification: {e}")

# ===============================
# Close connection properly
# ===============================
client.close()
print("\n✓ Connection closed successfully")
print("\n" + "="*50)
print("INDEXING COMPLETE!")
print("="*50)
print(f"Total nodes indexed: {len(nodes)}")
print(f"Weaviate class: {class_name}")
print(f"Docstore saved to: ./storage")