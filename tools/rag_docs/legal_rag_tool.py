# =========================
# Expose the legal RAG knowledge base as a RETRIEVAL tool
# =========================

import logging
from langchain.tools import tool
from tools.rag_docs.rag_retriver import query_documents
from typing import Optional

logger = logging.getLogger(__name__)

@tool
def legal_rag_retriever_tool(question: str, trace: Optional[any] =None) -> str:
    """
    Retrieve relevant legal knowledge in genral and information about the United Nations, including principles, uses, laws, articles, and policy documents
    from the knowledge base. 

    Use this tool whenever a user asks a legal question, seeks guidance on regulations, 
    or requests information about legal principles. This tool does NOT generate answers; 
    it only retrieves context that the LLM can then use to generate a grounded response.

    Args:
        question (str): The user’s legal query.
        trace (optional): Langfuse session trace for logging and monitoring.

    Returns:
        str: Formatted retrieved context with source references for the LLM to use.
    """

    logger.info("RAG tool invoked")
    logger.debug(f"Tool input question: {question}")

    # Retrieve documents (NO tracing here)
    nodes = query_documents(question)

    if not nodes:
        logger.warning("No relevant documents found")
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    # Format context for the LLM
    context_blocks = []
    for i, node in enumerate(nodes, start=1):
        context_blocks.append(
            f"[SOURCE {i}] (node_id={node['node_id']}, similarity={node['similarity']})\n"
            f"{node['text']}"
        )

    return "\n\n".join(context_blocks)
