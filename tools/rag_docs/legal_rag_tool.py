# =========================
# Expose the legal RAG knowledge base as a RETRIEVAL tool
# =========================

import logging
from langchain.tools import tool
from tools.rag_docs.rag_retriver import query_documents
from typing import Optional

logger = logging.getLogger(__name__)

@tool(
    name="LegalRAGRetriever",  
    description=str(
        "Retrieve relevant legal or policy documents from the knowledge base. "
        "This tool does NOT generate answers, it only retrieves context."
    )
)
def legal_rag_retriever_tool(question: str, trace: Optional[any] =None) -> str:
    """
    RAG retrieval tool with Langfuse tracing.
    
    Args:
        question (str): The user query.
        trace (Langfuse trace, optional): Session-aware trace from the agent.
    
    Returns:
        str: Formatted retrieved context for the agent.
    """
    logger.info("RAG tool invoked")
    logger.debug(f"Tool input question: {question}")

    # Step 1: Retrieve documents with trace
    nodes = query_documents(question, trace=trace)

    if not nodes:
        logger.warning("No relevant documents found")
        return "NO_RELEVANT_DOCUMENTS_FOUND"

    # Step 2: Context formatting (wrap in Langfuse span)
    if trace:
        with trace.span(name="Context Formatting", input={"question": question}) as span:
            context_blocks = []
            for i, node in enumerate(nodes, start=1):
                context_blocks.append(
                    f"[SOURCE {i}] (node_id={node['node_id']}, similarity={node['similarity']})\n"
                    f"{node['text']}"
                )
            span.end(output={"num_sources": len(nodes)})
    else:
        # Fallback if no trace is passed
        context_blocks = []
        for i, node in enumerate(nodes, start=1):
            context_blocks.append(
                f"[SOURCE {i}] (node_id={node['node_id']}, similarity={node['similarity']})\n"
                f"{node['text']}"
            )

    return "\n\n".join(context_blocks)


# =========================
# LangChain Tool Definition
# =========================
# legal_rag_tool = Tool(
#     name="retrieve_legal_documents",
#     func=legal_rag_retriever_tool,
#     description=(
#         "Retrieve relevant legal or policy documents from the knowledge base. "
#         "This tool does NOT generate answers, it only retrieves context."
#     ),
# )
