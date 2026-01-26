# ===============================
# Retriever + LLM RAG Pipeline
# ===============================
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI 
from llama_index.core import PromptTemplate
from tools.rag_docs.rag_retriver import query_documents  

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = OpenAI(
    api_key=OPENAI_API_KEY, 
    model="gpt-4.1-mini",
    temperature=0.2)

def rag_pipeline(question: str):
    """
    Full RAG pipeline:
    1. Retrieve relevant nodes using the existing retriever
    2. Pass retrieved text to the LLM to generate a refined answer

    Args:
        question (str): The user's natural language question.

    Returns:
        dict: {
            "answer": <str>,  # LLM-generated answer
            "retrieved_nodes": <list>  # list of retrieved nodes with text and metadata
        }
    """
    # Step 1: Retrieve relevant documents
    nodes = query_documents(question)

    if not nodes:
        return {"answer": "No relevant documents found.", "retrieved_nodes": []}

    # Step 2: Concatenate node texts as context for LLM
    context_text = "\n\n".join([node["text"] for node in nodes])

    # Step 3: Ask LLM to answer the question using retrieved context
    prompt = f"""
    Answer the following question based only on the context provided.

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """
    prompt_obj = PromptTemplate(template=prompt)
    answer = llm.predict(prompt_obj)

    return {"answer": answer, "retrieved_nodes": nodes}

# result = rag_pipeline("What are the Purposes of the United Nations?")
# result = rag_pipeline("what article talks about THE SECURITY COUNCIL")

# print("ANSWER:\n", result["answer"])
# print("\nRETRIEVED NODES:", len(result["retrieved_nodes"])) 
