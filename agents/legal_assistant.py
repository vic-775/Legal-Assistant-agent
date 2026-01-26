# ==========================
# LEGAL ASSISTANT AGENT
# ==========================

import os
from dotenv import load_dotenv
import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from tools.rag_docs.legal_rag_tool import legal_rag_retriever_tool
from langfuse.langchain import CallbackHandler
from observability.langfuse_tracing import start_trace
from observability.langfuse_client import langfuse

# --------------------------
# Logging
# --------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

# --------------------------
# langfuse Callback
# --------------------------
lf_callback = CallbackHandler(langfuse_client=langfuse)

# --------------------------
# LLM Setup
# --------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    callbacks=[lf_callback],
)

# --------------------------
# Tools List
# --------------------------
tools = [legal_rag_retriever_tool]

# --------------------------
# Prompt for the ReAct agent
# --------------------------
react_prompt = PromptTemplate(
    template="""
You are a Legal Assistant. Answer questions ONLY using the provided context from legal documents.
Use tools if needed to retrieve relevant documents.
Be concise, accurate, and cite the source node IDs when possible.

{{input}}
""",
    input_variables=["input"]
)

# --------------------------
# Agent creation ad executor
# --------------------------
agent_executor = create_agent(
    tools=tools,
    llm=llm,
    verbose=True,
    return_intermediate_steps=True,
    prompt=react_prompt,
)

# --------------------------
# Function to run the agent with Langfuse tracing
# --------------------------
def run_legal_agent(session_id: str, user_question: str):
    """
    Run the Legal Assistant ReAct agent with session-aware Langfuse tracing.

    Args:
        session_id (str): Unique conversation/session ID
        user_question (str): The user's legal question

    Returns:
        str: The agent's final answer
    """
    # Start a session-aware trace
    trace = start_trace(session_id, user_question)

    # Step 1: Track LLM generation as a span
    with trace.generation(name="Agent LLM Generation", model="gpt-4o-mini") as gen_span:
        result = agent_executor.invoke({"input": user_question, "trace": trace})

        gen_span.end(
            output=result["output"],
            usage=result.get("usage", {})
        )

    # Step 2: End the trace with final output + intermediate steps
    trace.end(output={
        "final_answer": result["output"],
        "intermediate_steps": result.get("intermediate_steps", [])
    })

    logger.info(f"Legal Agent final answer: {result['output']}")

    return result["output"]
