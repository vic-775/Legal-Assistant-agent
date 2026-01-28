# ==========================
# LEGAL ASSISTANT AGENT
# ==========================

import os
from dotenv import load_dotenv
import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.messages import AIMessage, HumanMessage

from tools.rag_docs.legal_rag_tool import legal_rag_retriever_tool
from langfuse.langchain import CallbackHandler
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
lf_callback = CallbackHandler()

# --------------------------
# LLM Setup
# --------------------------
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    callbacks=[lf_callback],
    verbose=True
)

# --------------------------
# Tools List
# --------------------------
tools = [legal_rag_retriever_tool]

# --------------------------
# Prompt for the ReAct agent
# --------------------------
react_prompt = """
You are a Legal Assistant. Answer questions ONLY using the provided context from legal documents.
Use tools if needed to retrieve relevant documents.
Be concise, accurate, and cite the source node IDs when possible.
"""

# --------------------------
# Middleware to handle tool errors
# --------------------------
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

# --------------------------
# Agent creation ad executor
# --------------------------
agent_executor = create_agent(
    tools=tools,
    model=llm,
    middleware=[handle_tool_errors],
    system_prompt=react_prompt,
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
    # Pass session metadata to langfuse callback
    result = agent_executor.invoke(
        {"input": user_question},
        config={
            "callbacks": [lf_callback],
            "metadata": {"langfuse_session_id": session_id}
        }
    )

    # Extract AI message contents
    final_texts = []
    messages = result.get("messages", [])
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            final_texts.append(msg.content)
        elif isinstance(msg, dict) and msg.get("type") == "ai" and msg.get("content"):
            # fallback if messages are dicts
            final_texts.append(msg["content"])

    final_answer = "\n".join(final_texts).strip()

    if not final_answer:
        logger.error(f"Agent returned unexpected response: {result}")
        raise ValueError("Agent did not return a valid output")

    logger.info(f"Legal Agent final answer: {final_answer}")
    return final_answer

