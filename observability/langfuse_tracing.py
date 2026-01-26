from observability.langfuse_client import langfuse

def start_trace(session_id: str, user_input: str):
    """
    Start a session-aware Langfuse trace for a single user turn.
    
    Args:
        session_id (str): Unique ID for the conversation/session.
        user_input (str): The user's question or input.
        
    Returns:
        trace: Langfuse trace object
    """
    return langfuse.trace(
        name="react-agent-session",
        session_id=session_id,
        input={"user_question": user_input},
        metadata={
            "agent_type": "react",
            "memory_enabled": True
        }
    )
