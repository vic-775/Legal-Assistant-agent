import uuid
from agents.legal_assistant import run_legal_agent

# Generate a unique session ID for this conversation
session_id = str(uuid.uuid4())

# Example user question
user_question = "What are the purposes of the United Nations?"

# Run the Legal Assistant agent
final_answer = run_legal_agent(session_id, user_question)

# Print the final answer
print("\n===== FINAL ANSWER =====")
print(final_answer)
