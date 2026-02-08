from tools.rag_docs.rag_retriver import retrieve_nodes
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ===============================
# Example usage
# ===============================
question = "What are the purposes of the United Nations?"
results = retrieve_nodes(question)

for i, result in enumerate(results):
    print(f"--- Node {i+1} ---")
    print("Content:", result.node.get_content())
    print("Similarity Score:", result.score)
    if hasattr(result.node, "extra_info"):
        print("Metadata:", result.node.extra_info)
    print("\n")
