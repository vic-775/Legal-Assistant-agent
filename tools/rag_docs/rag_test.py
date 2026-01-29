from tools.rag_docs.rag_retriver import query_documents

query = "What are the purposes of the United Nations?"
metadata_filter = {"document_type": "article"}

results = query_documents(query, metadata_filters=metadata_filter, top_k=10)

for i, res in enumerate(results, 1):
    print(f"Result {i}")
    print(f"Similarity Score: {res['similarity_score']}")
    print(f"Metadata: {res['metadata']}")
    print(f"Text:\n{res['text']}\n")

