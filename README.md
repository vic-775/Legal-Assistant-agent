# Legal Assistant Agent with RAG & Langfuse Tracking

## Project Description
This project is a **Legal Assistant Agent** that uses a **Retriever-Augmented Generation (RAG)** pipeline to answer legal and policy questions. The system retrieves relevant documents from a Weaviate vector store and passes them to a large language model (LLM) to generate grounded responses. All interactions, including tool calls, retrieved documents, and model outputs, are tracked with **Langfuse** for session-aware observability.

---

## Features
- **RAG Pipeline:** Retrieve relevant legal documents and feed them to an LLM.
- **Langfuse Tracking:** Full observability of sessions, inputs, outputs, costs, tokens, and retrieved documents.
- **Extensible Tools:** Modular tools architecture, currently includes a Legal RAG retriever; more tools can be added easily.
- **Logging:** Comprehensive logging to debug and track pipeline execution.

---

## Project Structure

- agents/                # Agent definitions and runners
- tools/                 # RAG tools and retrievers
- observability/         # Langfuse tracing and logging setup
- .env                   # Environment variables for API keys and configuration
- requirements.txt       # Python dependencies
