GenAI RAG Challenge - LLM-powered chatbot with RAG over company documents.

How to run the app:

Option A: Docker - All required configurations with one command
```bash
make docker
Frontend: http://localhost:8501 | API: http://localhost:8000/docs
```

Option B:
```bash
make all      # Install, configure, ingest documents
make run      # Start backend (terminal 1)
make frontend # Start frontend (terminal 2)
```
Requirements

- Docker & Docker Compose, OR
- Python 3.12+, uv, Ollama running locally
