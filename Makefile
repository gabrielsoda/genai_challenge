.PHONY: install setup ingest run frontend test lint format docker docker-down clean help all

# Default target
help:
	@echo "GenAI Challenge - Available commands:"
	@echo ""
	@echo "  make install    - Install dependencies with uv"
	@echo "  make setup      - Set up environment (copy .env, pull Ollama model)"
	@echo "  make ingest     - Ingest documents into vector store"
	@echo "  make run        - Run the backend API server"
	@echo "  make frontend   - Run the Streamlit frontend"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter (ruff)"
	@echo "  make format     - Format code with ruff"
	@echo "  make docker     - Build and run with Docker Compose"
	@echo "  make docker-down - Stop Docker services"
	@echo "  make clean      - Clean up generated files"
	@echo "  make all        - Full setup (install, setup, ingest)"
	@echo ""

# Install dependencies
install:
	uv sync
	uv pip install -e .

# Set up environment
setup:
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from template"; fi
	@echo "Pulling Ollama model (requires Ollama to be running)..."
	-ollama pull llama3.2:3b

# Ingest documents into vector store
ingest:
	uv run python scripts/ingest_documents.py data/documents/

# Run backend API server
run:
	uv run uvicorn genai_challenge.main:app --app-dir src --reload --host 0.0.0.0 --port 8000

# Run Streamlit frontend
frontend:
	uv run streamlit run src/genai_challenge/frontend/app.py

# Run tests
test:
	uv run pytest tests/ -v

# Run linter
lint:
	uv run ruff check src/ tests/ scripts/
	uv run ruff format --check src/ tests/ scripts/

# Format code
format:
	uv run ruff format src/ tests/ scripts/

# Build and run with Docker
docker:
	docker compose up --build

# Stop Docker services
docker-down:
	docker compose down

# Clean up generated files
clean:
	rm -rf chroma_data/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Full setup: install, setup env, and ingest
all: install setup ingest
	@echo "Setup complete! Run 'make run' to start the backend."