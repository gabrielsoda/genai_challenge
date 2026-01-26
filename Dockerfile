# Backend Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install the package
RUN uv pip install -e .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "genai_challenge.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]