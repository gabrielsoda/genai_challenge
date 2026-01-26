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

# Make entrypoint executable
RUN chmod +x /app/scripts/entrypoint.sh

# Install the package
RUN uv pip install -e .

# Expose port
EXPOSE 8000

# Use entrypoint for auto-setup
ENTRYPOINT ["/app/scripts/entrypoint.sh"]