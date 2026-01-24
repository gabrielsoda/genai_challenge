from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "v1"
    environment: str = "development"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ChromaDB
    chroma_persist_directory: str = "./chroma_data"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    default_top_k: int = 3

    # Logging
    log_level: str = "INFO"
settings = Settings()
