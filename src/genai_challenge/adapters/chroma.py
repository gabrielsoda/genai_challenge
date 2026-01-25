"""
ChormaDB adapter for vector storage and retrieval

Uses LangChain's Chroma integration with local sentence-transformers embeddings.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from genai_challenge.config import settings


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get the embedding model for vectorizing text.

    Uses sentence-transformers model specified in settings.
    Runs locally, no API calls needed.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},  # Use cuda if gpu available
    )


def get_vector_store() -> Chroma:
    """
    Get or create the ChromaDB vector store

    Returns:
        Chroma instance connected to persistent storage.
    """
    # ensure persist directory exists
    persist_dir = Path(settings.chroma_persist_directory)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name="acme_docs",
        embedding_function=get_embeddings(),
        persist_directory=str(persist_dir),
    )


def similarity_search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Search for similar documents in the vector store.

    Args:
        query: The search query
        top_k: Number of results to return (default from settings)

    Returns:
        List of dicts with 'content' and 'metadata' keys
    """
    k = top_k or settings.default_top_k
    vector_store = get_vector_store()

    results = vector_store.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in results
    ]
