"""
Pydantic schemas for RAG endpoint
"""

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Request body for RAG query endpoint."""

    query: str = Field(..., min_length=1, description="User's question")
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Number of documents to retrieve"
    )


class SourceDocument(BaseModel):
    """Information about a retrieved source document."""

    source: str = Field(..., description="Document filename")
    chunk_id: int = Field(..., description="Chunk number within document")
    content_preview: str = Field(..., description="Preview of chunk content")


class RAGResponse(BaseModel):
    """Response from RAG query endpoint."""

    answer: str = Field(..., description="Generated answer based on documents")
    sources: list[SourceDocument] = Field(..., description="Source documents used")
