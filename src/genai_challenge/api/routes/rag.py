"""
RAG endpoint - answers questions using retrieved documents.
"""

from fastapi import APIRouter

from genai_challenge.api.schemas.rag import RAGRequest, RAGResponse
from genai_challenge.services.rag_service import rag_query

router = APIRouter()


@router.post("/rag-query", response_model=RAGResponse)
async def rag_query_endpoint(request: RAGRequest) -> RAGResponse:
    """
    Answer questions using Retrieval Augmented Generation.

    - Retrieves relevant documents from the vector store
    - Uses retrieved context to ground the LLM response
    - Returns the answer along with source documents
    """
    result = await rag_query(
        query=request.query,
        top_k=request.top_k,
    )

    return RAGResponse(**result)
