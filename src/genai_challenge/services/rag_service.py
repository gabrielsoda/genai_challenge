"""
RAG service

Combines docuement retrieval from ChromaDB with LLM generation to answer questions grounded in company documents.
"""

from genai_challenge.adapters.chroma import similarity_search
from genai_challenge.adapters.ollama import generate_response
from genai_challenge.core.prompts import format_rag_prompt


async def rag_query(
    query: str,
    top_k: int | None = None,
) -> dict:
    """
    Anser a question using rag pipeline

    Args:
        query: users question
        top_k: number of docuemnts to retrieve (optional)

    Returns:
        Dict with 'answer' and 'sources' keys
    """
    # 1: retrieve relevant documents
    retrieved_docs = similarity_search(query, top_k=top_k)

    if not retrieved_docs:
        return {
            "answer": "I couldn't find any relevant information in the available documents.",
            "sources": [],
        }
    # 2: build context from retrieved docuements
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc["metadata"].get("source", "Unknown")
        context_parts.append(f"[Document {i}: {source}]\n{doc['content']}")

    context = "\n\n".join(context_parts)

    # 3: Create RAG prompt with context
    system_prompt = format_rag_prompt(context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # 4: response
    answer = await generate_response(messages)

    # 5: format sources for response
    sources = [
        {
            "source": doc["metadata"].get("source", "Unknown"),
            "chunk_id": doc["metadata"].get("chunk_id", 0),
            "content_preview": doc["content"][:200] + "..."
            if len(doc["content"]) > 200
            else doc["content"],
        }
        for doc in retrieved_docs
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
