"""
Docuement ingestion script for RAG pipeline

Read documents from a directory, splits them into chunks, and stores them in ChromaDB with embeddings.

Usage:
    uv run python scripts/ingest_documents.py /path/to/documents
"""

import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genai_challenge.adapters.chroma import get_vector_store, get_embeddings
from genai_challenge.config import settings


def load_documents(docs_path: Path) -> list[dict]:
    """
    Load all text files from a directory

    Returns:
        List of dicts with 'content' and 'source' keys
    """
    documents = []

    for file_path in docs_path.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        documents.append(
            {
                "content": content,
                "source": file_path.name,
            }
        )
        print(f"Loaded: {file_path.name}")
    return documents


def split_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into smaller chunks (for better retrieval)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            chunks.append(
                {
                    "content": chunk,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_id": i,
                    },
                }
            )
    return chunks


def ingest_to_chroma(chunks: list[dict]) -> int:
    """
    Store document chunks in ChromaDB

    Returns:
        Number of chunks ingested
    """
    vector_store = get_vector_store()

    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vector_store.add_texts(texts=texts, metadatas=metadatas)

    return len(chunks)


def main(docs_path: str):
    """Main ingestion pipeline"""
    path = Path(docs_path)

    if not path.exists():
        print(f"Error: Path does not exists: {path}")
        sys.exit(1)

    print(f"\n--- DOCUMENT INGESTION ---")
    print(f"Source: {path}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Chink size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")

    # Step 1: Load documents
    print(f"\n[1/3] Loading documents...")
    documents = load_documents(path)
    print(f"  Total documents: {len(documents)}")

    # Step 2: Split into chunks
    print(f"\n[2/3] Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"  Total chunks: {len(chunks)}")

    # Step 3: Ingest to ChromaDB
    print(f"\n[3/3] Ingesting to ChromaDB...")
    count = ingest_to_chroma(chunks)
    print(f"  Ingested: {count} chunks")

    print(f"\n--- Done ---")
    print(f"Vector store location: {settings.chroma_persist_directory}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest_documents.py <documents_directory>")
        sys.exit(1)

    main(sys.argv[1])
