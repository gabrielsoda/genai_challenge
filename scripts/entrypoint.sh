#!/bin/bash
set -e

echo "Checking if documents are ingested..."

# Check if ChromaDB has documents
python -c "
from genai_challenge.adapters.chroma import get_vector_store
vs = get_vector_store()
count = vs._collection.count()
print(f'📊 Documents in ChromaDB: {count}')
if count == 0:
    exit(1)
exit(0)
" && echo "✅ Documents already ingested." || {
    echo "No documents found. Running ingestion..."
    uv run python scripts/ingest_documents.py data/documents/
    echo "✅ Ingestion complete!"
}

echo "Starting server..."
exec uv run uvicorn genai_challenge.main:app --app-dir src --host 0.0.0.0 --port 8000