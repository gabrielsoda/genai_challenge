from fastapi import FastAPI

from genai_challenge.api.routes import chat, health, rag

app = FastAPI(
    title="GenAI Challenge API",
    description="RAG-powered conversational assitant",
    version="0.1.0",
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(rag.router, prefix="/api/v1", tags=["rag"])
