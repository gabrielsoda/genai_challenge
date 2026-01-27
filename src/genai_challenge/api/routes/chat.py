"""
Chat endpoint - processes user messages through the LLM.
"""

from fastapi import APIRouter

from genai_challenge.api.schemas.chat import ChatRequest, ChatResponse
from genai_challenge.services.llm_service import chat as llm_chat

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that processes user messages and returns AI responses.

    - Mantains conversation history via session_id
    - Ollama LLM through LangChain
    - Applies system prompt for consistency
    """
    response_text, session_id = await llm_chat(
        message=request.message,
        session_id=request.session_id,
    )
    return ChatResponse(
        response=response_text,
        session_id=session_id,
    )