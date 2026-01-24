import uuid
from fastapi import APIRouter
from genai_challenge.api.schemas.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that processes user messages and returns AI responses.

    Currently return a mock response. Will be connected to Ollama LLM
    and converation memory
    """
    # Generate session_id if not provided (for new conversations)
    session_id = request.session_id or str(uuid.uuid4())

    # next steps: Replace with actual LLM call
    # 1. retrieve conversation history from memory
    # 2. build prompt with system message, history and user message
    # 3. Call Ollama for response
    # 4. Save to conversation memory

    mock_response = f"Mock response to: {request.message}"

    return ChatResponse(
        response=mock_response,
        session_id=session_id,
    )