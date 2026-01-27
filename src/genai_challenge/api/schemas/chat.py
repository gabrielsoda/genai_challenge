from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation memory. "
        "If not provided, a new session is created.",
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str = Field(..., description="Assistant response")
    session_id: str = Field(
        ..., description="Session ID for continuing the conversation"
    )