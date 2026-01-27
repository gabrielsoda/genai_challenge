"""
LLM service - orchestrates chat interactions.

Combines Ollama adapter with conversation memory for chat with history.
"""

import uuid

from genai_challenge.adapters.ollama import generate_response
from genai_challenge.core.prompts import SYSTEM_PROMPT
from genai_challenge.services.memory import conversation_store


async def chat(
    message: str,
    session_id: str | None = None,
    model_name: str | None = None,) -> tuple[str, str]:
    """
    Process a chat message and return the assistant's response.

    Args:
        message: User's message
        session_id: Optional session ID for conversation continuity
                    If None, a new session is created and an ID is assignated
            model_name: Optional model override.
    
    Return: 
        Tuple of (response_text, session_id)
    """

    # Generate session_id if not provided
    if session_id is None:
        session_id = str(uuid.uuid4())
    
    history = conversation_store.get_history(session_id)

    # messages list: system + history + current message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    # call ollama
    response = await generate_response(messages, model_name)

    # Save interaction to memory
    conversation_store.add_interaction(session_id, message, response)

    return response, session_id