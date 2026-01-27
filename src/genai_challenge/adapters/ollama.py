"""
Ollama adapter usings LangChain ChatOllama.

This module wraps LangCHain's ChatOllama to keep the arquitecture clean.
Upper layers (services) interact with this adapter, not directly with LangChain.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from genai_challenge.config import settings


def get_chat_model(model_name: str | None = None) -> ChatOllama:
    """
    Factory function. Creates a ChatOllama instance

    Args:
        model_name: Optional model override. Uses default if not provided.
    Returns:
        Configured ChatOllama instance.
    """
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model_name or settings.ollama_model,
    )


async def generate_response(
    messages: list[dict[str, str]],
    model_name: str | None = None,
) -> str:
    """
    Generate a response from Ollama given a list of messages.

    Args:
        messages: list of message dicts with 'role' and 'content' keys.
                Roles: 'system', 'user', 'assistant'
        model_name: Optional model override.
    Returns:
        The assistant's response text (LLM).
    """
    chat_model = get_chat_model(model_name)

    # Convert dicts to Langchain message objects
    langchain_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
    # Call Ollama via Langchain
    response = await chat_model.ainvoke(langchain_messages)

    return response.content
