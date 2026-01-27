"""
Unit tests for LLM service

Tests the chat orchestratio logic with mocked dependencies
"""

from unittest.mock import AsyncMock

import pytest

from genai_challenge.core.prompts import SYSTEM_PROMPT
from genai_challenge.services.llm_service import chat
from genai_challenge.services.memory import conversation_store


class TestChatService:
    """Tests for chat() function."""

    @pytest.fixture(autouse=True)
    def clean_memory(self):
        """Clear conversation store before each test."""
        # Store sessions to clean up after test
        yield
        # tests use unique session_ids

    @pytest.fixture
    def mock_generate_response(self, mocker):
        """Mock the Ollama adapter."""
        mock = mocker.patch(
            "genai_challenge.services.llm_service.generate_response",
            new_callable=AsyncMock,
        )
        mock.return_value = "Mocked LLM response"
        return mock

    @pytest.mark.asyncio
    async def test_generates_session_id_when_not_provided(self, mock_generate_response):
        """Should create a new UUID session_id if none provided."""
        response, session_id = await chat(message="Hello")

        assert session_id is not None
        assert len(session_id) == 36  # UUID format: 8-4-4-4-12
        assert "-" in session_id

    @pytest.mark.asyncio
    async def test_uses_provided_session_id(self, mock_generate_response):
        """Should use the session_id passed by caller."""
        response, session_id = await chat(
            message="Hello", session_id="my-custom-session"
        )

        assert session_id == "my-custom-session"

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, mock_generate_response):
        """Should return the response from generate_response."""
        mock_generate_response.return_value = "I am a helpful assistant"

        response, _ = await chat(message="Who are you?")

        assert response == "I am a helpful assistant"

    @pytest.mark.asyncio
    async def test_builds_messages_with_system_prompt(self, mock_generate_response):
        """Should include system prompt in messages sent to LLM."""
        await chat(message="Hello", session_id="test-session-1")

        # Verify generate_response was called
        mock_generate_response.assert_called_once()

        # Get the messages argument
        call_args = mock_generate_response.call_args
        messages = call_args[0][0]  # First positional arg

        # First message should be system prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_includes_user_message(self, mock_generate_response):
        """Should include user message in messages sent to LLM."""
        await chat(message="What is Python?", session_id="test-session-2")

        call_args = mock_generate_response.call_args
        messages = call_args[0][0]

        # Last message should be the user's message
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is Python?"

    @pytest.mark.asyncio
    async def test_saves_interaction_to_memory(self, mock_generate_response):
        """Should save the interaction to conversation store."""
        mock_generate_response.return_value = "Hello human!"
        session_id = "test-session-save"

        await chat(message="Hi there", session_id=session_id)

        history = conversation_store.get_history(session_id)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi there"}
        assert history[1] == {"role": "assistant", "content": "Hello human!"}

    @pytest.mark.asyncio
    async def test_includes_history_in_subsequent_calls(self, mock_generate_response):
        """Should include conversation history in messages."""
        session_id = "test-session-history"

        # First interaction
        await chat(message="My name is Alice", session_id=session_id)

        # Second interaction
        await chat(message="What is my name?", session_id=session_id)

        # Check the messages sent in second call
        call_args = mock_generate_response.call_args
        messages = call_args[0][0]

        # Should have: system + 2 history messages + current user message = 4
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "My name is Alice"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "What is my name?"

    @pytest.mark.asyncio
    async def test_passes_model_name_to_adapter(self, mock_generate_response):
        """Should pass model_name parameter to generate_response."""
        await chat(
            message="Hello", session_id="test-session-model", model_name="llama3:8b"
        )

        call_args = mock_generate_response.call_args
        # model_name is the second positional argument
        assert call_args[0][1] == "llama3:8b"
