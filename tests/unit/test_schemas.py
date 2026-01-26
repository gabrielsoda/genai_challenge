"""
unit test for pydantic schemas validation
"""

import pytest
from pydantic import ValidationError

from genai_challenge.api.schemas.chat import ChatRequest, ChatResponse
from genai_challenge.api.schemas.rag import RAGRequest, RAGResponse, SourceDocument


class TestChatRequest:
    """Tests for ChatRequest schema"""

    def test_valid_message(self):
        request = ChatRequest(message="Hello")
        assert request.message == "Hello"
        assert request.session_id is None

    def test_valid_message_with_session_id(self):
        request = ChatRequest(message="Hello", session_id="abc-123")
        assert request.message == "Hello"
        assert request.session_id == "abc-123"

    def test_rejects_empy_message(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("message",)

    def test_rejects_missing_message(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatResponse:
    """Tests for ChatResponse schema"""

    def test_valid_response(self):
        response = ChatResponse(response="Hi there!", session_id="abc-123")
        assert response.response == "Hi there!"
        assert response.session_id == "abc-123"

    def test_serialization(self):
        response = ChatResponse(response="Hi!", session_id="abc-123")
        data = response.model_dump()

        assert data == {"response": "Hi!", "session_id": "abc-123"}


class TestRAGRequest:
    """tests for RAGRequest schema"""

    def test_valid_query(self):
        request = RAGRequest(query="What is the refund policy?")
        assert request.query == "What is the refund policy?"
        assert request.top_k is None

    def test_valid_query_with_top_k(self):
        request = RAGRequest(query="Question", top_k=5)
        assert request.top_k == 5

    def test_rejects_empty_query(self):
        with pytest.raises(ValidationError) as exc_info:
            RAGRequest(query="")

        errors = exc_info.value.errors()
        assert errors[0]["loc"] == ("query",)

    def test_rejects_top_k_below_minimum(self):
        with pytest.raises(ValidationError) as exc_info:
            RAGRequest(query="Question", top_k=0)

        errors = exc_info.value.errors()
        assert errors[0]["loc"] == ("top_k",)

    def test_rejects_top_k_above_maximum(self):
        with pytest.raises(ValidationError) as exc_info:
            RAGRequest(query="Question", top_k=11)

        errors = exc_info.value.errors()
        assert errors[0]["loc"] == ("top_k",)


class TestSourceDocument:
    """Tests for SourceDocument schema"""

    def test_valid_source_document(self):
        doc = SourceDocument(
            source="policy.txt", chunk_id=1, content_preview="This is a preview..."
        )
        assert doc.source == "policy.txt"
        assert doc.chunk_id == 1


class TestRAGResponse:
    """Tests for RAGResponse schema."""

    def test_valid_response_with_sources(self):
        response = RAGResponse(
            answer="The refund policy is...",
            sources=[
                SourceDocument(
                    source="policy.txt", chunk_id=1, content_preview="Preview..."
                )
            ],
        )
        assert response.answer == "The refund policy is..."
        assert len(response.sources) == 1

    def test_valid_response_empty_sources(self):
        response = RAGResponse(answer="No info found.", sources=[])
        assert response.sources == []

    def test_serialization_nested(self):
        response = RAGResponse(
            answer="Answer",
            sources=[
                SourceDocument(source="doc.txt", chunk_id=0, content_preview="...")
            ],
        )
        data = response.model_dump()

        assert data["answer"] == "Answer"
        assert data["sources"][0]["source"] == "doc.txt"
