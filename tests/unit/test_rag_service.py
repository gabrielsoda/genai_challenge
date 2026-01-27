"""
Unit tests for RAG service.

Tests the RAG pipeline orchestration with mocked dependencies.
"""

from unittest.mock import AsyncMock

import pytest

from genai_challenge.services.rag_service import rag_query


class TestRAGService:
    """Tests for rag_query() function."""

    @pytest.fixture
    def mock_similarity_search(self, mocker):
        """Mock the ChromaDB adapter."""
        return mocker.patch("genai_challenge.services.rag_service.similarity_search")

    @pytest.fixture
    def mock_generate_response(self, mocker):
        """Mock the Ollama adapter."""
        mock = mocker.patch(
            "genai_challenge.services.rag_service.generate_response",
            new_callable=AsyncMock,
        )
        mock.return_value = "Generated answer from LLM"
        return mock

    @pytest.fixture
    def sample_documents(self):
        """Sample documents returned by similarity search."""
        return [
            {
                "content": "Our refund policy allows returns within 30 days.",
                "metadata": {"source": "refund_policy.txt", "chunk_id": 0},
            },
            {
                "content": "To request a refund, contact support@acme.com.",
                "metadata": {"source": "refund_policy.txt", "chunk_id": 1},
            },
        ]

    @pytest.mark.asyncio
    async def test_returns_no_info_message_when_no_documents(
        self, mock_similarity_search, mock_generate_response
    ):
        """Should return special message when no documents are retrieved."""
        mock_similarity_search.return_value = []

        result = await rag_query(query="Unknown topic")

        assert "couldn't find relevant information" in result["answer"]
        assert result["sources"] == []
        # Should NOT call LLM when no documents
        mock_generate_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_answer_and_sources(
        self, mock_similarity_search, mock_generate_response, sample_documents
    ):
        """Should return answer from LLM and formatted sources."""
        mock_similarity_search.return_value = sample_documents
        mock_generate_response.return_value = "You can return items within 30 days."

        result = await rag_query(query="What is the refund policy?")

        assert result["answer"] == "You can return items within 30 days."
        assert len(result["sources"]) == 2

    @pytest.mark.asyncio
    async def test_formats_sources_correctly(
        self, mock_similarity_search, mock_generate_response, sample_documents
    ):
        """Should format source documents with required fields."""
        mock_similarity_search.return_value = sample_documents

        result = await rag_query(query="Refund policy?")

        source = result["sources"][0]
        assert source["source"] == "refund_policy.txt"
        assert source["chunk_id"] == 0
        assert "content_preview" in source

    @pytest.mark.asyncio
    async def test_truncates_long_content_preview(
        self, mock_similarity_search, mock_generate_response
    ):
        """Should truncate content preview if longer than 200 chars."""
        long_content = "A" * 300
        mock_similarity_search.return_value = [
            {
                "content": long_content,
                "metadata": {"source": "doc.txt", "chunk_id": 0},
            }
        ]

        result = await rag_query(query="Question")

        preview = result["sources"][0]["content_preview"]
        assert len(preview) == 203  # 200 chars + "..."
        assert preview.endswith("...")

    @pytest.mark.asyncio
    async def test_keeps_short_content_preview_intact(
        self, mock_similarity_search, mock_generate_response
    ):
        """Should not truncate content preview if 200 chars or less."""
        short_content = "Short content"
        mock_similarity_search.return_value = [
            {
                "content": short_content,
                "metadata": {"source": "doc.txt", "chunk_id": 0},
            }
        ]

        result = await rag_query(query="Question")

        preview = result["sources"][0]["content_preview"]
        assert preview == short_content
        assert "..." not in preview

    @pytest.mark.asyncio
    async def test_passes_top_k_to_similarity_search(
        self, mock_similarity_search, mock_generate_response
    ):
        """Should pass top_k parameter to similarity search."""
        mock_similarity_search.return_value = []

        await rag_query(query="Question", top_k=5)

        mock_similarity_search.assert_called_once_with("Question", top_k=5)

    @pytest.mark.asyncio
    async def test_builds_context_from_documents(
        self, mock_similarity_search, mock_generate_response, sample_documents
    ):
        """Should build context string from retrieved documents."""
        mock_similarity_search.return_value = sample_documents

        await rag_query(query="Refund policy?")

        # Check that generate_response was called with messages
        mock_generate_response.assert_called_once()
        call_args = mock_generate_response.call_args
        messages = call_args[0][0]

        # System message should contain the document content
        system_content = messages[0]["content"]
        assert "refund policy allows returns within 30 days" in system_content
        assert "contact support@acme.com" in system_content

    @pytest.mark.asyncio
    async def test_includes_user_query_in_messages(
        self, mock_similarity_search, mock_generate_response, sample_documents
    ):
        """Should include user query as user message to LLM."""
        mock_similarity_search.return_value = sample_documents

        await rag_query(query="How do I get a refund?")

        call_args = mock_generate_response.call_args
        messages = call_args[0][0]

        # Last message should be user query
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "How do I get a refund?"

    @pytest.mark.asyncio
    async def test_handles_missing_metadata_gracefully(
        self, mock_similarity_search, mock_generate_response
    ):
        """Should handle documents with missing metadata fields."""
        mock_similarity_search.return_value = [
            {
                "content": "Some content",
                "metadata": {},  # No source or chunk_id
            }
        ]

        result = await rag_query(query="Question")

        source = result["sources"][0]
        assert source["source"] == "Unknown"
        assert source["chunk_id"] == 0
