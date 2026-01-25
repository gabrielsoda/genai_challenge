"""
Integration tests for API endpoints.
"""


class TestHealthcheck:
    """Test for GET /api/v1/healthcheck"""

    def test_healthcheck_returns_200(self, client):
        response = client.get("/api/v1/healthcheck")
        assert response.status_code == 200

    def test_healthcheck_returns_healthy_status(self, client):
        response = client.get("/api/v1/healthcheck")
        assert response.json() == {"status": "healthy"}


class TestChat:
    """Tests for POST /api/v1/chat"""

    def test_chat_returns_200(self, client, mocker):
        # Mock the LLM to avoid real Ollama calls in tests
        mocker.patch(
            "genai_challenge.services.llm_service.generate_response",
            return_value="Mocked response",
        )

        response = client.post("/api/v1/chat", json={"message": "Hello there!"})
        data = response.json()

        assert "session_id" in data
        assert "response" in data
        assert len(data["session_id"]) > 0

    def test_chat_preserves_session_id(self, client, mocker):
        mocker.patch(
            "genai_challenge.services.llm_service.generate_response",
            return_value="Mocked response",
        )

        # First request - no session_id
        response1 = client.post("/api/v1/chat", json={"message": "Hello"})
        session_id = response1.json()["session_id"]

        # Second request - with session_id
        response2 = client.post(
            "/api/v1/chat", json={"message": "Hello again", "session_id": session_id}
        )
        assert response2.json()["session_id"] == session_id

    def test_chat_rejects_empty_message(self, client):
        response = client.post("/api/v1/chat", json={"message": ""})
        assert response.status_code == 422  # validation error
