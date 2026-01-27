"""
Unit tests for conversation memory store
"""

import pytest

from genai_challenge.services.memory import ConversationStore


class TestConversationStore:
    """Test for ConversationStore"""

    @pytest.fixture
    def store(self):
        """New ConversationStore for each test."""
        return ConversationStore()

    def test_get_history_empty_session(self, store):
        history = store.get_history("new-session")
        assert history == []

    def test_add_interaction(self, store):
        store.add_interaction("session-1", "Hello", "Hi there!")
        history = store.get_history("session-1")

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi there!"}

    def test_multiple_interactions(self, store):
        store.add_interaction("session-1", "Hello", "Hi!")
        store.add_interaction("session-1", "How are you?", "I'm good!")

        history = store.get_history("session-1")
        assert len(history) == 4

    def test_sessions_are_isolated(self, store):
        store.add_interaction("session-1", "Hello", "Hi!")
        store.add_interaction("session-2", "Goodbye", "Bye!")

        history1 = store.get_history("session-1")
        history2 = store.get_history("session-2")

        assert len(history1) == 2
        assert len(history2) == 2
        assert history1[0]["content"] == "Hello"
        assert history2[0]["content"] == "Goodbye"

    def test_clear_session(self, store):
        store.add_interaction("session-1", "Hello", "Hi!")
        store.clear_session("session-1")

        history = store.get_history("session-1")
        assert history == []

    def test_clear_nonexistent_session_no_error(self, store):
        # should not raise.
        store.clear_session("nonexistent")

    def test_get_history_returns_copy(self, store):
        store.add_interaction("session-1", "Hello", "Hi!")
        history = store.get_history("session-1")

        # modifying returned list shouldn't affect store
        history.append({"role": "user", "content": "Modified"})

        assert len(store.get_history("session-1")) == 2
