"""
Conversation memory store

Simple in-memory storage conversation history indexed by session_id.
Uses in-memory storage (dict)
>>>> *Consider a database persistence*
"""

class ConversationStore:
    """
    Manages conversation memory for multiple sessions.
    """

    def __init__(self):
        self._sessions: dict[str, list[dict[str, str]]] = {}

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Get conversation history as list of message dicts.
        """
        return self._sessions.get(session_id, []).copy()

    def add_interaction(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str
    ) -> None:
        """
        Save a user message and assistant response to memory
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append({"role": "user", "content": user_message})
        self._sessions[session_id].append({"role": "assistant", "content": assistant_response})

    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session."""
        self._sessions.pop(session_id, None)



# singleton instance for the app
conversation_store = ConversationStore()