"""
Streamlit frontend GenAI Challenge

web UI for:
- Chat with the LLM (with conversation memory)
- RAG Q&A over company documents
"""

import streamlit as st
import requests
import os
# backend API URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

def call_chat_api(message: str, session_id: str | None = None) -> dict:
    """Call the /chat endpoint"""
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    response = requests.post(f"{API_BASE_URL}/chat", json=payload)
    response.raise_for_status()
    return response.json()

def call_rag_api(query: str, top_k: int = 3) -> dict:
    """Call the /rag-query endpoint"""
    payload = {"query": query, "top_k": top_k}

    response = requests.post(f"{API_BASE_URL}/rag-query", json=payload)
    response.raise_for_status()
    return response.json()


def render_chat_tab():
    """Render the Chat tab content"""
    st.header("Chat")
    st.caption("Chat with the AI assistant.")

    # Initialize session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # display chat memory
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # chat input
    if prompt := st.chat_input("Type your message..."):
        # add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = call_chat_api(prompt, st.session_state.session_id)
                    response = result["response"]
                    st.session_state.session_id = result["session_id"]
                    st.write(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connection to API: {e}")
    
    #sidebar controls for chat
    with st.sidebar:
        st.subheader("Chat Controls")
        if st.button("Clear Conversation"):
            st.session_state.chat_messages = []
            st.session_state.session_id = None
            st.rerun()

        if st.session_state.session_id:
            st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

def render_rag_tab():
    """render the RAG Q&A tab content."""
    st.header("RAG Q&A")
    st.caption("Ask questions about ACME company documents.")

    # Initialize RAG history
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("RAG Settings")
        top_k = st.slider("Documents to retrieve", min_value=1, max_value=10, value=3)

        if st.button("Clear RAG History"):
            st.session_state.rag_history = []
            st.rerun()
    
    # query input
    query = st.text_input("Ask a question about company documents:", key="rag_query")

    if st.button("Search", type="primary") and query:
        try:
            result = call_rag_api(query, top_k)
            st.session_state.rag_history.append({
                "query": query,
                "answer": result["answer"],
                "sources": result["sources"],
            })
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")

    # display RAG history (most recent first)
    for item in reversed(st.session_state.rag_history):
        with st.container():
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")                

            with st.expander(f"Sources ({len(item['sources'])} documents)"):
                for source in item["sources"]:
                    st.markdown(f"**{source['source']}** (chunk {source['chunk_id']})")
                    st.caption(source["content_preview"])
                    st.divider()

            st.divider()

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="ACME GenAI Assistant",
        page_icon="🧨",
        layout="wide",
    )
    st.title("🤖 ACME Assistant")

    # Create tabs for chat and RAG
    tab_chat, tab_rag = st.tabs(["Chat", "RAG Q&A"])

    with tab_chat:
        render_chat_tab()

    with tab_rag:
        render_rag_tab()

    # footer
    st.sidebar.markdown("---")
    st.sidebar.caption("GenAI Challenge - ACME Corpóration")

if __name__ == "__main__":
    main()
