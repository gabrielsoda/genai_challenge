"""Streamlit frontend for GenAI Challenge."""

import os

import httpx
import streamlit as st

# Configuration - use environment variable for Docker compatibility
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="ACME GenAI Assistant",
    page_icon="🧨",
    layout="wide",
)

st.title("🤖 ACME Assistant")
st.markdown("Chat with an LLM or ask questions about company documents using RAG.")

# Sidebar for mode selection and settings
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Select Mode",
        ["💬 Chat", "📚 RAG Q&A"],
        help="Chat: Direct conversation with LLM\nRAG Q&A: Questions answered from documents",
    )

    st.divider()

    # Health check
    st.subheader("System Status")
    try:
        response = httpx.get(f"{API_BASE_URL}/healthcheck", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            st.success(f"API: {data['status']}")
        else:
            st.error("API: Error")
    except Exception:
        st.error("API: Offline")
        st.caption("Start the backend: `uv run uvicorn genai_challenge.main:app --app-dir src`")

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []


def call_chat_api(message: str, session_id: str | None = None) -> dict:
    """Call the chat API endpoint."""
    try:
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def call_rag_api(query: str, top_k: int = 3) -> dict:
    """Call the RAG query API endpoint."""
    try:
        response = httpx.post(
            f"{API_BASE_URL}/rag-query",
            json={"query": query, "top_k": top_k},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


# Chat Mode
if mode == "💬 Chat":
    st.header("💬 Chat with LLM")
    st.caption("Have a direct conversation with the language model.")

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = call_chat_api(prompt, st.session_state.chat_session_id)

            if "error" in result:
                st.error(result["error"])
                st.caption("Make sure Ollama is running: `ollama serve`")
            else:
                reply = result.get("response", "No response")
                st.session_state.chat_session_id = result.get("session_id")
                st.write(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.session_state.chat_session_id = None
        st.rerun()

# RAG Mode
else:
    st.header("📚 RAG Q&A")
    st.caption("Ask questions about ACME company documents. Answers are grounded in the document content.")

    # RAG settings in sidebar
    with st.sidebar:
        st.subheader("RAG Settings")
        top_k = st.slider("Number of sources", min_value=1, max_value=5, value=3)

    # Display RAG history
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 Sources ({len(msg['sources'])} documents)"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['source']} (chunk {source['chunk_id']})")
                        st.caption(source["content_preview"])
                        st.divider()

    # RAG input
    if query := st.chat_input("Ask a question about the documents..."):
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = call_rag_api(query, top_k)

            if "error" in result:
                st.error(result["error"])
                st.caption("Make sure the backend is running and Ollama is available.")
            else:
                answer = result.get("answer", "No answer generated")
                sources = result.get("sources", [])

                st.write(answer)

                if sources:
                    with st.expander(f"📄 Sources ({len(sources)} documents)"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** {source['source']} (chunk {source['chunk_id']})")
                            st.caption(source["content_preview"])
                            st.divider()

                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

    # Clear RAG history button
    if st.button("Clear Q&A History"):
        st.session_state.rag_messages = []
        st.rerun()

# Footer
st.divider()
st.caption("GenAI Challenge - ACME Corporation | Built with FastAPI, ChromaDB, and Streamlit")