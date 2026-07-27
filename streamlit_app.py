import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph import app, memory
from ingest import process_and_store_document
import chat_store

UPLOAD_DIR = "uploaded_books"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Philosophical CRAG Chatbot", page_icon="🏛️", layout="wide")

# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    existing = chat_store.list_chats(limit=1)
    st.session_state.thread_id = existing[0]["thread_id"] if existing else chat_store.create_chat()

if "renaming" not in st.session_state:
    st.session_state.renaming = None


def _load_history(thread_id: str):
    """Reconstructs past messages for a thread from the LangGraph checkpoint."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = memory.get_tuple(config)
    except Exception:
        checkpoint_tuple = None
    if not checkpoint_tuple:
        return []
    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    return channel_values.get("messages", [])


def _switch_chat(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.renaming = None


# ---------------------------------------------------------
# Sidebar: chat list, new chat, rename, and library upload
# ---------------------------------------------------------
with st.sidebar:
    st.header("🏛️ Chats")

    if st.button("🆕 New Chat", use_container_width=True):
        new_id = chat_store.create_chat()
        _switch_chat(new_id)
        st.rerun()

    st.divider()

    for chat in chat_store.list_chats():
        is_active = chat["thread_id"] == st.session_state.thread_id

        col1, col2 = st.columns([5, 1])
        with col1:
            label = f"**{chat['title']}**" if is_active else chat["title"]
            if st.button(label, key=f"switch_{chat['thread_id']}", use_container_width=True):
                _switch_chat(chat["thread_id"])
                st.rerun()
        with col2:
            if st.button("✏️", key=f"rename_btn_{chat['thread_id']}"):
                st.session_state.renaming = chat["thread_id"]
                st.rerun()

        if st.session_state.renaming == chat["thread_id"]:
            new_title = st.text_input(
                "New name",
                value=chat["title"],
                key=f"rename_input_{chat['thread_id']}",
                label_visibility="collapsed",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Save", key=f"save_{chat['thread_id']}", use_container_width=True):
                    chat_store.touch_chat(chat["thread_id"], title=new_title.strip() or chat["title"])
                    st.session_state.renaming = None
                    st.rerun()
            with rc2:
                if st.button("Cancel", key=f"cancel_{chat['thread_id']}", use_container_width=True):
                    st.session_state.renaming = None
                    st.rerun()

    st.divider()
    st.subheader("📚 Library Management")
    uploaded_file = st.file_uploader("Choose a file (PDF/TXT)", type=["pdf", "txt"])
    if st.button("Ingest Book"):
        if uploaded_file is not None:
            with st.spinner("Processing and chunking document..."):
                try:
                    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    process_and_store_document(file_path)
                    st.success(f"Successfully ingested: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to ingest document: {e}")
        else:
            st.warning("Please select a file first.")

# ---------------------------------------------------------
# Main chat area
# ---------------------------------------------------------
st.title("🏛️ Philosophical CRAG Assistant")
st.caption("Powered by Gemini, LangGraph, and Corrective RAG Architecture")

history = _load_history(st.session_state.thread_id)

if not history:
    with st.chat_message("assistant"):
        st.markdown("Greetings. I am your philosophical assistant. What concepts shall we explore today?")
else:
    for msg in history:
        role = "assistant" if isinstance(msg, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(msg.content)

prompt = st.chat_input("Enter your philosophical inquiry...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {
        "messages": [HumanMessage(content=prompt)],
        "question": prompt,
    }

    def _extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        return str(content)

    def token_stream():
        # stream_mode="messages" yields (message_chunk, metadata) tuples.
        # We only forward tokens from the "generate" node — the grader and
        # query-rewriter nodes also call the LLM and we don't want their
        # internal output shown to the user.
        for msg_chunk, metadata in app.stream(inputs, config=config, stream_mode="messages"):
            if metadata.get("langgraph_node") == "generate":
                text = _extract_text(msg_chunk.content)
                if text:
                    yield text

    with st.chat_message("assistant"):
        try:
            st.write_stream(token_stream())
        except Exception as e:
            st.error(f"Something went wrong: {e}")

        # Pull the final graph state (post-generation) to list cited sources.
        final_state = app.get_state(config)
        good_docs = final_state.values.get("good_docs", []) if final_state else []

        if good_docs:
            with st.expander("📖 Sources"):
                for i, doc in enumerate(good_docs, start=1):
                    meta = doc.metadata or {}
                    page = meta.get("page")
                    source = meta.get("source", "Unknown source")
                    label = f"{source}" + (f" (page {page + 1})" if isinstance(page, int) else "")
                    snippet = doc.page_content[:800]
                    if len(doc.page_content) > 800:
                        snippet += "..."
                    st.markdown(f"**[{i}] {label}**")
                    st.markdown(snippet)
                    st.markdown("---")

    # Auto-title the chat from its first user message
    current_title = chat_store.get_chat_title(st.session_state.thread_id)
    if current_title == "New Chat":
        chat_store.touch_chat(st.session_state.thread_id, title=prompt.strip()[:40] or "New Chat")
    else:
        chat_store.touch_chat(st.session_state.thread_id)

    st.rerun()