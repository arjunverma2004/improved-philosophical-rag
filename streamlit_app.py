import os
import requests
import streamlit as st
from urllib.parse import quote

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Philosophical CRAG Chatbot", page_icon="🏛️", layout="wide")


# ---------------------------------------------------------
# Thin API client — every call to the backend goes through here.
# ---------------------------------------------------------
def api_create_chat():
    r = requests.post(f"{API_URL}/chats")
    r.raise_for_status()
    return r.json()["thread_id"]


def api_list_chats():
    r = requests.get(f"{API_URL}/chats")
    r.raise_for_status()
    return r.json()


def api_rename_chat(thread_id: str, title: str):
    r = requests.patch(f"{API_URL}/chats/{thread_id}", json={"title": title})
    r.raise_for_status()


def api_delete_chat(thread_id: str):
    r = requests.delete(f"{API_URL}/chats/{thread_id}")
    r.raise_for_status()
    return r.json()


def api_history(thread_id: str):
    r = requests.get(f"{API_URL}/chats/{thread_id}/history")
    r.raise_for_status()
    return r.json()


def api_sources(thread_id: str):
    r = requests.get(f"{API_URL}/chats/{thread_id}/sources")
    r.raise_for_status()
    return r.json()


def api_stream_message(thread_id: str, message: str):
    """Generator yielding response text chunks as the backend streams them."""
    with requests.post(
        f"{API_URL}/chats/{thread_id}/message",
        json={"message": message},
        stream=True,
    ) as resp:
        resp.raise_for_status()
        resp.encoding = resp.encoding or "utf-8"
        for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def api_upload(filename: str, file_bytes: bytes):
    files = {"file": (filename, file_bytes)}
    r = requests.post(f"{API_URL}/upload", files=files)
    r.raise_for_status()
    return r.json()


def api_list_library():
    r = requests.get(f"{API_URL}/library")
    r.raise_for_status()
    return r.json()


def api_delete_book(filename: str):
    r = requests.delete(f"{API_URL}/library/{quote(filename, safe='')}")
    r.raise_for_status()
    return r.json()


def _stream_with_thinking_indicator(token_generator):
    """Wraps a token generator so a spinner shows during the gap before the
    first token arrives — the CRAG pipeline runs retrieval + relevance
    grading (and sometimes a web search) before generation even starts,
    which otherwise looks like the UI has frozen.
    """
    with st.spinner("Retrieving context and reasoning..."):
        try:
            first_token = next(token_generator)
        except StopIteration:
            return
    yield first_token
    yield from token_generator


# ---------------------------------------------------------
# Session state init
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    try:
        existing = api_list_chats()
        st.session_state.thread_id = existing[0]["thread_id"] if existing else api_create_chat()
    except requests.exceptions.RequestException as e:
        st.error(
            f"Can't reach the backend at {API_URL}. Make sure it's running "
            f"(`uvicorn server:server --reload`).\n\nDetails: {e}"
        )
        st.stop()

if "renaming" not in st.session_state:
    st.session_state.renaming = None

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None

if "confirm_delete_book" not in st.session_state:
    st.session_state.confirm_delete_book = None


def _switch_chat(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.renaming = None
    st.session_state.confirm_delete = None


def _truncate(title: str, max_len: int = 26) -> str:
    """Keeps sidebar chat buttons single-line so they stay the same height
    as the ✏️ rename button next to them — a long title wrapping to two
    lines was what threw off the alignment."""
    if len(title) <= max_len:
        return title
    return title[: max_len - 1].rstrip() + "…"


# ---------------------------------------------------------
# Sidebar: chat list, new chat, rename, and library upload
# ---------------------------------------------------------
with st.sidebar:
    st.header("🏛️ Chats")

    if st.button("🆕 New Chat", use_container_width=True):
        try:
            _switch_chat(api_create_chat())
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Couldn't create a new chat: {e}")

    st.divider()

    try:
        chats = api_list_chats()
    except requests.exceptions.RequestException as e:
        chats = []
        st.error(f"Couldn't load chats: {e}")

    for chat in chats:
        is_active = chat["thread_id"] == st.session_state.thread_id

        col1, col2 = st.columns([5, 1], vertical_alignment="center")
        with col1:
            label = _truncate(chat["title"])
            label = f"**{label}**" if is_active else label
            if st.button(label, key=f"switch_{chat['thread_id']}", use_container_width=True):
                _switch_chat(chat["thread_id"])
                st.rerun()
        with col2:
            if st.button("✏️", key=f"rename_btn_{chat['thread_id']}"):
                st.session_state.renaming = chat["thread_id"]
                st.session_state.confirm_delete = None
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
                    try:
                        api_rename_chat(chat["thread_id"], new_title.strip() or chat["title"])
                    except requests.exceptions.RequestException as e:
                        st.error(f"Couldn't rename chat: {e}")
                    st.session_state.renaming = None
                    st.rerun()
            with rc2:
                if st.button("Cancel", key=f"cancel_{chat['thread_id']}", use_container_width=True):
                    st.session_state.renaming = None
                    st.session_state.confirm_delete = None
                    st.rerun()

            if st.session_state.confirm_delete != chat["thread_id"]:
                if st.button("🗑️ Delete chat", key=f"delete_btn_{chat['thread_id']}", use_container_width=True):
                    st.session_state.confirm_delete = chat["thread_id"]
                    st.rerun()
            else:
                st.warning("Delete this chat permanently? This can't be undone.")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Yes, delete", key=f"confirm_delete_{chat['thread_id']}", use_container_width=True):
                        try:
                            api_delete_chat(chat["thread_id"])
                        except requests.exceptions.RequestException as e:
                            st.error(f"Couldn't delete chat: {e}")

                        st.session_state.renaming = None
                        st.session_state.confirm_delete = None

                        # If we just deleted the active chat, switch to another
                        # one (or create a fresh one if none are left).
                        if st.session_state.thread_id == chat["thread_id"]:
                            try:
                                remaining = api_list_chats()
                            except requests.exceptions.RequestException:
                                remaining = []
                            st.session_state.thread_id = (
                                remaining[0]["thread_id"] if remaining else api_create_chat()
                            )

                        st.rerun()
                with dc2:
                    if st.button("Cancel", key=f"cancel_delete_{chat['thread_id']}", use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()

    st.divider()
    st.subheader("📚 Library Management")
    uploaded_file = st.file_uploader("Choose a file (PDF/TXT)", type=["pdf", "txt"])
    if st.button("Ingest Book"):
        if uploaded_file is not None:
            with st.spinner("Processing and chunking document..."):
                try:
                    result = api_upload(uploaded_file.name, uploaded_file.getvalue())
                    if result.get("status") == "already_ingested":
                        st.warning(f"'{uploaded_file.name}' is already in the library.")
                    else:
                        st.success(f"Successfully ingested: {uploaded_file.name}")
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to ingest document: {e}")
        else:
            st.warning("Please select a file first.")

    st.caption("Books already in the library:")
    try:
        library_books = api_list_library()
    except requests.exceptions.RequestException:
        library_books = []

    if library_books:
        for book in library_books:
            bcol1, bcol2 = st.columns([5, 1], vertical_alignment="center")
            with bcol1:
                st.markdown(f"📄 {book['filename']}  \n`{book['chunk_count']} chunks`")
            with bcol2:
                if st.button("🗑️", key=f"delete_book_btn_{book['filename']}"):
                    st.session_state.confirm_delete_book = book["filename"]
                    st.rerun()

            if st.session_state.confirm_delete_book == book["filename"]:
                st.warning(f"Remove '{book['filename']}' from the library permanently?")
                bd1, bd2 = st.columns(2)
                with bd1:
                    if st.button(
                        "Yes, delete", key=f"confirm_delete_book_{book['filename']}", use_container_width=True
                    ):
                        try:
                            api_delete_book(book["filename"])
                        except requests.exceptions.RequestException as e:
                            st.error(f"Couldn't delete book: {e}")
                        st.session_state.confirm_delete_book = None
                        st.rerun()
                with bd2:
                    if st.button(
                        "Cancel", key=f"cancel_delete_book_{book['filename']}", use_container_width=True
                    ):
                        st.session_state.confirm_delete_book = None
                        st.rerun()
    else:
        st.caption("_No books ingested yet._")

# ---------------------------------------------------------
# Main chat area
# ---------------------------------------------------------
st.title("🏛️ Philosophical CRAG Assistant")
st.caption("Powered by Gemini, LangGraph, and Corrective RAG Architecture")

try:
    history = api_history(st.session_state.thread_id)
except requests.exceptions.RequestException as e:
    history = []
    st.error(f"Couldn't load chat history: {e}")

def _render_sources(sources):
    with st.expander("📖 Sources"):
        for src in sources:
            snippet = src["snippet"][:800]
            if len(src["snippet"]) > 800:
                snippet += "..."
            st.markdown(f"**[{src['index']}] {src['label']}**")
            st.markdown(snippet)
            st.markdown("---")


if not history:
    with st.chat_message("assistant"):
        st.markdown("Greetings. I am your philosophical assistant. What concepts shall we explore today?")
else:
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Each assistant message carries its own sources now (persisted
            # per-message on the backend), so every past answer's citations
            # are shown here, not just the most recent one.
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

prompt = st.chat_input("Enter your philosophical inquiry...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            st.write_stream(
                _stream_with_thinking_indicator(api_stream_message(st.session_state.thread_id, prompt))
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Something went wrong talking to the backend: {e}")

    # Note: auto-titling now happens server-side (server.py handles it after
    # each message), and sources are now persisted per-message on the
    # backend too — so a plain rerun is enough to pick up both.
    st.rerun()