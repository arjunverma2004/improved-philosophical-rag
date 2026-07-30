import os
import requests
import streamlit as st

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

if "thread_sources" not in st.session_state:
    st.session_state.thread_sources = {}  # thread_id -> list of source dicts, for the most recent answer


def _switch_chat(thread_id: str):
    st.session_state.thread_id = thread_id
    st.session_state.renaming = None


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
            st.markdown(f"📄 {book['filename']}  \n`{book['chunk_count']} chunks`")
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
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Only the most recent assistant turn has cached sources available
            # (they reflect the graph's latest retrieval, not a per-message log).
            is_last = i == len(history) - 1
            if is_last and msg["role"] == "assistant":
                cached = st.session_state.thread_sources.get(st.session_state.thread_id)
                if cached:
                    _render_sources(cached)

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

        try:
            sources = api_sources(st.session_state.thread_id)
        except requests.exceptions.RequestException:
            sources = []

        st.session_state.thread_sources[st.session_state.thread_id] = sources

    # Note: auto-titling now happens server-side (server.py handles it after
    # each message), so the frontend doesn't need to manage chat titles itself.
    st.rerun()