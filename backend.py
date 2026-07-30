"""
Chat service layer for the Philosophical CRAG chatbot.

This is the internal service/business-logic layer: it talks to
LangGraph/LangChain directly and knows nothing about HTTP or FastAPI.
server.py (the API layer) imports and calls into this module — it's not
part of the public API surface itself.

Keeping this framework-agnostic means it can be unit-tested directly
(no FastAPI test client needed) and reused by any other frontend/adapter
(CLI, Slack bot, a different web framework, etc.) without touching the
RAG/LangGraph logic itself.
"""

import os
from langchain_core.messages import HumanMessage, AIMessage

from graph import app, memory
from ingest import process_and_store_document
import chat_store
import library_store


def _extract_text(content) -> str:
    """Handles LLM responses where .content may be a str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


# ---------------------------------------------------------
# Chat session management
# ---------------------------------------------------------
def create_new_chat(title: str = "New Chat") -> str:
    """Creates a new chat session and returns its thread_id."""
    return chat_store.create_chat(title)


def list_chats(limit: int = 20):
    """Returns recent chats as a list of {'thread_id', 'title', 'updated_at'} dicts,
    most recently active first."""
    return chat_store.list_chats(limit)


def rename_chat(thread_id: str, new_title: str):
    chat_store.touch_chat(thread_id, title=new_title)


def get_chat_title(thread_id: str):
    return chat_store.get_chat_title(thread_id)


def touch_chat(thread_id: str):
    """Marks a chat as recently active without changing its title."""
    chat_store.touch_chat(thread_id)


# ---------------------------------------------------------
# Conversation history + answering
# ---------------------------------------------------------
def load_history(thread_id: str):
    """Returns past messages for a thread as plain dicts:
    [{'role': 'user' | 'assistant', 'content': str}, ...]
    """
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = memory.get_tuple(config)
    except Exception:
        checkpoint_tuple = None

    if not checkpoint_tuple:
        return []

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])

    return [
        {
            "role": "assistant" if isinstance(msg, AIMessage) else "user",
            "content": msg.content,
        }
        for msg in messages
    ]


def stream_answer(thread_id: str, user_message: str):
    """Generator that yields response text chunks as the model generates them.

    Only forwards tokens from the graph's final "generate" node — the CRAG
    grader and query-rewriter nodes also call the LLM internally, and their
    output shouldn't be shown to the user.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=user_message)],
        "question": user_message,
    }

    for msg_chunk, metadata in app.stream(inputs, config=config, stream_mode="messages"):
        if metadata.get("langgraph_node") == "generate":
            text = _extract_text(msg_chunk.content)
            if text:
                yield text


def get_sources(thread_id: str):
    """Returns the sources used in the most recent answer on this thread, as:
    [{'index': 1, 'label': 'oldmansea.pdf (page 12)', 'snippet': '...'}, ...]
    Indexes line up with the [1], [2] citation markers in the answer text.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    docs = state.values.get("good_docs", []) if state else []

    sources = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        raw_source = meta.get("source", "Unknown source")
        page = meta.get("page")

        # Local PDFs/TXTs store a full file path in metadata — show just the
        # filename. Web search results store a URL — show it as-is.
        if isinstance(raw_source, str) and raw_source.startswith(("http://", "https://")):
            label = raw_source
        else:
            label = os.path.basename(str(raw_source))

        if isinstance(page, int):
            label += f" (page {page + 1})"

        sources.append({
            "index": i,
            "label": label,
            "snippet": doc.page_content,
        })

    return sources


# ---------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------
def ingest_document(file_path: str) -> dict:
    """Chunks and stores a PDF/TXT into the vector library.

    Returns {'status': 'already_ingested' | 'ingested', 'filename': ...}
    so callers can tell the user whether anything new actually happened.
    """
    filename = os.path.basename(file_path)

    if library_store.is_ingested(filename):
        return {"status": "already_ingested", "filename": filename}

    process_and_store_document(file_path)
    return {"status": "ingested", "filename": filename}


def is_book_ingested(filename: str) -> bool:
    return library_store.is_ingested(filename)


def list_ingested_books():
    """Returns books already in the library as:
    [{'filename': ..., 'ingested_at': ..., 'chunk_count': ...}, ...]
    """
    return library_store.list_ingested()