"""
Lightweight SQLite store for chat *metadata* (titles, timestamps).

This is deliberately separate from LangGraph's own checkpoints.db:
- checkpoints.db (via SqliteSaver in graph.py) stores the actual
  conversation state/messages per thread_id.
- chat_sessions.db (this file) stores just enough info to let the
  Chainlit UI list "your chats" and give them readable titles.
"""

import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = "chat_sessions.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def create_chat(title: str = "New Chat") -> str:
    """Creates a new chat session row and returns its thread_id."""
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_sessions (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (thread_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return thread_id


def list_chats(limit: int = 20):
    """Returns recent chats, most recently updated first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT thread_id, title, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [{"thread_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def touch_chat(thread_id: str, title: str | None = None):
    """Updates last-active time, and optionally renames the chat."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    if title:
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ?, title = ? WHERE thread_id = ?",
            (now, title, thread_id),
        )
    else:
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE thread_id = ?",
            (now, thread_id),
        )
    conn.commit()
    conn.close()


def get_chat_title(thread_id: str):
    conn = _get_conn()
    row = conn.execute(
        "SELECT title FROM chat_sessions WHERE thread_id = ?", (thread_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None