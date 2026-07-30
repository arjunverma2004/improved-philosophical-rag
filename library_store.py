"""
Tracks which books have already been ingested into the Chroma library.

Chroma itself doesn't make "have I already added this file?" a cheap or
obvious question to answer, so we keep a small side-table of ingested
filenames here — checked before re-ingesting, and used to populate the
sidebar's "already in the library" list.
"""

import sqlite3
from datetime import datetime, timezone

DB_PATH = "library.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_books (
            filename TEXT PRIMARY KEY,
            ingested_at TEXT NOT NULL,
            chunk_count INTEGER NOT NULL
        )
        """
    )
    return conn


def is_ingested(filename: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM ingested_books WHERE filename = ?", (filename,)
    ).fetchone()
    conn.close()
    return row is not None


def record_ingested(filename: str, chunk_count: int):
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO ingested_books (filename, ingested_at, chunk_count)
        VALUES (?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            ingested_at = excluded.ingested_at,
            chunk_count = excluded.chunk_count
        """,
        (filename, now, chunk_count),
    )
    conn.commit()
    conn.close()


def list_ingested(limit: int = 50):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT filename, ingested_at, chunk_count FROM ingested_books "
        "ORDER BY ingested_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"filename": r[0], "ingested_at": r[1], "chunk_count": r[2]}
        for r in rows
    ]