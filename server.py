"""
FastAPI backend (API layer) for the Philosophical CRAG chatbot.

This file owns all HTTP concerns: routes, request/response schemas, status
codes, and streaming. The actual RAG/LangGraph logic lives in
chat_service.py, which this file calls into but never duplicates.
"""

import os
import shutil
import traceback
from typing import List
from dotenv import load_dotenv

load_dotenv()  # MUST run before importing chat_service, which loads the LLM/embeddings

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import backend as chat_service
from paths import UPLOAD_DIR

server = FastAPI(
    title="Philosophical CRAG API",
    description="A Corrective RAG API powered by LangGraph and Gemini for philosophical inquiries.",
    version="2.0"
)

# Allow a separately-hosted frontend (Streamlit, React, etc.) to call this API
# from a different origin/port during local development.
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class ChatSummary(BaseModel):
    thread_id: str
    title: str
    updated_at: str

class RenameRequest(BaseModel):
    title: str

class MessageRequest(BaseModel):
    message: str

class Source(BaseModel):
    index: int
    label: str
    snippet: str

class HistoryMessage(BaseModel):
    role: str
    content: str
    sources: List[Source] = []


# ---------------------------------------------------------
# Chat session endpoints
# ---------------------------------------------------------
@server.post("/chats", response_model=ChatSummary)
def create_chat():
    """Creates a new, empty chat session."""
    thread_id = chat_service.create_new_chat()
    return ChatSummary(thread_id=thread_id, title=chat_service.get_chat_title(thread_id), updated_at="")


@server.get("/chats", response_model=List[ChatSummary])
def get_chats():
    """Lists existing chats, most recently active first."""
    return chat_service.list_chats()


@server.patch("/chats/{thread_id}")
def rename_chat(thread_id: str, req: RenameRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    chat_service.rename_chat(thread_id, title)
    return {"thread_id": thread_id, "title": title}


@server.delete("/chats/{thread_id}")
def delete_chat(thread_id: str):
    chat_service.delete_chat(thread_id)
    return {"thread_id": thread_id, "status": "deleted"}


@server.get("/chats/{thread_id}/history", response_model=List[HistoryMessage])
def get_history(thread_id: str):
    return chat_service.load_history(thread_id)


@server.get("/chats/{thread_id}/sources", response_model=List[Source])
def get_sources(thread_id: str):
    """Returns the sources cited in the most recent answer on this thread."""
    return chat_service.get_sources(thread_id)


# ---------------------------------------------------------
# Sending a message (streamed response)
# ---------------------------------------------------------
@server.post("/chats/{thread_id}/message")
def send_message(thread_id: str, req: MessageRequest):
    """Streams the assistant's answer back as plain text chunks.

    Client-side, read this as a streamed response rather than a single JSON
    body — e.g. with `requests.post(..., stream=True)` and
    `response.iter_content(decode_unicode=True)`.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    def token_generator():
        try:
            for token in chat_service.stream_answer(thread_id, req.message):
                yield token
        except Exception as e:
            print("\n--- CHAT STREAM ERROR ---")
            traceback.print_exc()
            print("-------------------------\n")
            yield f"\n\n[ERROR] {str(e)}"
        finally:
            # Auto-title the chat from its first message, same as the UI does
            if chat_service.get_chat_title(thread_id) == "New Chat":
                chat_service.rename_chat(thread_id, req.message.strip()[:40] or "New Chat")
            else:
                chat_service.touch_chat(thread_id)

    return StreamingResponse(token_generator(), media_type="text/plain")


class LibraryBook(BaseModel):
    filename: str
    ingested_at: str
    chunk_count: int


# ---------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------
@server.get("/library", response_model=List[LibraryBook])
def get_library():
    """Lists books already ingested into the vector library."""
    return chat_service.list_ingested_books()


@server.delete("/library/{filename}")
def delete_book(filename: str):
    """Removes a book's chunks from the vector library entirely, and clears
    its 'already ingested' record so it can be re-uploaded later."""
    try:
        chat_service.delete_ingested_book(filename)
    except Exception as e:
        print("\n--- LIBRARY DELETE ERROR TRACEBACK ---")
        traceback.print_exc()
        print("---------------------------------------\n")
        raise HTTPException(status_code=500, detail=f"Failed to delete book: {str(e)}")
    return {"filename": filename, "status": "deleted"}


@server.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads and ingests a PDF/TXT into the shared vector library.
    If the file was already ingested before, skips re-ingesting it and
    reports that back instead."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = chat_service.ingest_document(file_path)

        if result["status"] == "already_ingested":
            return {
                "filename": file.filename,
                "status": "already_ingested",
                "message": "This book is already in the library.",
            }
        return {
            "filename": file.filename,
            "status": "ingested",
            "message": "Successfully uploaded and ingested.",
        }
    except Exception as e:
        print("\n--- UPLOAD ERROR TRACEBACK ---")
        traceback.print_exc()
        print("------------------------------\n")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")