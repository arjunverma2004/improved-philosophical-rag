import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

import aiosqlite
import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from graph import build_crag_graph
from nodes import _extract_text
from ingest import process_and_store_document
import chat_store

UPLOAD_DIR = "uploaded_books"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DB_DIR = "checkpoints"
DB_PATH = os.path.join(DB_DIR, "checkpoints.db")

# ---------------------------------------------------------
# Async graph + checkpointer, built lazily once per process.
# Chainlit streams via astream_events(), which needs an
# async-capable checkpointer (AsyncSqliteSaver), not the sync
# SqliteSaver that server.py / main.py use with .invoke().
# ---------------------------------------------------------
_graph_app = None
_checkpointer = None
_init_lock = asyncio.Lock()


async def get_graph():
    global _graph_app, _checkpointer
    async with _init_lock:
        if _graph_app is None:
            os.makedirs(DB_DIR, exist_ok=True)  # aiosqlite won't create the folder either
            conn = await aiosqlite.connect(DB_PATH)
            _checkpointer = AsyncSqliteSaver(conn)
            await _checkpointer.setup()
            _graph_app = build_crag_graph(_checkpointer)
    return _graph_app, _checkpointer


async def _load_history(thread_id: str):
    """Reconstructs the past messages for a thread from the LangGraph checkpoint."""
    _, checkpointer = await get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception:
        checkpoint_tuple = None

    if not checkpoint_tuple:
        return []

    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
    return channel_values.get("messages", [])


# ---------------------------------------------------------
# Sidebar: a persistent, read-only panel listing your chats.
# (Chainlit's ElementSidebar can't hold clickable buttons, so
# switching/renaming still happens via the action buttons under
# each message — this panel is just the always-visible list.)
# ---------------------------------------------------------
async def _refresh_sidebar():
    chats = chat_store.list_chats()
    current = cl.user_session.get("thread_id")

    if not chats:
        content = "No chats yet. Start typing to create one."
    else:
        lines = []
        for c in chats:
            marker = "👉 **" if c["thread_id"] == current else "• "
            closer = "**" if c["thread_id"] == current else ""
            lines.append(f"{marker}{c['title']}{closer}")
        content = "\n\n".join(lines)

    await cl.ElementSidebar.set_title("Your Chats")
    await cl.ElementSidebar.set_elements([cl.Text(name="chat_list", content=content, display="side")])


async def _send_chat_menu():
    actions = [
        cl.Action(name="new_chat", label="🆕 New Chat", payload={}),
        cl.Action(name="list_chats", label="📂 Switch Chat", payload={}),
        cl.Action(name="rename_chat", label="✏️ Rename Chat", payload={}),
    ]
    await cl.Message(content="", actions=actions).send()


async def _start_new_chat():
    thread_id = chat_store.create_chat(title="New Chat")
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(
        content="🏛️ Greetings. I am your philosophical assistant. What concepts shall we explore today?",
        author="Assistant",
    ).send()
    await _refresh_sidebar()
    await _send_chat_menu()


@cl.on_chat_start
async def on_chat_start():
    await _start_new_chat()


@cl.action_callback("new_chat")
async def on_new_chat(action: cl.Action):
    await _start_new_chat()


@cl.action_callback("list_chats")
async def on_list_chats(action: cl.Action):
    chats = chat_store.list_chats()
    if not chats:
        await cl.Message(content="No previous chats yet.").send()
        return

    actions = [
        cl.Action(
            name="resume_chat",
            label=f"💬 {c['title']}",
            payload={"thread_id": c["thread_id"]},
        )
        for c in chats
    ]
    await cl.Message(content="Pick a chat to resume:", actions=actions).send()


@cl.action_callback("resume_chat")
async def on_resume_chat(action: cl.Action):
    thread_id = action.payload.get("thread_id")
    cl.user_session.set("thread_id", thread_id)

    history = await _load_history(thread_id)
    if not history:
        await cl.Message(content="(This chat has no messages yet.)").send()
    else:
        for msg in history:
            is_ai = isinstance(msg, AIMessage)
            await cl.Message(
                content=msg.content,
                author="Assistant" if is_ai else "You",
            ).send()

    await _refresh_sidebar()
    await _send_chat_menu()


@cl.action_callback("rename_chat")
async def on_rename_chat(action: cl.Action):
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        await cl.Message(content="Start or resume a chat first.").send()
        return

    res = await cl.AskUserMessage(content="What would you like to rename this chat to?", timeout=60).send()
    if not res:
        return

    # cl.AskUserMessage's return shape has varied across Chainlit versions
    # (plain dict with "output" vs an object) — handle both.
    if isinstance(res, dict):
        new_title = (res.get("output") or "").strip()
    else:
        new_title = str(getattr(res, "content", res)).strip()

    if new_title:
        chat_store.touch_chat(thread_id, title=new_title)
        await cl.Message(content=f"Renamed chat to **{new_title}**.").send()
        await _refresh_sidebar()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = chat_store.create_chat(title="New Chat")
        cl.user_session.set("thread_id", thread_id)

    # Handle any attached PDF/TXT files as library ingestion requests
    for element in (message.elements or []):
        name = getattr(element, "name", "") or ""
        path = getattr(element, "path", None)
        if path and name.lower().endswith((".pdf", ".txt")):
            file_path = os.path.join(UPLOAD_DIR, name)
            with open(path, "rb") as src, open(file_path, "wb") as dst:
                dst.write(src.read())

            async with cl.Step(name="Ingesting document"):
                process_and_store_document(file_path)

            await cl.Message(content=f"📚 Ingested **{name}** into the library.").send()

    if not message.content or not message.content.strip():
        return

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [HumanMessage(content=message.content)],
        "question": message.content,
    }

    reply = cl.Message(content="")
    await reply.send()

    good_docs = []
    try:
        graph_app, _ = await get_graph()

        async for event in graph_app.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                # Only forward tokens from the final "generate" node — the
                # grader/rewriter nodes also call the LLM and we don't want
                # their internal chatter streamed to the user.
                if event.get("metadata", {}).get("langgraph_node") == "generate":
                    chunk = event["data"]["chunk"]
                    token = _extract_text(chunk.content)
                    if token:
                        await reply.stream_token(token)

            elif kind == "on_chain_end":
                output = event.get("data", {}).get("output")
                if isinstance(output, dict) and "good_docs" in output:
                    good_docs = output["good_docs"]

    except Exception as e:
        await reply.stream_token(f"\n\n⚠️ Something went wrong: {e}")

    # Attach sources as clickable citations, e.g. "[1]" in the text links to
    # the element named "[1]".
    if good_docs:
        elements = []
        for i, doc in enumerate(good_docs, start=1):
            meta = doc.metadata or {}
            page = meta.get("page")
            source = meta.get("source", "Unknown source")
            label = f"{source}" + (f" (page {page + 1})" if isinstance(page, int) else "")
            elements.append(
                cl.Text(
                    name=f"[{i}]",
                    content=f"**Source:** {label}\n\n---\n\n{doc.page_content}",
                    display="side",
                )
            )
        reply.elements = elements

    await reply.update()

    # Auto-title the chat from its first user message
    current_title = chat_store.get_chat_title(thread_id)
    if current_title == "New Chat":
        new_title = message.content.strip()[:40] or "New Chat"
        chat_store.touch_chat(thread_id, title=new_title)
    else:
        chat_store.touch_chat(thread_id)

    await _refresh_sidebar()