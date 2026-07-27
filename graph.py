import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import State
from nodes import (
    retrieve_node,
    eval_each_doc_node,
    rewrite_query_node,
    web_search_node,
    refine,
    generate_node
)
from edges import route_after_eval

# ---------------------------------------------------------
# Persistent (SQLite-backed) checkpointing
# ---------------------------------------------------------
DB_DIR = "checkpoints"
DB_PATH = os.path.join(DB_DIR, "checkpoints.db")
os.makedirs(DB_DIR, exist_ok=True)  # sqlite3 won't create the folder itself

# check_same_thread=False lets the same connection be used across
# threads/async calls, which both FastAPI and Chainlit will do.
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(_conn)
memory.setup()  # creates the checkpoint tables if they don't exist yet


def build_crag_graph(checkpointer=None):
    """Builds and compiles the Corrective RAG (CRAG) graph.

    Pass in whichever checkpointer matches how you'll call the graph:
    - sync SqliteSaver (the module-level `memory` below) if you call
      graph.invoke(...) — e.g. server.py, main.py
    - AsyncSqliteSaver if you call graph.ainvoke(...) — e.g. chainlit_app.py,
      which builds and passes its own instance.
    """
    if checkpointer is None:
        checkpointer = memory

    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")

    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query",
        }
    )

    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")

    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile(checkpointer=checkpointer)


# Default sync-checkpointed app, for callers that use .invoke() (sync):
# e.g. server.py, main.py
app = build_crag_graph()