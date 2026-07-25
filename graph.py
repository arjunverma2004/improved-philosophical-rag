from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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

def build_crag_graph():
    """Builds and compiles the Corrective RAG (CRAG) graph."""
    
    # 1. Initialize Graph and Memory
    g = StateGraph(State)
    memory = MemorySaver()

    # 2. Add CRAG Nodes
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate_node)

    # 3. Define Execution Flow
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")

    # The CRAG Conditional Edge: Decides whether to trust the local docs or search the web
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query",
        }
    )

    # Web Search Fallback Pathway
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")

    # Generation Pathway
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    # 4. Compile with Memory
    app = g.compile(checkpointer=memory)
    return app

# Export the compiled app
app = build_crag_graph()