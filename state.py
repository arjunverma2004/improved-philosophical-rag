from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Chat history
    messages: Annotated[list, add_messages]
    
    # RAG & Routing State
    question: str 
    docs: List[Document]
    good_docs: List[Document]
    verdict: str
    reason: str
    strips: List[str]
    kept_strips: List[str]
    refined_context: str
    web_query: str
    web_docs: List[Document]