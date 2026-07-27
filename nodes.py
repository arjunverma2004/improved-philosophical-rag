from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from state import State
from config import llm, answer_prompt, retriever, web_search_tool


def _extract_text(content) -> str:
    """Handles LLM responses where .content may be a str or a list of content blocks
    (Gemini sometimes returns the latter)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------
# CRAG Grader Setup
# ---------------------------------------------------------
grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing the relevance of a retrieved document to a user question.\n"
               "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
               "Give a binary score 'yes' or 'no' to indicate whether the document is relevant. Return ONLY 'yes' or 'no'."),
    ("user", "Retrieved document: \n\n {document} \n\n User question: {question}")
])

retrieval_grader = grader_prompt | llm

def retrieve_node(state: State) -> State:
    """Fetches documents from the vector store."""
    question = state["question"]
    docs = retriever.invoke(question)
    return {"docs": docs}

def eval_each_doc_node(state: State) -> State:
    """
    CRAG Logic: Evaluates retrieved documents.
    Filters out irrelevant ones and decides if web search is needed.
    """
    question = state["question"]
    docs = state["docs"]

    good_docs = []
    web_search_required = False

    for doc in docs:
        score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        grade = _extract_text(score.content).strip().lower()

        if "yes" in grade:
            good_docs.append(doc)
        else:
            web_search_required = True

    if not good_docs:
        web_search_required = True

    verdict = "web_search" if web_search_required else "generate"

    return {"good_docs": good_docs, "verdict": verdict}

def rewrite_query_node(state: State) -> State:
    """CRAG Logic: Rewrites the query to optimize for web search fallback."""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at reformulating philosophical questions for web search engines. "
                   "Look at the input and isolate the core philosophical concept or philosopher being asked about. "
                   "Return ONLY the optimized search query."),
        ("user", "Original question: {question}")
    ])

    rewriter = rewrite_prompt | llm
    optimized_query = rewriter.invoke({"question": state["question"]})
    return {"web_query": _extract_text(optimized_query.content)}

def web_search_node(state: State) -> State:
    """Executes the web search using the rewritten query."""
    query = state["web_query"]
    web_results = web_search_tool(query)

    current_docs = state.get("good_docs", [])

    from langchain_core.documents import Document
    web_docs = [
        Document(
            page_content=result["content"],
            metadata={"source": result.get("url", "Web search result")},
        )
        for result in web_results
    ]

    current_docs.extend(web_docs)
    return {"good_docs": current_docs}

def refine(state: State) -> State:
    """Compiles the final, numbered context string used for generation + citation."""
    docs = state.get("good_docs", [])
    numbered_blocks = [f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1)]
    refined_text = "\n\n".join(numbered_blocks)
    return {"refined_context": refined_text}

def generate_node(state: State, config: RunnableConfig) -> State:
    """Generates the final response using the refined context.

    Uses .stream() internally (rather than .invoke()) so that callers using
    graph.stream(..., stream_mode="messages") receive token-by-token chunks
    for this node and can forward them live to a UI (e.g. Streamlit's
    st.write_stream).
    """
    chain = answer_prompt | llm
    full_text = ""

    for chunk in chain.stream(
        {
            "messages": state["messages"],
            "refined_context": state.get("refined_context", "")
        },
        config=config,
    ):
        full_text += _extract_text(chunk.content)

    return {"messages": [AIMessage(content=full_text)]}