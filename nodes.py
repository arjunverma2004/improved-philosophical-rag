from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from state import State
from config import llm, answer_prompt, retriever, web_search_tool

# ---------------------------------------------------------
# CRAG Grader Setup (Optimized for GLM / Gemma)
# ---------------------------------------------------------
grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing the relevance of a retrieved document to a user question.\n"
               "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
               "Give a binary score 'yes' or 'no' to indicate whether the document is relevant. Return ONLY 'yes' or 'no'."),
    ("user", "Retrieved document: \n\n {document} \n\n User question: {question}")
])

# Assuming `llm` in config.py is instantiated with your GLM or Gemma model
retrieval_grader = grader_prompt | llm 

def _extract_text(content):
    """Handles Gemini responses where .content may be a str or a list of blocks."""
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
            # If even one document is highly irrelevant in a small k-retrieval, 
            # we trigger web search to supplement the philosophical context.
            web_search_required = True

    # If no good docs were found at all, we definitely need to search
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
    
    # Append web results to the existing good documents
    current_docs = state.get("good_docs", [])
    
    # Assuming web_results returns a list of dictionaries with 'content'
    # Format depends on the specific search tool (e.g., Tavily)
    from langchain_core.documents import Document
    web_docs = [Document(page_content=result["content"]) for result in web_results]
    
    current_docs.extend(web_docs)
    return {"good_docs": current_docs}

def refine(state: State) -> State:
    """Compiles the final context string for generation."""
    docs = state.get("good_docs", [])
    refined_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return {"refined_context": refined_text}

def generate_node(state: State) -> State:
    """Generates the final response using the refined context."""
    out = (answer_prompt | llm).invoke({
        "messages": state["messages"],
        "refined_context": state.get("refined_context", "")
    })
    return {"messages": [AIMessage(content=_extract_text(out.content))]}
