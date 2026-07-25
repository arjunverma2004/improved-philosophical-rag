from state import State

def route_after_eval(state: State) -> str:
    """
    Routes to the next node based on the strict CRAG evaluation verdict.
    
    Returns:
    - "refine": If all local documents are relevant, skip web search.
    - "rewrite_query": If local context is missing or irrelevant, trigger the corrective web search loop.
    """
    verdict = state.get("verdict")
    
    if verdict == "web_search":
        print("---CRAG: IRRELEVANT DOCS DETECTED. ROUTING TO WEB SEARCH---")
        return "rewrite_query"
    elif verdict == "generate":
        print("---CRAG: RELEVANT DOCS DETECTED. ROUTING TO GENERATION---")
        return "refine"
    
    # Fallback to generation if verdict is somehow malformed
    return "refine"