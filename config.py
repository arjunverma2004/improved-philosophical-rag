import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from paths import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

# ---------------------------------------------------------
# Gemini API Setup
# ---------------------------------------------------------
# Ensure that the 'GOOGLE_API_KEY' environment variable is set on your system.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=2,
    max_output_tokens=4096,  # raised from the (unset) default, which was likely truncating long answers
)

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a highly analytical philosophical AI assistant with deep expertise across the history of "
     "philosophy. Answer the user's question ONLY using the provided context below.\n\n"
     "LENGTH & DEPTH: Always write a comprehensive, multi-paragraph answer — treat this like an in-depth essay "
     "response, not a quick summary. At minimum, cover: (1) a clear definition of the key concept(s) involved, "
     "(2) the underlying reasoning or argument, explained step by step, (3) relevant nuance, tensions, or "
     "counterarguments present in the context, and (4) how the idea connects to the broader question asked. "
     "A short, one-paragraph answer is considered incomplete even if it's technically correct — expand on it. "
     "If the context is insufficient for part of the question, say explicitly what information is missing "
     "rather than guessing or cutting the answer short.\n\n"
     "CITATIONS: The context is split into numbered blocks like [1], [2], etc. Cite the specific block number "
     "immediately after the exact claim or sentence it supports.\n"
     "- Put exactly ONE number per bracket.\n"
     "- Never bundle several numbers into one bracket like [1,2,3] — if a sentence draws on more than one "
     "block, write them as separate adjacent brackets instead, e.g. 'as seen in both texts [1][2].'\n"
     "- Only cite blocks you actually used, and never invent a number that isn't present in the context.\n\n"
     "Context:\n{refined_context}"),
    MessagesPlaceholder(variable_name="messages")
])

# ---------------------------------------------------------
# Retriever (local Chroma DB)
# ---------------------------------------------------------
def get_retriever():
    """Connects to the local Chroma DB and returns it as a retriever."""
    # Must match the exact embedding model used in ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    return db.as_retriever(search_kwargs={"k": 4})

retriever = get_retriever()

# ---------------------------------------------------------
# Web search tool (Tavily fallback for CRAG)
# ---------------------------------------------------------
def web_search_tool(query: str):
    search = TavilySearchResults(max_results=3)
    return search.invoke({"query": query})