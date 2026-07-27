import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------
# Gemini API Setup
# ---------------------------------------------------------
# Ensure that the 'GOOGLE_API_KEY' environment variable is set on your system.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=2
)

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly analytical philosophical AI assistant. Answer the user's question ONLY using "
               "the provided context.\n"
               "Synthesize the text carefully. If the context is insufficient, state exactly what information "
               "is missing.\n\n"
               "The context below is split into numbered blocks like [1], [2], etc. "
               "When you use information from a block, cite it inline using that exact bracket notation, "
               "e.g. 'Kierkegaard describes despair as a sickness unto death [1].' "
               "Only cite blocks you actually used, and don't invent citation numbers that aren't in the context.\n\n"
               "Context:\n{refined_context}"),
    MessagesPlaceholder(variable_name="messages")
])

# ---------------------------------------------------------
# Retriever (local Chroma DB)
# ---------------------------------------------------------
def get_retriever():
    """Connects to the local Chroma DB and returns it as a retriever."""
    persist_dir = "./chroma_db"

    # Must match the exact embedding model used in ingest.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="philosophical_library"
    )

    return db.as_retriever(search_kwargs={"k": 4})

retriever = get_retriever()

# ---------------------------------------------------------
# Web search tool (Tavily fallback for CRAG)
# ---------------------------------------------------------
def web_search_tool(query: str):
    search = TavilySearchResults(max_results=3)
    return search.invoke({"query": query})