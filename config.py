import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_retries=2
)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly analytical philosophical AI assistant. Answer the user's question ONLY using the provided context.\n"
               "Synthesize the text carefully. If the context is insufficient, state exactly what information is missing.\n\n"
               "Context:\n{refined_context}"),
    MessagesPlaceholder(variable_name="messages")
])

def get_retriever():
    """Connects to the local Chroma DB and returns it as a retriever."""
    persist_dir = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="philosophical_library"
    )
    return db.as_retriever(search_kwargs={"k": 4})

retriever = get_retriever()

def web_search_tool(query: str):
    search = TavilySearchResults(max_results=3)
    return search.invoke({"query": query})