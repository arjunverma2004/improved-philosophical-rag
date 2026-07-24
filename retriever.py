from langchain_chroma import Chroma
from vector_db import create_vector_db


def retrieve_with_mmr(vectorstore):
    # Enable MMR in the retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",                   # <-- This enables MMR
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )
    return retriever


if __name__ == "__main__":
    # Initialize the vector store (make sure to replace with your actual initialization)
    vector_db = create_vector_db()
    
    # Example query
    query = "The Stranger"
    
    # Retrieve documents using MMR
    retriever = retrieve_with_mmr(vector_db)
    results = retriever.invoke(query)
    
    for doc in results:
        print(doc.page_content)