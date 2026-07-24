from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def split_text_into_chunks(docs):
    # Using the same embedding model defined in your vector_db.py
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": "cpu"}
    )
    
    chunker = SemanticChunker(embeddings)
    chunks = chunker.split_documents(docs)
    
    return chunks