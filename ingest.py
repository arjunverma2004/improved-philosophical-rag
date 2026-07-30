import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import library_store
from paths import UPLOAD_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

def get_embeddings_model():
    """Initializes the lightweight local Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def _get_chroma_store():
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings_model(),
        collection_name=CHROMA_COLLECTION_NAME,
    )

def process_and_store_document(file_path: str):
    """Loads a PDF or TXT, chunks it locally, and stores it in Chroma."""
    
    # 1. Dynamically select the loader
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Please upload a PDF or TXT file.")

    print(f"Loading document: {file_path}...")
    raw_documents = loader.load()

    # 2. Initialize the Recursive Character Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       
        chunk_overlap=200,     
        length_function=len,
        separators=["\n\n", "\n", " ", ""] 
    )

    # 3. Split the Text
    print("Applying recursive character chunking...")
    docs = text_splitter.split_documents(raw_documents)
    print(f"Created {len(docs)} chunks.")

    # 4. Ingest into Chroma DB 
    print("Ingesting chunks into local vector database using Hugging Face...")
    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings_model(),
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # 5. Record that this file has been ingested, so the UI can show it in the
    # library list and skip re-ingesting the same file next time.
    library_store.record_ingested(os.path.basename(file_path), len(docs))

    print("Ingestion complete!")


def delete_document(filename: str):
    """Removes every chunk belonging to `filename` from the Chroma library.

    Chunks are matched by their 'source' metadata field, which was set at
    ingestion time to the full path the file was loaded from
    (UPLOAD_DIR/filename) — reconstructing that same path here is what lets
    this find the right chunks to delete.
    """
    file_path = os.path.join(UPLOAD_DIR, filename)
    store = _get_chroma_store()
    store.delete(where={"source": file_path})
    print(f"Deleted all chunks for {filename} from the vector library.")