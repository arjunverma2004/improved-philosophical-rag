from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()


def create_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vector_db = Chroma(collection_name="my_collection", embedding_function=embeddings, persist_directory='my_chroma_db')
    return vector_db

def add_documents(vector_db, documents):
    vector_db.add_documents(documents)
    
    
if __name__ == "__main__":
    vector_db = create_vector_db()
    
    # Example documents to add
    docs = [
        Document(page_content="Albert Camus wrote The Stranger.", metadata={"source": "doc1"}),
        Document(page_content="The Stranger is a philosophical novel.", metadata={"source": "doc2"}),
        Document(page_content="It explores absurdism and existentialism.", metadata={"source": "doc3"}),
    ]

    add_documents(vector_db, docs)

    