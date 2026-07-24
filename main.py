from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil

from data_loader import load_data
from vector_db import create_vector_db, add_documents
from text_splliter import split_text_into_chunks
from retriever import retrieve_with_mmr
from prompt_llm import get_brain
from dotenv import load_dotenv

load_dotenv() # Ensure the API key is loaded

app = FastAPI(title="Philosophical RAG API")
BOOKS_DIR = "uploaded_books"
os.makedirs(BOOKS_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    model_choice: str = "gemini-3.1-flash-lite"  # Defaulting to Gemini now

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(BOOKS_DIR, file.filename)
    
    if os.path.exists(file_path):
         return {"message": f"Skipped: {file.filename} (Already processed)"}
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    data = load_data(file_path)
    chunks = split_text_into_chunks(data)
    
    vector_db = create_vector_db()
    add_documents(vector_db, chunks)
    
    return {"message": f"Processed and embedded {file.filename} successfully."}

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    try:
        vector_db = create_vector_db()
        retriever = retrieve_with_mmr(vector_db)
        
        chain = get_brain(model_choice=request.model_choice)
        retrieved_docs = retriever.invoke(request.query)
        
        response = chain.invoke({
            "context": retrieved_docs,
            "question": request.query
        })
        
        return {
            "answer": response,
            "sources": [doc.page_content for doc in retrieved_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))