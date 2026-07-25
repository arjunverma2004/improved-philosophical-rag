import os
import shutil
import traceback
from dotenv import load_dotenv

# MUST BE BEFORE ANY LOCAL IMPORTS
load_dotenv() 

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

# Import the graph and ingestion logic
from graph import app
from ingest import process_and_store_document 

# Initialize FastAPI
server = FastAPI(
    title="Philosophical CRAG API",
    description="A Corrective RAG API powered by LangGraph and Gemini for philosophical inquiries.",
    version="1.0"
)

# Ensure an upload directory exists
UPLOAD_DIR = "uploaded_books"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define data models
class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

@server.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        inputs = {
        "messages": [HumanMessage(content=request.message)],
        "question": request.message,
    }
        
        result = app.invoke(inputs, config=config)
        
        final_message = result["messages"][-1].content
        return ChatResponse(response=final_message)
        
    except Exception as e:
        # THIS PRINTS THE ERROR TO YOUR TERMINAL
        print("\n--- ERROR TRACEBACK ---")
        traceback.print_exc() 
        print("-----------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

@server.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Endpoint to upload and ingest philosophical texts."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save the uploaded file locally
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Call the ingestion logic
        process_and_store_document(file_path)
        
        return {"filename": file.filename, "status": "Successfully uploaded and ingested."}
        
    except Exception as e:
        print("\n--- UPLOAD ERROR TRACEBACK ---")
        traceback.print_exc() 
        print("------------------------------\n")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")