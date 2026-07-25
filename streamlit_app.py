import streamlit as st
import requests
import uuid

# Configuration
API_URL = "http://127.0.0.1:8000/chat"
UPLOAD_URL = "http://127.0.0.1:8000/upload"

st.set_page_config(page_title="Philosophical CRAG Chatbot", page_icon="🏛️", layout="centered")
st.title("🏛️ Philosophical CRAG Assistant")
st.caption("Powered by Gemini, LangGraph, and Corrective RAG Architecture")

# ---------------------------------------------------------
# Sidebar: Document Ingestion
# ---------------------------------------------------------
with st.sidebar:
    st.header("📚 Library Management")
    st.write("Upload philosophical texts to expand the knowledge base.")
    
    uploaded_file = st.file_uploader("Choose a file (PDF/TXT)", type=["pdf", "txt"])
    
    if st.button("Ingest Book"):
        if uploaded_file is not None:
            with st.spinner("Processing and chunking document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(UPLOAD_URL, files=files)
                    response.raise_for_status()
                    st.success(f"Successfully ingested: {uploaded_file.name}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to upload document. Ensure backend is running. Details: {e}")
        else:
            st.warning("Please select a file first.")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings. I am your philosophical assistant. What concepts shall we explore today?"}
    ]

# ---------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# Handle User Input
# ---------------------------------------------------------
if prompt := st.chat_input("Enter your philosophical inquiry..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Synthesizing context and reasoning..."):
            try:
                payload = {
                    "thread_id": st.session_state.thread_id,
                    "message": prompt
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status() 
                
                ai_reply = response.json().get("response", "No response received.")
                
                st.markdown(ai_reply)
                st.session_state.messages.append({"role": "assistant", "content": ai_reply})

            except requests.exceptions.RequestException as e:
                st.error(f"**Error connecting to the backend.** Please ensure your FastAPI server is running. Details: `{e}`")