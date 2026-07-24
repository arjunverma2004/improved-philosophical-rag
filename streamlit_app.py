import streamlit as st
import requests

# Point to your local FastAPI backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Philosophical RAG Assistant", page_icon="📚")
st.title("Philosophical RAG Assistant 📚")

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("1. Upload Books")
    uploaded_files = st.file_uploader("Select PDF books", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Save Books"):
        if uploaded_files:
            with st.spinner("Uploading to backend..."):
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        response = requests.post(f"{API_URL}/upload/", files=files)
                        if response.status_code == 200:
                            st.success(response.json().get("message", "Uploaded successfully"))
                        else:
                            st.error(f"Failed to upload {uploaded_file.name}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to backend. Is FastAPI running?")
        else:
            st.warning("Please upload a file first.")

# --- Main Area: Query & Response ---
st.header("2. Ask a Question")
query = st.text_area("Enter your philosophical question:")

if st.button("Generate Answer"):
    if query:
        with st.spinner("Consulting the texts using Gemini..."):
            try:
                # Send the query to the FastAPI backend
                payload = {"query": query, "model_choice": "gemini-3.1-flash-lite"}
                response = requests.post(f"{API_URL}/ask/", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    
                    st.markdown("### Sources")
                    with st.expander("View retrieved text chunks"):
                        for i, source_text in enumerate(data.get("sources", [])):
                            st.markdown(f"**Chunk {i + 1}:**")
                            st.write(source_text)
                            st.divider()
                else:
                    st.error(f"Backend error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Is FastAPI running?")
    else:
        st.warning("Please enter a question.")