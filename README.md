
# Philosophical RAG Assistant 📚

A Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, and Google Gemini.

This tool allows users to upload PDF books, index them into a local vector database, and ask complex philosophical questions. The AI is strictly prompted to answer only based on the provided text, ensuring high fidelity to the original authors' meanings without hallucinating outside knowledge.

---

## ✨ Core Features

### 1. 📚 Multi-Document Processing & Upload
**How it works:** Users can upload multiple PDF books via the Streamlit sidebar. Files are saved locally in the `uploaded_books/` directory.  

**Benefit:** Easily build a custom library without manual scripting.

---

### 2. ⚡ Smart Caching (Skip Duplicates)
**How it works:** The system checks if a file already exists before processing.  

**Benefit:** Saves time, compute resources, and API usage.

---

### 3. 🎯 Advanced Retrieval with MMR
**How it works:** Uses Maximal Marginal Relevance (MMR) instead of standard similarity search.  

**Benefit:** Improves answer quality by balancing relevance and diversity.

---

### 4. 🔍 Context Transparency (Source Verification)
**How it works:** Displays retrieved text chunks in a “Sources” dropdown.  

**Benefit:** Allows users to verify AI responses.

---

### 5. 💾 Automated Session Logging
**How it works:** Saves each query and response in `saved_responses/` as timestamped files.  

**Benefit:** Maintains a history of research and queries.

---

### 6. 🧠 Strict “No Hallucination” Guardrails
**How it works:** The model is restricted to only use provided text.  

**Benefit:** Ensures accuracy and prevents external bias.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Framework:** LangChain  
- **Vector Database:** ChromaDB  
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)  
- **LLM:** Google Gemini (gemini-2.5-flash-lite)  
- **Document Parsing:** PyPDF  

---

## 🚀 Installation

```bash
git clone <your-repo-url>
cd <your-project-folder>
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## 🎮 Usage

```bash
streamlit run app.py
```

Steps:
1. Upload PDF files
2. Click **Process & Save Books**
3. Ask a question
4. Click **Generate Answer**

---

## 📁 Project Structure

```
app.py                  # Streamlit UI
data_loader.py          # PDF loading
text_splitter.py        # Chunking
vector_db.py            # Vector store
retriever.py            # MMR retrieval
chains.py               # Prompt + LLM
uploaded_books/         # PDFs
saved_responses/        # Logs
books/                  # Saved Books
my_chroma_db/           # Embeddings
```
