from data_loader import load_data
from vector_db import create_vector_db, add_documents
from prompt_llm import get_brain
from text_splliter import split_text_into_chunks
from retriever import retrieve_with_mmr
from dotenv import load_dotenv

load_dotenv()

data = load_data("books/The_Stranger_Albert_Camus-removed-removed.pdf")
chunks = split_text_into_chunks(data)
vector_db = create_vector_db()
add_documents(vector_db, chunks)
retriever = retrieve_with_mmr(vector_db)
chain = get_brain()

query = input("Enter your philosophical question: ")
retrieved_docs = retriever.invoke(query)
response = chain.invoke({
    "context": retrieved_docs,
    "question": query
})
print(response)
print([doc.page_content for doc in retrieved_docs])