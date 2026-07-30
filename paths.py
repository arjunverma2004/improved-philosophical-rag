"""
Shared filesystem/DB constants used across ingestion, retrieval, and
deletion. Centralized here so the upload directory, Chroma persist
directory, and collection name can never drift out of sync between
ingest.py, config.py, server.py, etc.
"""

UPLOAD_DIR = "uploaded_books"
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "philosophical_library"