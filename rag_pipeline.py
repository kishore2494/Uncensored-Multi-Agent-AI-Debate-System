# rag_pipeline.py

import os
import time
import ollama
from typing import Union # <-- Import Union

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever


from config import KB_DIRECTORY, VECTOR_STORE_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


def load_documents(directory: str):
    # ... (same as before)
    print(f"Loading documents from {directory}...", flush=True)
    if not os.path.exists(directory):
        print(f"Knowledge base directory not found: {directory}", flush=True)
        return []
    loader = DirectoryLoader(directory, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.", flush=True)
    return documents

def split_text_into_chunks(documents, chunk_size: int, chunk_overlap: int):
    # ... (same as before)
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...", flush=True)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.", flush=True)
    return chunks


def create_embeddings(embedding_model: str):
    # ... (same as before)
    print(f"Creating embeddings model using Ollama: {embedding_model}...", flush=True)
    try:
        embeddings = OllamaEmbeddings(model=embedding_model)
        print("Embeddings model created successfully.", flush=True)
        return embeddings
    except Exception as e:
        print(f"Error creating Ollama embeddings model '{embedding_model}': {e}", flush=True)
        print("Please ensure Ollama is running and the embedding model is pulled.", flush=True)
        return None


def create_vector_store(chunks, embeddings, vector_store_path: str):
    # ... (same as before)
    print(f"Creating vector store at {vector_store_path}...", flush=True)
    if not embeddings:
        print("Embeddings model is not available. Cannot create vector store.", flush=True)
        return None
    try:
        vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=vector_store_path)
        # vector_store.persist() # Deprecated in newer Chroma
        print("Vector store created and persisted.", flush=True)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}", flush=True)
        return None

def load_vector_store(embeddings, vector_store_path: str):
    # ... (same as before)
    print(f"Loading vector store from {vector_store_path}...", flush=True)
    if not os.path.exists(vector_store_path):
        print("Vector store directory not found. Cannot load.", flush=True)
        return None
    if not embeddings:
         print("Embeddings model is not available. Cannot load vector store.", flush=True)
         return None
    try:
        vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
        print("Vector store loaded successfully.", flush=True)
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}", flush=True)
        return None


# Correct the type hint here: BaseRetriever | None becomes Union[BaseRetriever, None]
def get_retriever(vector_store) -> Union[BaseRetriever, None]: # <-- Use Union
    """Gets a retriever object from the vector store using configured k."""
    if not vector_store:
        return None
    print(f"Creating retriever with k={RETRIEVER_K}...", flush=True)
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
        return retriever
    except Exception as e:
        print(f"Error creating retriever: {e}", flush=True)
        return None


def index_knowledge_base(kb_directory: str,
                         vector_store_path: str,
                         embedding_model: str,
                         chunk_size: int,
                         chunk_overlap: int):
    # ... (same as before)
    print("Starting knowledge base indexing/loading...", flush=True)
    embeddings = create_embeddings(embedding_model)
    if not embeddings:
        print("Embedding model creation failed. Cannot index/load knowledge base.", flush=True)
        return None

    if os.path.exists(vector_store_path):
        print("Vector store directory found. Attempting to load...", flush=True)
        vector_store = load_vector_store(embeddings, vector_store_path)
        if vector_store:
            print("Vector store loaded successfully. Skipping indexing.", flush=True)
            return vector_store

    print("Vector store not found or loading failed. Indexing documents...", flush=True)
    documents = load_documents(kb_directory)
    if not documents:
        print("No documents found in KB directory to index.", flush=True)
        return None

    chunks = split_text_into_chunks(documents, chunk_size, chunk_overlap)
    vector_store = create_vector_store(chunks, embeddings, vector_store_path)

    if vector_store:
        print("Verifying created vector store...", flush=True)
        verified_store = load_vector_store(embeddings, vector_store_path)
        if not verified_store:
            print("Warning: Could not load the newly created vector store.", flush=True)
        return verified_store

    return None

# Removed __main__ block