# etl/chroma_loader.py

import logging
from typing import List, Dict
import chromadb

from . import config

logging.basicConfig(level=config.LOG_LEVEL)

# Initialize the ChromaDB client
try:
    chroma_client = chromadb.HttpClient(
        host=config.CHROMA_HOST, port=config.CHROMA_PORT
    )
    logging.info(
        f"ChromaDB client initialized. Host: {config.CHROMA_HOST}, Port: {config.CHROMA_PORT}"
    )
except Exception as e:
    logging.error(f"Failed to initialize ChromaDB client: {e}")
    chroma_client = None


def load_documents_to_chroma(documents: List[Dict]):
    """
    Loads a list of processed documents (with embeddings) into ChromaDB.
    """
    if not chroma_client:
        logging.error("ChromaDB client not available. Cannot load documents.")
        return

    if not documents:
        logging.warning("No documents to load into ChromaDB.")
        return

    # Get or create the collection
    collection = chroma_client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    logging.info(
        f"Loading {len(documents)} documents into Chroma collection '{config.CHROMA_COLLECTION_NAME}'..."
    )

    # Prepare data for batch upsert
    ids = [doc["id"] for doc in documents]
    embeddings = [
        doc["embedding"].tolist() for doc in documents
    ]  # Convert numpy arrays to lists
    metadatas = [doc["metadata"] for doc in documents]
    contents = [doc["text"] for doc in documents]

    # Use upsert in batches to handle large volumes of data if necessary
    # For this project, a single batch is likely fine.
    try:
        collection.upsert(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=contents
        )
        logging.info("Successfully loaded documents into ChromaDB.")
    except Exception as e:
        logging.error(f"An error occurred while loading documents to ChromaDB: {e}")
