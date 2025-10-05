# etl/vectorizer.py

import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from . import config

logging.basicConfig(level=config.LOG_LEVEL)

# Initialize the model and splitter once to be reused.
# This is a heavyweight object, so we avoid re-creating it.
try:
    logging.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load embedding model: {e}")
    model = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len,
)


def chunk_document(doc_text: str, source_id: str) -> List[Dict]:
    """
    Splits a document text into chunks and prepares them for embedding.
    Each chunk is a dictionary containing the text and metadata.
    """
    if not doc_text:
        return []

    logging.info(f"Chunking document: {source_id}...")
    chunks = text_splitter.split_text(doc_text)

    chunked_docs = []
    for i, chunk in enumerate(chunks):
        chunked_docs.append(
            {
                "id": f"{source_id}_{i}",
                "text": chunk,
                "metadata": {"source": source_id, "chunk_number": i},
            }
        )
    logging.info(f"Document {source_id} was split into {len(chunked_docs)} chunks.")
    return chunked_docs


def create_embeddings(chunked_docs: List[Dict]) -> List[Dict]:
    """

    Creates vector embeddings for a list of document chunks.
    """
    if not model:
        logging.error("Embedding model is not available. Cannot create embeddings.")
        return chunked_docs

    if not chunked_docs:
        return []

    logging.info(f"Creating embeddings for {len(chunked_docs)} chunks...")
    # Extract just the text for batch processing
    texts_to_embed = [doc["text"] for doc in chunked_docs]

    # Create embeddings in a single batch for efficiency
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    # Add the embedding vector to each chunk dictionary
    for i, doc in enumerate(chunked_docs):
        doc["embedding"] = embeddings[i]

    logging.info("Embeddings created successfully.")
    return chunked_docs
