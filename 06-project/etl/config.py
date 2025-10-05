# etl/config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

# API Configuration
BASE_URL = "https://apicr.minzdrav.gov.ru/api.ashx"
LIST_RECS_URL = f"{BASE_URL}?op=GetJsonClinrecs"
DETAIL_REC_URL_TEMPLATE = f"{BASE_URL}?op=GetClinrec2&id={{id}}"
PDF_REC_URL_TEMPLATE = f"{BASE_URL}?op=GetClinrecPdf&id={{id}}"

# Data Storage Configuration (for MinIO)
# These will be read from environment variables
MINIO_URL = os.getenv("MINIO_URL", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "clinical-recommendations")
STAGING_DIR_MD = "staging/markdown"
STAGING_DIR_PDF = "staging/pdf"
STAGING_DIR_OCR = "staging/ocr_text"

# ChromaDB Configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "clinical_recommendations")

# Vectorization Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_SIZE = 1024  # The number of characters in each chunk
CHUNK_OVERLAP = 128   # The number of characters to overlap between chunks

# Logging Configuration
LOG_LEVEL = "INFO"

# For local development, create a .env file in the project root:
# MINIO_URL=your_minio_url:9000
# MINIO_ACCESS_KEY=your_access_key
# MINIO_SECRET_KEY=your_secret_key
# MINIO_BUCKET_NAME=clinical-recommendations
