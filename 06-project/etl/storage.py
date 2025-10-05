# etl/storage.py

import logging
import io
from minio import Minio
from minio.error import S3Error

from . import config

# It's good practice to initialize the client once and reuse it.
# For a script, we can initialize it at the module level.
try:
    minio_client = Minio(
        config.MINIO_URL,
        access_key=config.MINIO_ACCESS_KEY,
        secret_key=config.MINIO_SECRET_KEY,
        secure=False,  # Set to True if you have TLS/SSL configured
    )
    logging.info("MinIO client initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize MinIO client: {e}")
    minio_client = None


def ensure_bucket_exists(bucket_name: str):
    """Checks if a bucket exists and creates it if it doesn't."""
    if not minio_client:
        logging.error("MinIO client is not available.")
        return

    try:
        found = minio_client.bucket_exists(bucket_name)
        if not found:
            minio_client.make_bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' created successfully.")
        else:
            logging.info(f"Bucket '{bucket_name}' already exists.")
    except S3Error as e:
        logging.error(f"Error checking or creating bucket '{bucket_name}': {e}")


def upload_to_minio(
    bucket_name: str,
    object_name: str,
    content: bytes,
    content_type: str = "application/octet-stream",
):
    """
    Uploads content (bytes) to a specified MinIO bucket.

    :param bucket_name: Name of the target bucket.
    :param object_name: Full path/name of the object in the bucket.
    :param content: The content to upload as bytes.
    :param content_type: The MIME type of the content.
    """
    if not minio_client:
        logging.error(f"Cannot upload '{object_name}': MinIO client is not available.")
        return

    try:
        content_stream = io.BytesIO(content)
        content_length = len(content)

        minio_client.put_object(
            bucket_name,
            object_name,
            content_stream,
            content_length,
            content_type=content_type,
        )
        logging.info(
            f"Successfully uploaded '{object_name}' to bucket '{bucket_name}'."
        )
    except S3Error as e:
        logging.error(f"Failed to upload '{object_name}': {e}")
