# etl/main.py

import logging
import io

from . import config
from . import downloader
from . import parser
from . import storage
from . import vectorizer
from . import chroma_loader

logging.basicConfig(level=config.LOG_LEVEL)


def run_pipeline():
    """
    Main function to run the full ETL pipeline.
    """
    logging.info("Starting ETL pipeline...")

    # === Phase 1: Data Ingestion (API -> MinIO) ===
    storage.ensure_bucket_exists(config.MINIO_BUCKET_NAME)
    all_recs = downloader.get_recommendations_list()
    if not all_recs:
        logging.error("No recommendations found. Exiting pipeline.")
        return

    relevant_recs = downloader.filter_recommendations(all_recs)

    for i, rec_meta in enumerate(relevant_recs):
        rec_id = rec_meta["id"]
        rec_name = rec_meta.get("name", "Unknown")
        logging.info(
            f"Ingesting recommendation {i+1}/{len(relevant_recs)}: ID {rec_id} - {rec_name}"
        )

        detailed_rec = downloader.get_recommendation_detail(rec_id)
        markdown_content = parser.parse_rec_to_markdown(detailed_rec)

        if markdown_content:
            object_name = f"{config.STAGING_DIR_MD}/KR_{rec_id}.md"
            content_bytes = markdown_content.encode("utf-8")
            storage.upload_to_minio(
                bucket_name=config.MINIO_BUCKET_NAME,
                object_name=object_name,
                content=content_bytes,
                content_type="text/markdown",
            )
        else:
            logging.warning(f"No text content for {rec_id}. Fetching PDF as fallback.")
            pdf_content = downloader.get_recommendation_pdf(rec_id)
            if pdf_content:
                object_name = f"{config.STAGING_DIR_PDF}/KR_{rec_id}.pdf"
                storage.upload_to_minio(
                    bucket_name=config.MINIO_BUCKET_NAME,
                    object_name=object_name,
                    content=pdf_content,
                    content_type="application/pdf",
                )
            else:
                logging.error(f"Could not retrieve any content for {rec_id}.")

    logging.info("--- Data Ingestion Phase Complete ---")

    # === Phase 2: Vectorization (MinIO -> ChromaDB) ===
    logging.info("Starting Vectorization Phase...")

    all_chunked_docs = []

    # List all markdown files from our staging directory in MinIO
    staged_objects = storage.minio_client.list_objects(
        config.MINIO_BUCKET_NAME, prefix=config.STAGING_DIR_MD, recursive=True
    )

    for obj in staged_objects:
        logging.info(f"Processing staged file: {obj.object_name}")
        try:
            response = storage.minio_client.get_object(
                config.MINIO_BUCKET_NAME, obj.object_name
            )
            markdown_content = response.read().decode("utf-8")
            source_id = obj.object_name

            # Chunk the document
            chunked_docs = vectorizer.chunk_document(markdown_content, source_id)
            all_chunked_docs.extend(chunked_docs)

        finally:
            response.close()
            response.release_conn()

    if not all_chunked_docs:
        logging.warning("No documents were chunked. Skipping embedding and loading.")
    else:
        # Create embeddings for all chunks in a single batch
        embedded_docs = vectorizer.create_embeddings(all_chunked_docs)
        # Load the final documents into ChromaDB
        chroma_loader.load_documents_to_chroma(embedded_docs)

    logging.info("--- Vectorization Phase Complete ---")
    logging.info("ETL pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
