# etl/downloader.py

import requests
import logging
from typing import List, Dict, Optional

from . import config

logging.basicConfig(level=config.LOG_LEVEL)


def get_recommendations_list() -> List[Dict]:
    """Fetches the full list of clinical recommendations."""
    logging.info("Fetching the list of all clinical recommendations...")
    try:
        response = requests.get(config.LIST_RECS_URL, timeout=30)
        response.raise_for_status()
        logging.info(f"Successfully fetched {len(response.json())} recommendations.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch recommendations list: {e}")
        return []


def filter_recommendations(recs: List[Dict]) -> List[Dict]:
    """Filters for pediatric and approved recommendations."""
    logging.info(f"Filtering {len(recs)} recommendations...")
    filtered = [
        rec
        for rec in recs
        if rec.get("age") in [2, 3] and rec.get("NPC_approved") is True
    ]
    logging.info(f"Found {len(filtered)} relevant recommendations after filtering.")
    return filtered


def get_recommendation_detail(rec_id: str) -> Optional[Dict]:
    """Fetches the detailed JSON for a single recommendation."""
    detail_url = config.DETAIL_REC_URL_TEMPLATE.format(id=rec_id)
    logging.info(f"Fetching details for recommendation ID: {rec_id}...")
    try:
        response = requests.get(detail_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch details for ID {rec_id}: {e}")
        return None


def get_recommendation_pdf(rec_id: str) -> Optional[bytes]:
    """Fetches the PDF content for a single recommendation."""
    pdf_url = config.PDF_REC_URL_TEMPLATE.format(id=rec_id)
    logging.info(f"Fetching PDF for recommendation ID: {rec_id}...")
    try:
        response = requests.get(pdf_url, timeout=60)  # Longer timeout for PDFs
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch PDF for ID {rec_id}: {e}")
        return None
