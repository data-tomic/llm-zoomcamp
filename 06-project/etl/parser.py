# etl/parser.py

from bs4 import BeautifulSoup
from typing import Dict

def parse_rec_to_markdown(rec_json: Dict) -> str:
    """
    Parses a detailed clinical recommendation JSON object into a clean Markdown string.
    Returns an empty string if parsing is not possible.
    """
    if not rec_json or not isinstance(rec_json, dict):
        return ""

    main_title = rec_json.get('name', 'Untitled Recommendation')
    markdown_content = f"# {main_title}\n\n"

    sections = rec_json.get('obj', {}).get('sections', [])
    if not sections:
        # This signals that we might need to use the PDF fallback
        return ""

    for section in sections:
        section_title = section.get('title', 'Untitled Section')
        html_content = section.get('content', '')

        markdown_content += f"## {section_title}\n\n"

        if html_content:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Using .get_text() is robust and simple
            clean_text = soup.get_text(separator='\n', strip=True)
            markdown_content += f"{clean_text}\n\n"
            
    return markdown_content.strip()
