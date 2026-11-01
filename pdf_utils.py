"""
pdf_utils.py
------------
Helper functions for pulling clean plain text out of PDF files (job postings and
resumes) so it can be fed into the ML pipeline. Handles unicode normalization,
private use symbols, and bullet artifacts cleanly.
"""

import re
import unicodedata
import pypdf


def clean_extracted_text(text: str) -> str:
    """Normalize unicode, strip bullet glyphs, and collapse whitespace."""
    if not text:
        return ""

    # Normalize unicode to NFKC to resolve ligatures (e.g. fi, fl) and combined accents
    text = unicodedata.normalize("NFKC", text)

    # Remove private-use area characters and common font bullet artifacts (\uf0b7, \ufffd, etc.)
    text = re.sub(r"[\uf000-\uf8ff\ufffd\u2022\u25cf\u25aa\u25b6\u25c0\u2642\u260e\u2709]", " ", text)

    # Collapse bullet dashes and asterisks at start of lines
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_lines = []
    for ln in lines:
        if not ln:
            continue
        # Strip lone bullets or leading bullet symbols
        ln = re.sub(r"^[\s\*\-\–\—\•\·]+\s*", "", ln)
        if ln and ln not in {"•", "-", "*", "·", "–", "—"}:
            cleaned_lines.append(ln)

    # Join and collapse whitespace
    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and thoroughly clean all text from a single PDF file."""
    text_parts = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            except Exception:
                continue

    raw_text = "\n".join(text_parts)
    return clean_extracted_text(raw_text)
