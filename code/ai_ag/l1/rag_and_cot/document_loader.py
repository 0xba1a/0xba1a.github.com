"""
Document Loader - Load and parse different document types
"""

import json
import csv
import os
import logging
from typing import List, Dict

# Get logger for this module
logger = logging.getLogger(__name__)


def load_json_file(filepath: str) -> Dict:
    """
    Load a JSON file and return its contents.

    Args:
        filepath: Path to the JSON file

    Returns:
        Dictionary containing the JSON data
    """
    logger.debug(f"Attempting to load JSON file: {filepath}")
    if not os.path.exists(filepath):
        logger.warning(f"JSON file not found: {filepath}")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Successfully loaded JSON file: {filepath}")
            return data
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        return {}


def load_csv_file(filepath: str) -> List[Dict]:
    """
    Load a CSV file and return its contents as a list of dictionaries.

    Args:
        filepath: Path to the CSV file

    Returns:
        List of dictionaries containing the CSV data
    """
    logger.debug(f"Attempting to load CSV file: {filepath}")
    if not os.path.exists(filepath):
        logger.warning(f"CSV file not found: {filepath}")
        return []

    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        logger.info(f"Successfully loaded CSV file: {filepath} ({len(data)} rows)")
        return data
    except Exception as e:
        logger.error(f"Error reading CSV {filepath}: {e}")
        return []


def extract_text_with_ocr(filepath: str) -> str:
    """
    Extract text from a PDF file using OCR.
    Requires pdf2image and pytesseract.

    Args:
        filepath: Path to the PDF file

    Returns:
        Extracted text using OCR
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        logger.info(f"Attempting OCR on: {filepath}")
        # Convert PDF to images
        images = convert_from_path(filepath)
        text = ""

        for i, image in enumerate(images):
            # Extract text from each image
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
            logger.debug(f"OCR extracted {len(page_text)} chars from page {i+1}")

        return text
    except ImportError:
        logger.warning("OCR libraries (pdf2image, pytesseract) not installed. Skipping OCR.")
        print("Warning: OCR libraries not installed. Install with: pip install pdf2image pytesseract")
        return ""
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        error_msg = str(e).lower()
        # Check if it's likely a missing poppler issue
        if "poppler" in error_msg:
            print("Warning: Poppler is required for pdf2image. Please install it (e.g., 'sudo apt install poppler-utils').")
        elif "tesseract" in error_msg:
             print("Warning: Tesseract is required for OCR. Please install it (e.g., 'sudo apt install tesseract-ocr').")
        return ""


def load_pdf_file(filepath: str) -> str:
    """
    Extract text from a PDF file.
    Tries standard extraction first, then falls back to OCR if text is sparse.

    Args:
        filepath: Path to the PDF file

    Returns:
        Extracted text as a string
    """
    logger.debug(f"Attempting to load PDF file: {filepath}")
    if not os.path.exists(filepath):
        logger.warning(f"PDF file not found: {filepath}")
        return ""

    text = ""

    # Method 1: Standard pypdf extraction
    try:
        import pypdf

        with open(filepath, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            logger.debug(f"PDF has {num_pages} pages")
            for page in pdf_reader.pages:
                # Try to extract text with layout mode if possible, else default
                try:
                    # extraction_mode="layout" is available in newer pypdf versions
                    page_text = page.extract_text(extraction_mode="layout")
                except TypeError:
                    # Fallback for older versions
                    page_text = page.extract_text()
                except Exception as e:
                    logger.warning(f"Layout extraction failed, falling back to default: {e}")
                    page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        logger.info(f"pypdf extracted {len(text)} characters")

    except ImportError:
        logger.error("pypdf not installed. Install with: pip install pypdf")
        print("Warning: pypdf not installed. Install with: pip install pypdf")
    except Exception as e:
        logger.error(f"pypdf extraction failed: {e}")

    # Method 2: OCR Fallback
    # If text is empty or very short (likely scanned or image-based), try OCR
    # Threshold set to 100 characters as a heuristic
    if len(text.strip()) < 100:
        logger.info("Text content is sparse or empty. Attempting OCR...")
        ocr_text = extract_text_with_ocr(filepath)

        # If OCR produced more text, use it
        if len(ocr_text.strip()) > len(text.strip()):
            text = ocr_text
            logger.info(f"OCR extracted {len(text)} characters")
        else:
            logger.info("OCR did not yield better results")

    return text.strip()


def json_to_text(data: Dict, prefix: str = "") -> str:
    """
    Convert JSON data to a readable text format.

    Args:
        data: Dictionary to convert
        prefix: Prefix for context (e.g., "LinkedIn", "GitHub")

    Returns:
        Formatted text string
    """
    lines = []

    if prefix:
        lines.append(f"=== {prefix} Profile ===\n")

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n{key.upper()}:")
            lines.append(json_to_text(value, ""))
        elif isinstance(value, list):
            lines.append(f"\n{key.upper()}:")
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    lines.append(f"\n{i}.")
                    for k, v in item.items():
                        lines.append(f"  {k}: {v}")
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def load_linkedin_profile(filepath: str) -> str:
    """
    Load and format LinkedIn profile data from CSV.

    Args:
        filepath: Path to LinkedIn Profile.csv

    Returns:
        Formatted text string
    """
    logger.debug(f"Loading LinkedIn profile from: {filepath}")
    # Check if it's a CSV or JSON file
    if filepath.endswith('.csv'):
        rows = load_csv_file(filepath)
        if not rows:
            logger.warning("No data found in LinkedIn CSV file")
            return ""

        # Convert CSV rows to readable text
        lines = ["=== LinkedIn Profile ===\n"]

        for row in rows:
            for key, value in row.items():
                if value and value.strip():  # Only include non-empty values
                    lines.append(f"{key}: {value}")
            lines.append("")  # Add blank line between rows

        result = "\n".join(lines)
        logger.info(f"LinkedIn profile loaded successfully ({len(result)} characters)")
        return result
    else:
        # Fallback to JSON for backward compatibility
        logger.debug("LinkedIn file is JSON format")
        data = load_json_file(filepath)
        if not data:
            return ""
        return json_to_text(data, "LinkedIn")


def load_github_profile(filepath: str) -> str:
    """
    Load and format GitHub profile data.

    Args:
        filepath: Path to github_profile.json

    Returns:
        Formatted text string
    """
    data = load_json_file(filepath)
    if not data:
        return ""

    return json_to_text(data, "GitHub")


def load_all_documents(linkedin_path: str, github_path: str, resume_path: str) -> List[Dict[str, str]]:
    """
    Load all documents and return them as a list.

    Args:
        linkedin_path: Path to LinkedIn Profile.csv
        github_path: Path to GitHub profile JSON
        resume_path: Path to resume PDF

    Returns:
        List of documents with metadata
    """
    logger.info("=== Loading all documents ===")
    documents = []

    # Load LinkedIn
    print("  Loading LinkedIn profile...")
    logger.debug(f"Loading LinkedIn from: {linkedin_path}")
    linkedin_text = load_linkedin_profile(linkedin_path)
    if linkedin_text:
        documents.append({
            "content": linkedin_text,
            "source": "linkedin",
            "type": "profile"
        })
        print(f"    ✓ Loaded ({len(linkedin_text)} characters)")
        logger.info(f"LinkedIn profile loaded: {len(linkedin_text)} characters")
        logger.debug(f"LinkedIn: {json.dumps(linkedin_text, indent=4)[:500]}...")
    else:
        print("    ⚠ File not found or empty")
        logger.warning("LinkedIn profile not loaded")

    # Load GitHub
    print("  Loading GitHub profile...")
    logger.debug(f"Loading GitHub from: {github_path}")
    github_text = load_github_profile(github_path)
    if github_text:
        documents.append({
            "content": github_text,
            "source": "github",
            "type": "profile"
        })
        print(f"    ✓ Loaded ({len(github_text)} characters)")
        logger.info(f"GitHub profile loaded: {len(github_text)} characters")
        logger.debug(f"GitHub: {json.dumps(github_text, indent=4)[:500]}...")
    else:
        print("    ⚠ File not found or empty")
        logger.warning("GitHub profile not loaded")

    # Load Resume
    print("  Loading resume PDF...")
    logger.debug(f"Loading resume from: {resume_path}")
    resume_text = load_pdf_file(resume_path)
    if resume_text:
        documents.append({
            "content": resume_text,
            "source": "resume",
            "type": "pdf"
        })
        print(f"    ✓ Loaded ({len(resume_text)} characters)")
        logger.info(f"Resume loaded: {len(resume_text)} characters")
    else:
        print("    ⚠ File not found or empty")
        logger.warning("Resume not loaded")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)

        # Move start position
        start = end - overlap

    return chunks


def process_documents(documents: List[Dict[str, str]], chunk_size: int = 800, overlap: int = 200) -> List[Dict[str, str]]:
    """
    Process documents into chunks with metadata.

    Args:
        documents: List of documents to process
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of chunks with metadata
    """
    logger.info(f"Processing {len(documents)} documents into chunks")
    logger.debug(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    all_chunks = []

    print("\n  Chunking documents...")
    for doc in documents:
        chunks = chunk_text(doc["content"], chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "source": doc["source"],
                "type": doc["type"],
                "chunk_id": i
            })

        print(f"    {doc['source']}: {len(chunks)} chunks created")
        logger.info(f"Source '{doc['source']}': {len(chunks)} chunks created")

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
