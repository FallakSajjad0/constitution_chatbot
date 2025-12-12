# backend/constants.py
import os
from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Processing parameters
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# Vector database settings
COLLECTION_NAME = "pakistan_constitution"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Disable Chroma telemetry (optional)
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# PDF processing
PDF_EXTENSIONS = (".pdf", ".PDF")

# Metadata keys
METADATA_ARTICLE = "article_no"
METADATA_PAGE = "page_no"
METADATA_SOURCE = "source_pdf"
METADATA_CHUNK_ID = "chunk_id"
METADATA_PDF_HASH = "pdf_hash"
METADATA_CHUNK_INDEX = "chunk_index"
METADATA_TOTAL_CHUNKS = "total_chunks"

# Ingestion tracking
SEEN_PDFS_FILE = CHROMA_DIR / "seen_pdfs.pkl"
INGESTION_LOG_FILE = LOGS_DIR / "ingestion.log"

# Performance settings
MAX_PDF_SIZE_MB = 50
BATCH_SIZE = 100