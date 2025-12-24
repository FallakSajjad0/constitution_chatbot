# backend/constants.py
import os
from pathlib import Path

# Base directory of backend
BASE_DIR = Path(__file__).parent

# Persistent storage (Railway volume)
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", "/data"))

# Runtime directories (NOT inside repo)
DATA_DIR = PERSIST_DIR / "data"
CHROMA_DIR = PERSIST_DIR / "chroma_db"
EMBEDDINGS_DIR = PERSIST_DIR / "embeddings"
LOGS_DIR = PERSIST_DIR / "logs"

# Create dirs safely at runtime
for d in (DATA_DIR, CHROMA_DIR, EMBEDDINGS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Processing parameters
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

# Vector DB
COLLECTION_NAME = "pakistan_constitution"
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Disable telemetry
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

# Performance
MAX_PDF_SIZE_MB = 50
BATCH_SIZE = 100
