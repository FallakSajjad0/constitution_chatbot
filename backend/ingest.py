# backend/ingest.py
import os
import pickle
import hashlib
import warnings
from pathlib import Path

# Force HuggingFace cache to persistent volume
from constants import EMBEDDINGS_DIR
os.environ["HF_HOME"] = str(EMBEDDINGS_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(EMBEDDINGS_DIR)

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from constants import (
    DATA_DIR,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
    EMBEDDINGS_MODEL,
    PDF_EXTENSIONS,
    METADATA_ARTICLE,
    METADATA_PAGE,
    METADATA_SOURCE,
    METADATA_CHUNK_ID,
    METADATA_PDF_HASH,
    METADATA_CHUNK_INDEX,
    METADATA_TOTAL_CHUNKS,
    SEEN_PDFS_FILE,
    BATCH_SIZE,
    MAX_PDF_SIZE_MB
)

warnings.filterwarnings("ignore")

# ---------------- UTILITIES ----------------

def calculate_file_hash(filepath):
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_pdf_size(filepath):
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    return size_mb <= MAX_PDF_SIZE_MB


def load_seen_pdfs():
    if SEEN_PDFS_FILE.exists():
        try:
            return pickle.loads(SEEN_PDFS_FILE.read_bytes())
        except Exception:
            return {}
    return {}


def save_seen_pdfs(seen):
    SEEN_PDFS_FILE.write_bytes(pickle.dumps(seen))


# ---------------- LOAD PDFs ----------------

def load_new_pdfs(seen):
    pdfs = []
    for ext in PDF_EXTENSIONS:
        pdfs.extend(DATA_DIR.glob(f"*{ext}"))

    docs, new = [], []

    for pdf in pdfs:
        if not check_pdf_size(pdf):
            continue

        h = calculate_file_hash(pdf)
        if seen.get(str(pdf)) == h:
            continue

        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for i, p in enumerate(pages):
            p.metadata.update({
                METADATA_SOURCE: pdf.name,
                METADATA_PAGE: i + 1,
                METADATA_PDF_HASH: h,
            })

        docs.extend(pages)
        new.append((str(pdf), h))

    return docs, new


# ---------------- SPLIT ----------------

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# ---------------- CHROMA ----------------

def update_chroma(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        db.add_documents(chunks[i:i + BATCH_SIZE])

    return db


# ---------------- MAIN ----------------

def main():
    seen = load_seen_pdfs()
    docs, new = load_new_pdfs(seen)

    if not docs:
        print("✅ No new PDFs to ingest")
        return

    chunks = split_docs(docs)

    for i, c in enumerate(chunks):
        c.metadata[METADATA_CHUNK_ID] = i + 1
        c.metadata[METADATA_CHUNK_INDEX] = i + 1
        c.metadata[METADATA_TOTAL_CHUNKS] = len(chunks)
        c.metadata[METADATA_ARTICLE] = "unknown"

    update_chroma(chunks)

    for path, h in new:
        seen[path] = h

    save_seen_pdfs(seen)
    print(f"✅ Ingested {len(new)} PDFs")


if __name__ == "__main__":
    main()
