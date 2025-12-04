import os
from dotenv import load_dotenv

load_dotenv()

# -------- API KEYS --------
HF_API_KEY = os.getenv("HF_API_KEY")

# -------- HF MODELS --------
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "google/gemma-2-2b-it")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "intfloat/e5-small")


# -------- CHUNKING --------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# -------- RAG RETRIEVAL --------
TOP_K = int(os.getenv("TOP_K", 4))

# -------- CHROMA DB --------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
