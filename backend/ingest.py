import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ----------------------------
# SETTINGS
# ----------------------------


# Change Hugging Face cache to D: drive
os.environ['HF_HOME'] = r'D:\huggingface_cache'


DATA_PATHS = [
    os.path.join("data", "consttution-2024.pdf"),
    os.path.join("data", "27 amedment.pdf"),
]

DB_DIR = os.path.join(os.path.dirname(__file__), "db")


# ----------------------------
# LOAD PDF
# ----------------------------
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


# ----------------------------
# MAIN INGEST PROCESS
# ----------------------------
def main():

    # Check file existence
    for path in DATA_PATHS:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ PDF not found: {path}")

    print("📄 Loading PDFs...")
    raw_docs = []

    for path in DATA_PATHS:
        raw_docs.extend(load_pdf(path))

    print(f"✅ Loaded {len(raw_docs)} total pages")

    # ----------------------------
    # TEXT SPLITTING
    # ----------------------------
    print("🔪 Splitting text into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=75,
        length_function=len,
    )

    docs = splitter.split_documents(raw_docs)
    print(f"📌 Created {len(docs)} chunks")

    # Add metadata
    for i, d in enumerate(docs):
        d.metadata["chunk_id"] = i
        d.metadata["source"] = d.metadata.get("source", "constitution")

    # ----------------------------
    # VECTOR DB CREATION
    # ----------------------------
    print("🧠 Embedding and storing in ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"✅ Vectorstore written to: {DB_DIR}")


if __name__ == "__main__":
    main()
