import os
import re
import shutil

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ----------------------------
# SETTINGS
# ----------------------------
os.environ['HF_HOME'] = r'D:\huggingface_cache'

DATA_DIR = os.path.join("..", "data")
DB_DIR = os.path.join("..", "db")

CHUNK_SIZE = 150        # Small chunks = better RAG accuracy
CHUNK_OVERLAP = 50      # Balanced overlap for context

# ----------------------------
# ARTICLE DETECTION
# ----------------------------
def extract_articles_from_text(text):
    pattern = r'(?:^|\n)\s*(ARTICLE|Article)\s+(\d+[A-Za-z]?)\s*[.:\-]?\s*'
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))

    if not matches:
        return [(None, text)]

    articles = []
    for i, match in enumerate(matches):
        article_num = match.group(2)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        articles.append((article_num, text[start:end].strip()))

    return articles

# ----------------------------
# PDF LOADER
# ----------------------------
def load_pdf_with_article_detection(path):
    print(f"\n📄 Loading: {os.path.basename(path)}")
    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"   ❌ Cannot open PDF: {e}")
        return [], 0, 0

    total_pages = len(reader.pages)
    total_chars = 0
    print(f"   📘 Total Pages: {total_pages}")

    documents = []
    for page_idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"   ⚠ Failed Page {page_idx+1}: {e}")
            continue

        if not text:
            continue

        total_chars += len(text)
        print(f"      • Page {page_idx+1} → {len(text)} characters")

        articles = extract_articles_from_text(text)
        for article_num, article_text in articles:
            documents.append(
                Document(
                    page_content=article_text,
                    metadata={
                        "source": os.path.basename(path),
                        "page": page_idx + 1,
                        "article": article_num if article_num else "general"
                    }
                )
            )

    print(f"   ✨ Extracted {len(documents)} article blocks")
    print(f"   🔢 Total Characters: {total_chars}")
    return documents, total_pages, total_chars

# ----------------------------
# MAIN INGESTION PIPELINE
# ----------------------------
def main():
    print("=" * 70)
    print("CONSTITUTION INGESTION — AUTO PDF LOADER")
    print("=" * 70)

    if not os.path.exists(DATA_DIR):
        print(f"❌ Missing directory: {DATA_DIR}")
        return

    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("❌ No PDF files found in ../data/")
        return

    print(f"📄 Found {len(pdf_files)} PDF files:")
    for f in pdf_files:
        print("   •", os.path.basename(f))

    all_docs = []
    total_pages_all = 0
    total_chars_all = 0

    # LOAD EACH PDF
    for path in pdf_files:
        docs, pages, chars = load_pdf_with_article_detection(path)
        total_pages_all += pages
        total_chars_all += chars
        all_docs.extend(docs)

    print("\n===============================================")
    print(f"📘 TOTAL PAGES IN ALL PDFs: {total_pages_all}")
    print(f"🔢 TOTAL CHARACTERS IN ALL PDFs: {total_chars_all}")
    print(f"📝 TOTAL TEXT BLOCKS EXTRACTED: {len(all_docs)}")
    print("===============================================")

    # ----------------------------
    # CHUNKING
    # ----------------------------
    print(f"\n🔪 Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    final_docs = []
    for doc in all_docs:
        if len(doc.page_content) <= CHUNK_SIZE:
            final_docs.append(doc)
        else:
            chunks = splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
                final_docs.append(chunk)

    print(f"✅ Total chunks created: {len(final_docs)}")

    # ADD CHUNK ID
    for i, doc in enumerate(final_docs):
        doc.metadata["chunk_id"] = i

    # ----------------------------
    # CREATE VECTORSTORE (CHROMA DB)
    # ----------------------------
    if os.path.exists(DB_DIR):
        print("\n🗑️ Removing old Chroma DB...")
        shutil.rmtree(DB_DIR)

    print("\n🧠 Creating Chroma vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        persist_directory=DB_DIR   # DB will be persisted automatically
    )

    count = vectorstore._collection.count()
    print("\n📊 VERIFICATION:")
    print(f"   ➤ Chunks saved: {count}")
    print(f"   ➤ DB saved to: {DB_DIR}")

    # TEST QUERY
    print("\n🔍 Test Query: 'fundamental rights'")
    test = vectorstore.similarity_search("fundamental rights", k=2)
    print(f"   ➤ Test results: {len(test)}")

    print("\n" + "=" * 70)
    print("🎉 INGESTION COMPLETE — DATABASE READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
