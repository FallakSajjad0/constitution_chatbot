# backend/ingest.py (COMPLETE WORKING VERSION)
import os
import re
import shutil

# ✅ CORRECT IMPORT
from pypdf import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ----------------------------
# SETTINGS
# ----------------------------
os.environ['HF_HOME'] = r'D:\huggingface_cache'
DATA_PATHS = [
    os.path.join("..", "data", "consttution-2024.pdf"),
    os.path.join("..", "data", "27 amedment.pdf"),
]
DB_DIR = os.path.join("..", "db")

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 150

# ----------------------------
# PDF PROCESSING
# ----------------------------
def extract_articles_from_text(text):
    """Split text by articles"""
    article_pattern = r'(?:^|\n)\s*(ARTICLE|Article)\s+(\d+[A-Za-z]?)\s*[.:\-]?\s*'
    articles = []
    matches = list(re.finditer(article_pattern, text, re.MULTILINE | re.IGNORECASE))

    if not matches:
        return [(None, text)]

    for i, match in enumerate(matches):
        article_num = match.group(2)
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        article_text = text[start:end].strip()
        articles.append((article_num, article_text))

    return articles

def load_pdf_with_article_detection(path):
    """Load PDF using pypdf"""
    print(f"   📄 Loading: {os.path.basename(path)}")
    
    documents = []
    
    with open(path, 'rb') as file:
        reader = PdfReader(file)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            articles = extract_articles_from_text(text)
            
            for article_num, article_text in articles:
                doc = Document(
                    page_content=article_text,
                    metadata={
                        'source': os.path.basename(path),
                        'page': page_num + 1,
                        'article': article_num if article_num else 'general',
                    }
                )
                documents.append(doc)
    
    print(f"   ✅ Extracted {len(documents)} chunks")
    return documents

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    print("="*70)
    print("CONSTITUTION INGESTION")
    print("="*70)

    # Check files
    for path in DATA_PATHS:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            print(f"Please add PDF files to: {os.path.dirname(path)}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return

    print("\n📚 Step 1: Loading PDFs...")
    all_docs = []
    for path in DATA_PATHS:
        print(f"\n📖 Processing: {os.path.basename(path)}")
        docs = load_pdf_with_article_detection(path)
        all_docs.extend(docs)

    print(f"\n✅ Total documents: {len(all_docs)}")

    print(f"\n🔪 Step 2: Chunking (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
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

    print(f"✅ Created {len(final_docs)} chunks")

    # Add IDs
    for i, doc in enumerate(final_docs):
        doc.metadata["chunk_id"] = i

    # Delete old DB
    if os.path.exists(DB_DIR):
        print(f"\n🗑️  Step 3: Deleting old database...")
        shutil.rmtree(DB_DIR)

    # Create vector DB
    print(f"\n🧠 Step 4: Creating vector database...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = Chroma.from_documents(
        documents=final_docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # Verify
    count = vectorstore._collection.count()
    print(f"\n✅ Step 5: Verification")
    print(f"   📊 Total chunks in database: {count}")
    print(f"   📁 Database location: {DB_DIR}")

    # Test
    print(f"\n🔍 Step 6: Testing retrieval...")
    results = vectorstore.similarity_search("fundamental rights", k=2)
    print(f"   Test query 'fundamental rights' found {len(results)} results")

    print("\n" + "="*70)
    print("🎉 INGESTION COMPLETE!")
    print("="*70)
    print(f"Database: {DB_DIR}")
    print(f"Total chunks: {count}")

if __name__ == "__main__":
    main()