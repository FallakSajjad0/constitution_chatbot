# backend/ingest.py
import os
import pickle
import hashlib
import warnings
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import constants
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
    INGESTION_LOG_FILE,
    BATCH_SIZE,
    MAX_PDF_SIZE_MB
)

# Suppress warnings
warnings.filterwarnings('ignore')


# ================================
# UTILITY FUNCTIONS
# ================================
def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_pdf_size(filepath):
    """Check if PDF file size is within limits"""
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        print(f"⚠ PDF {Path(filepath).name} is {size_mb:.1f}MB (exceeds {MAX_PDF_SIZE_MB}MB limit)")
        return False
    return True


def clean_text(text):
    """Clean text content"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove special characters that might cause issues
    text = text.replace('\x00', '')  # Remove null characters
    return text


# ================================
# LOAD OR CREATE TRACKER FOR NEW PDFs
# ================================
def load_seen_pdfs():
    """Load the dictionary of processed PDF files with their hashes"""
    if os.path.exists(SEEN_PDFS_FILE):
        try:
            with open(SEEN_PDFS_FILE, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError):
            print("⚠ Could not load seen PDFs file, starting fresh...")
            return {}
    return {}


def save_seen_pdfs(seen):
    """Save the dictionary of processed PDF files with their hashes"""
    try:
        with open(SEEN_PDFS_FILE, "wb") as f:
            pickle.dump(seen, f)
        return True
    except Exception as e:
        print(f"❌ Error saving seen PDFs: {e}")
        return False


# ================================
# LOAD PDF DOCUMENTS
# ================================
def load_new_pdfs(pdf_folder, seen_pdfs):
    """Load new PDF files that haven't been processed yet"""
    # Find all PDF files
    pdf_paths = []
    for ext in PDF_EXTENSIONS:
        pdf_paths.extend(Path(pdf_folder).glob(f"*{ext}"))
    
    if not pdf_paths:
        print("📘 No PDF files found in data directory")
        return [], []
    
    new_pdfs = []
    
    for pdf in pdf_paths:
        pdf_path = str(pdf)
        pdf_name = pdf.name
        
        # Check file size
        if not check_pdf_size(pdf_path):
            continue
            
        # Calculate file hash
        file_hash = calculate_file_hash(pdf_path)
        
        # Check if PDF is new or modified
        if pdf_path in seen_pdfs:
            if seen_pdfs[pdf_path] == file_hash:
                # Already processed and unchanged
                continue
            else:
                print(f"📄 Modified PDF: {pdf_name}")
                new_pdfs.append((pdf_path, file_hash))
        else:
            print(f"🆕 New PDF: {pdf_name}")
            new_pdfs.append((pdf_path, file_hash))
    
    if not new_pdfs:
        print("📘 No new or modified PDFs found")
        return [], []
    
    print(f"🔍 Found {len(new_pdfs)} new/modified PDFs")
    
    all_docs = []
    
    for pdf_path, pdf_hash in new_pdfs:
        try:
            pdf_name = Path(pdf_path).name
            print(f"\n📄 Loading PDF: {pdf_name}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            print(f"   ✅ Loaded {len(pages)} pages")
            
            # Add enhanced metadata
            for i, page in enumerate(pages):
                page.metadata[METADATA_SOURCE] = pdf_name
                page.metadata[METADATA_PAGE] = i + 1
                page.metadata["file_path"] = pdf_path
                page.metadata["total_pages"] = len(pages)
                page.metadata[METADATA_PDF_HASH] = pdf_hash
            
            all_docs.extend(pages)
            
        except Exception as e:
            print(f"   ❌ Failed to load {Path(pdf_path).name}: {str(e)}")
            continue
    
    return all_docs, new_pdfs


# ================================
# SPLITTING DOCUMENTS
# ================================
def split_docs(docs):
    """Split documents into chunks for vector storage"""
    print("\n🔪 Splitting documents into chunks...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    print(f"📦 Total chunks created: {len(chunks)}")
    
    return chunks


# ================================
# EXTRACT ARTICLE NUMBERS FROM TEXT
# ================================
def extract_article_number_from_text(text):
    """Try to extract article number from document text"""
    import re
    
    patterns = [
        r'article\s+(\d+[A-Z]*)',
        r'art\.\s*(\d+[A-Z]*)',
        r'article\s+(\d+)\s*\([a-zA-Z]+\)',
        r'art\.\s*(\d+)\s*\([a-zA-Z]+\)',
        r'\b(\d+[A-Z])\b',
        r'article\s+(\d+)[\s\.]',
        r'art\.\s*(\d+)[\s\.]'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            article_num = match.group(1).upper()
            article_num = re.sub(r'\s+', '', article_num)
            return article_num
    
    return None


# ================================
# PROCESS AND ADD METADATA TO CHUNKS
# ================================
def enhance_chunk_metadata(chunks):
    """Add article number detection and other metadata to chunks"""
    print("🔍 Enhancing chunk metadata...")
    
    article_counts = {}
    
    for i, chunk in enumerate(chunks):
        # Add basic metadata
        chunk.metadata[METADATA_CHUNK_ID] = i + 1
        chunk.metadata[METADATA_CHUNK_INDEX] = i + 1
        chunk.metadata[METADATA_TOTAL_CHUNKS] = len(chunks)
        
        # Clean text
        chunk.page_content = clean_text(chunk.page_content)
        
        # Extract article number
        text = chunk.page_content
        article_num = extract_article_number_from_text(text[:1000])  # Check first 1000 chars
        
        if article_num:
            chunk.metadata[METADATA_ARTICLE] = article_num
            article_counts[article_num] = article_counts.get(article_num, 0) + 1
        else:
            chunk.metadata[METADATA_ARTICLE] = "unknown"
    
    # Log article statistics
    if article_counts:
        print(f"📊 Detected {len(article_counts)} unique articles")
        top_articles = sorted(article_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for article, count in top_articles:
            print(f"   • Article {article}: {count} chunks")
    
    articles_found = sum(1 for c in chunks if c.metadata.get(METADATA_ARTICLE) != "unknown")
    print(f"📊 Chunks with article numbers: {articles_found}/{len(chunks)}")
    
    return chunks


# ================================
# CREATE OR UPDATE CHROMA DATABASE
# ================================
def update_chroma_db(chunks, collection_name=COLLECTION_NAME):
    """Update or create Chroma vector database"""
    print("\n🧠 Loading embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {str(e)}")
        return None
    
    print("📚 Connecting to Chroma DB...")
    
    try:
        # Check if Chroma DB exists
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            print("📂 Loading existing Chroma DB...")
            # Load existing vector store
            db = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings,
                collection_name=collection_name
            )
            
            # Get existing document count
            try:
                existing_count = db._collection.count()
                print(f"📊 Existing DB has {existing_count} document chunks")
            except:
                print("📊 Existing DB loaded")
        else:
            print("🆕 Creating new Chroma DB...")
            # Create new vector store
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR),
                collection_name=collection_name
            )
            print("✅ New Chroma DB created")
        
        print(f"\n💾 Adding {len(chunks)} chunks to vector database...")
        
        # Add documents in batches
        total_added = 0
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            db.add_documents(batch)
            total_added += len(batch)
            if len(batch) < BATCH_SIZE:
                print(f"   Added batch {i//BATCH_SIZE + 1}: {len(batch)} documents")
            else:
                print(f"   Added batch {i//BATCH_SIZE + 1}: {BATCH_SIZE} documents")
        
        print(f"✅ Added {total_added} new chunks")
        
        # Get final count
        try:
            final_count = db._collection.count()
            print(f"📊 Total documents in collection: {final_count}")
        except:
            pass
        
        return db
        
    except Exception as e:
        print(f"❌ Error during Chroma DB operation: {str(e)}")
        return None


# ================================
# RESET DATABASE (Optional function)
# ================================
def reset_database():
    """Reset the entire database - use with caution!"""
    print("⚠  Resetting database...")
    
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("✅ Chroma DB deleted")
    
    if SEEN_PDFS_FILE.exists():
        os.remove(SEEN_PDFS_FILE)
        print("✅ Seen PDFs file deleted")
    
    # Remove log file
    if INGESTION_LOG_FILE.exists():
        os.remove(INGESTION_LOG_FILE)
        print("✅ Log file deleted")
    
    print("✅ Database reset complete")


# ================================
# MAIN INGEST FUNCTION
# ================================
def main():
    print("\n" + "="*60)
    print("🚀 CONSTITUTION DOCUMENT INGESTION PIPELINE")
    print("="*60 + "\n")
    
    print(f"📁 Data directory: {DATA_DIR}")
    print(f"📁 Chroma directory: {CHROMA_DIR}")
    print(f"📊 Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print(f"🏷️ Collection: {COLLECTION_NAME}")
    
    # Check if data folder exists
    if not DATA_DIR.exists():
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        print(f"💡 Create the directory and put your PDFs in it.")
        return
    
    # Check if there are any PDFs in the data folder
    pdf_files = []
    for ext in PDF_EXTENSIONS:
        pdf_files.extend(list(DATA_DIR.glob(f"*{ext}")))
    
    if not pdf_files:
        print(f"\n❌ No PDF files found in '{DATA_DIR}' folder.")
        print(f"💡 Put your PDF files in the data folder.")
        return
    
    # Load seen PDFs
    seen_pdfs = load_seen_pdfs()
    print(f"\n📋 Currently tracking {len(seen_pdfs)} PDFs")
    
    # Load new PDFs
    docs, new_pdfs_info = load_new_pdfs(DATA_DIR, seen_pdfs)
    
    if not docs:
        print("\n✅ Ingestion completed - no new or modified PDFs")
        
        # Show database status
        if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
            print("\n📊 Database Status:")
            print(f"   PDFs in database: {len(seen_pdfs)}")
            print(f"   Database location: {CHROMA_DIR}")
            print(f"   Collection name: {COLLECTION_NAME}")
        
        print("\n💡 To add new PDFs:")
        print("   1. Add new PDF files to the data folder")
        print("   2. Run this script again")
        return
    
    print(f"\n🔄 Processing {len(new_pdfs_info)} new/modified PDFs...")
    
    # Split into chunks
    chunks = split_docs(docs)
    
    # Enhance metadata
    chunks = enhance_chunk_metadata(chunks)
    
    # Update Chroma database
    db = update_chroma_db(chunks)
    
    if db is None:
        print("❌ Failed to update database")
        return
    
    # Update seen PDFs with new hashes
    for pdf_path, pdf_hash in new_pdfs_info:
        seen_pdfs[pdf_path] = pdf_hash
    
    if save_seen_pdfs(seen_pdfs):
        print(f"✅ Updated tracking for {len(new_pdfs_info)} PDFs")
    
    # Summary
    print("\n" + "="*60)
    print("📊 INGESTION SUMMARY")
    print("="*60)
    print(f"• PDFs processed: {len(new_pdfs_info)}")
    print(f"• Pages processed: {len(docs)}")
    print(f"• Chunks created: {len(chunks)}")
    print(f"• Articles detected: {sum(1 for c in chunks if c.metadata.get(METADATA_ARTICLE) != 'unknown')}")
    print(f"• Collection: {COLLECTION_NAME}")
    print(f"• Database location: {CHROMA_DIR}")
    print("="*60)
    
    print("\n🎉 Ingestion completed successfully!")
    print("🤖 Your chatbot can now answer from the updated knowledge base!")


if __name__ == "__main__":
    import sys
    
    # Check for reset command
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        confirm = input("⚠ Are you sure you want to reset the database? This will delete ALL data! (y/n): ")
        if confirm.lower() == 'y':
            reset_database()
            print("\n✅ Database reset. Now you can run 'python ingest.py' to start fresh.")
        else:
            print("❌ Reset cancelled.")
    else:
        main()