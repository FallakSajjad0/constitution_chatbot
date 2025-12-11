# ingest.py - Optimized for constitutional/legal PDFs
import os
import logging
import hashlib
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# PDF processing
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# Local embeddings
from embeddings_local import LocalEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstitutionalPDFIngestor:
    def __init__(self):
        self.data_path = "./data"
        self.chroma_path = "./chroma_db"
        self.collection_name = "constitutional_docs"
        self.embeddings = LocalEmbeddings()
        
        # Create directories if they don't exist
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        pages = []
        try:
            logger.info(f"📖 Extracting text from: {os.path.basename(pdf_path)}")
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                logger.info(f"   📄 Total pages: {total_pages}")
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # Clean the text
                        text = text.strip()
                        # Remove excessive whitespace
                        text = ' '.join(text.split())
                        pages.append((text, page_num + 1))
                        
                        # Log first page content for verification
                        if page_num == 0:
                            preview = text[:200]
                            logger.info(f"   📋 Page 1 preview: '{preview}...'")
                    
                    # Progress logging
                    if (page_num + 1) % 50 == 0:
                        logger.info(f"   ⏳ Processed {page_num + 1}/{total_pages} pages...")
                
                logger.info(f"✅ Extracted {len(pages)} non-empty pages")
                return pages
                
        except Exception as e:
            logger.error(f"❌ Failed to extract from {pdf_path}: {str(e)}")
            return []
    
    def intelligent_chunking(self, pages: List[Tuple[str, int]], source_file: str) -> List[Dict]:
        """Specialized chunking for constitutional/legal documents"""
        documents = []
        
        # First pass: Try to preserve article/section boundaries
        for text, page_num in pages:
            # Split by common legal document markers
            lines = text.split('\n')
            current_chunk = []
            current_chunk_start_page = page_num
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new article/section
                is_new_section = any(marker in line.upper() for marker in [
                    'ARTICLE', 'SECTION', 'CHAPTER', 'PART',
                    'CLAUSE', 'SUBSECTION', 'SCHEDULE'
                ])
                
                # Check if line contains article numbers like "25A", "25-A", "25 A"
                if any(marker in line for marker in ['Article', 'article']):
                    # Extract potential article number
                    import re
                    article_match = re.search(r'Article\s+([0-9]+[A-Z]*)', line, re.IGNORECASE)
                    if article_match:
                        logger.debug(f"📌 Found article reference: {article_match.group(0)} on page {page_num}")
                
                # If we have content and hit a new section, save current chunk
                if current_chunk and (is_new_section or len(' '.join(current_chunk)) > 800):
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 50:  # Only save meaningful chunks
                        documents.append({
                            "text": chunk_text,
                            "metadata": {
                                "source": source_file,
                                "page": current_chunk_start_page,
                                "type": "legal_chunk",
                                "chunk_hash": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                            }
                        })
                    
                    # Start new chunk
                    current_chunk = [line]
                    current_chunk_start_page = page_num
                else:
                    current_chunk.append(line)
            
            # Don't forget the last chunk from this page
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) > 50:
                    documents.append({
                        "text": chunk_text,
                        "metadata": {
                            "source": source_file,
                            "page": current_chunk_start_page,
                            "type": "legal_chunk",
                            "chunk_hash": hashlib.md5(chunk_text.encode()).hexdigest()[:8]
                        }
                    })
        
        # Second pass: If no good chunks found, use standard text splitter
        if len(documents) < 5:
            logger.warning("⚠️ Few chunks found, using standard text splitter...")
            all_text = ' '.join([text for text, _ in pages])
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
            
            chunks = text_splitter.split_text(all_text)
            documents = []
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 50:
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": source_file,
                            "page": "N/A",
                            "type": "standard_chunk",
                            "chunk_id": i
                        }
                    })
        
        logger.info(f"📊 Created {len(documents)} chunks from {source_file}")
        
        # Log sample chunks for verification
        logger.info("📋 Sample chunks:")
        for i, doc in enumerate(documents[:3]):
            sample = doc['text'][:150]
            page = doc['metadata']['page']
            logger.info(f"   Chunk {i+1} (Page {page}): '{sample}...'")
        
        # Check if we have any article references
        article_count = 0
        for doc in documents:
            if 'article' in doc['text'].lower() or 'Article' in doc['text']:
                article_count += 1
        
        logger.info(f"📑 Found {article_count} chunks with article references")
        
        return documents
    
    def create_or_update_chromadb(self, documents: List[Dict]):
        """Create or update ChromaDB with documents"""
        try:
            # Extract texts and metadatas
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # Check if ChromaDB exists
            if os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
                logger.info("📂 Loading existing ChromaDB...")
                
                # Get existing collection count
                try:
                    client = chromadb.PersistentClient(path=self.chroma_path)
                    collection = client.get_collection(self.collection_name)
                    existing_count = collection.count()
                    logger.info(f"📚 Existing collection has {existing_count} documents")
                except:
                    existing_count = 0
                
                # Add to existing ChromaDB
                vector_store = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
                
                # Add new documents
                vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                
                logger.info(f"✅ Added {len(documents)} new chunks to existing DB")
                
            else:
                logger.info("🆕 Creating new ChromaDB...")
                
                # Create new ChromaDB
                vector_store = Chroma.from_texts(
                    texts=texts,
                    metadatas=metadatas,
                    embedding=self.embeddings,
                    persist_directory=self.chroma_path,
                    collection_name=self.collection_name
                )
                
                logger.info(f"✅ Created new DB with {len(documents)} chunks")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update ChromaDB: {str(e)}")
            return False
    
    def ingest_single_pdf(self, pdf_filename: str) -> bool:
        """Ingest a single PDF file"""
        pdf_path = os.path.join(self.data_path, pdf_filename)
        
        if not os.path.exists(pdf_path):
            logger.error(f"❌ File not found: {pdf_path}")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 PROCESSING: {pdf_filename}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract text
            pages = self.extract_text_from_pdf(pdf_path)
            if not pages:
                logger.error(f"❌ No text extracted from {pdf_filename}")
                return False
            
            # Create chunks
            documents = self.intelligent_chunking(pages, pdf_filename)
            if not documents:
                logger.error(f"❌ No chunks created from {pdf_filename}")
                return False
            
            # Add to ChromaDB
            success = self.create_or_update_chromadb(documents)
            
            if success:
                logger.info(f"✅ Successfully ingested {pdf_filename}")
            else:
                logger.error(f"❌ Failed to add {pdf_filename} to ChromaDB")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_filename}: {str(e)}")
            return False
    
    def ingest_all_pdfs(self):
        """Ingest all PDFs in the data directory"""
        try:
            # Get all PDF files
            pdf_files = []
            for file in os.listdir(self.data_path):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(file)
            
            if not pdf_files:
                logger.error(f"❌ No PDF files found in {self.data_path}")
                return False
            
            logger.info(f"📚 Found {len(pdf_files)} PDF files:")
            for pdf in pdf_files:
                logger.info(f"   📄 {pdf}")
            
            # Start with constitution file first (likely contains Article 25A)
            constitution_files = [f for f in pdf_files if 'constitution' in f.lower()]
            other_files = [f for f in pdf_files if f not in constitution_files]
            
            # Process constitution files first
            files_to_process = constitution_files + other_files
            
            logger.info(f"\n🎯 Processing {len(files_to_process)} files...")
            
            success_count = 0
            failed_files = []
            
            for pdf_file in files_to_process:
                if self.ingest_single_pdf(pdf_file):
                    success_count += 1
                else:
                    failed_files.append(pdf_file)
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info("📊 INGESTION SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"✅ Successfully ingested: {success_count}/{len(files_to_process)} files")
            
            if failed_files:
                logger.info("❌ Failed files:")
                for failed in failed_files:
                    logger.info(f"   - {failed}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest PDFs: {str(e)}")
            return False
    
    def verify_ingestion(self):
        """Verify that content was properly ingested"""
        try:
            logger.info(f"\n🔍 VERIFYING CHROMADB")
            logger.info(f"{'='*60}")
            
            if not os.path.exists(self.chroma_path):
                logger.error("❌ ChromaDB not found")
                return False
            
            # Load ChromaDB
            vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test search for Article 25A
            test_queries = [
                "Article 25A",
                "25A",
                "Right to education",
                "free and compulsory education",
                "Article 25 A",
                "Article 25-A"
            ]
            
            for query in test_queries:
                logger.info(f"\n🔎 Searching for: '{query}'")
                try:
                    results = vector_store.similarity_search(query, k=3)
                    
                    if results:
                        logger.info(f"✅ Found {len(results)} results")
                        for i, doc in enumerate(results[:2]):
                            content_preview = doc.page_content[:200].replace('\n', ' ')
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'N/A')
                            logger.info(f"   Result {i+1} [{source}, Page {page}]:")
                            logger.info(f"      '{content_preview}...'")
                    else:
                        logger.info(f"❌ No results found")
                        
                except Exception as e:
                    logger.error(f"❌ Search error for '{query}': {str(e)}")
            
            # Get total count
            try:
                client = chromadb.PersistentClient(path=self.chroma_path)
                collection = client.get_collection(self.collection_name)
                count = collection.count()
                logger.info(f"\n📊 Total documents in ChromaDB: {count}")
                
                # List unique sources
                results = collection.get(include=["metadatas"])
                sources = set()
                if results and "metadatas" in results:
                    for metadata in results["metadatas"]:
                        if metadata and "source" in metadata:
                            sources.add(metadata["source"])
                
                logger.info("📁 Sources in database:")
                for source in sorted(sources):
                    logger.info(f"   - {source}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not get collection info: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            return False


def main():
    """Main ingestion process"""
    print("\n" + "="*60)
    print("📚 CONSTITUTIONAL DOCUMENTS INGESTOR")
    print("="*60)
    
    ingestor = ConstitutionalPDFIngestor()
    
    # Step 1: Ingest all PDFs
    print("\n🔄 Step 1: Ingesting all PDFs...")
    success = ingestor.ingest_all_pdfs()
    
    if not success:
        print("❌ Ingestion failed!")
        return
    
    # Step 2: Verify
    print("\n🔄 Step 2: Verifying ingestion...")
    ingestor.verify_ingestion()
    
    print("\n" + "="*60)
    print("✅ INGESTION COMPLETE")
    print("="*60)
    print("\n💡 Now test with:")
    print("python -c \"from rag_chain import answer_question; print(answer_question('What does Article 25A say?'))\"")


if __name__ == "__main__":
    main()