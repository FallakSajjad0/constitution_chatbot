import os
import sys
import chromadb
from pathlib import Path

print("🔍 ChromaDB Diagnosis Script")
print("=" * 50)

# Check if chroma_db directory exists
chroma_path = "./chroma_db"
if not os.path.exists(chroma_path):
    print(f"❌ ChromaDB directory not found at: {chroma_path}")
    print("💡 Solution: Run ingest.py to create the database")
    sys.exit(1)

print(f"✅ ChromaDB directory exists at: {chroma_path}")

# List contents of chroma_db
print("\n📁 Contents of chroma_db directory:")
for item in os.listdir(chroma_path):
    item_path = os.path.join(chroma_path, item)
    if os.path.isdir(item_path):
        print(f"  📂 {item}/")
        # List subitems
        try:
            subitems = os.listdir(item_path)[:5]  # Show first 5 items
            for subitem in subitems:
                print(f"    📄 {subitem}")
        except:
            pass
    else:
        print(f"  📄 {item}")

# Try to connect to ChromaDB
print("\n🔗 Attempting to connect to ChromaDB...")
try:
    client = chromadb.PersistentClient(path=chroma_path)
    print("✅ Connected to ChromaDB successfully")
    
    # List collections
    print("\n📚 Collections in database:")
    collections = client.list_collections()
    if not collections:
        print("❌ No collections found!")
    else:
        for collection in collections:
            print(f"  📖 Collection: {collection.name}")
            print(f"    Count: {collection.count()} documents")
            print(f"    Metadata: {collection.metadata}")
            
            # Show sample documents
            try:
                sample = collection.peek(limit=3)
                if sample and 'documents' in sample and sample['documents']:
                    print(f"    Sample documents: {len(sample['documents'])}")
                    for i, doc in enumerate(sample['documents'][:2]):
                        print(f"      {i+1}. {doc[:100]}...")
            except Exception as e:
                print(f"    Error peeking: {e}")
    
except Exception as e:
    print(f"❌ Failed to connect to ChromaDB: {str(e)}")

print("\n" + "=" * 50)