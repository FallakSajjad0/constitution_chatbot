# extract_real_text.py
import chromadb
import re

def extract_all_articles():
    """Extract actual article text from PDFs"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("constitutional_docs")
    
    print("🔍 EXTRACTING ACTUAL ARTICLE TEXT")
    print("="*60)
    
    all_docs = collection.get()
    
    articles_found = {}
    
    for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
        # Skip table of contents
        doc_lower = doc[:300].lower()
        if any(term in doc_lower for term in ["contents", "table of", "article s page"]):
            continue
        
        # Find all articles in this document
        article_patterns = [
            r'Article\s+(\d+[A-Z]*)[\.\:\-]\s+(.*?)(?=Article\s+\d+|$)',
            r'Art\.\s+(\d+[A-Z]*)[\.\:\-]\s+(.*?)(?=Art\.\s+\d+|$)',
            r'(\d+[A-Z]*)\.\s+(.*?)(?=\d+[A-Z]*\.\s+|$)',
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, doc, re.IGNORECASE | re.DOTALL)
            for match in matches:
                article_num = match[0].upper()
                article_text = match[1].strip()
                
                # Clean text
                article_text = re.sub(r'\s+', ' ', article_text)
                article_text = article_text.strip()
                
                # Only keep if it's meaningful text (not just TOC entry)
                if len(article_text) > 30 and not re.match(r'^\d+\s*$', article_text):
                    if article_num not in articles_found:
                        articles_found[article_num] = []
                    
                    source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
                    page = metadata.get('page', 'N/A')
                    
                    articles_found[article_num].append({
                        'text': article_text,
                        'source': source,
                        'page': page,
                        'doc_index': i
                    })
    
    # Display results
    if articles_found:
        print(f"✅ Found {len(articles_found)} articles with actual text")
        print()
        
        # Sort by article number
        sorted_articles = sorted(articles_found.items(), 
                               key=lambda x: (int(re.match(r'(\d+)', x[0]).group(1)) 
                                             if re.match(r'(\d+)', x[0]) else 9999, x[0]))
        
        for article_num, texts in sorted_articles:
            print(f"📜 **ARTICLE {article_num}**")
            print(f"   Found in {len(texts)} locations")
            
            # Show the best (longest) text
            texts.sort(key=lambda x: len(x['text']), reverse=True)
            best_text = texts[0]
            
            print(f"   Source: {best_text['source']}")
            if best_text['page'] != 'N/A':
                print(f"   Page: {best_text['page']}")
            
            # Show first 200 chars of text
            preview = best_text['text'][:200]
            if len(best_text['text']) > 200:
                preview += "..."
            print(f"   Text: {preview}")
            print()
    else:
        print("❌ No actual article text found!")
        print("\n⚠️ Your PDFs appear to contain only table of contents.")
        print("   The articles are listed but their text content is not searchable.")

def search_specific_term(term: str):
    """Search for specific term in non-TOC content"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("constitutional_docs")
    
    print(f"\n🔍 SEARCHING FOR: '{term}'")
    print("="*60)
    
    # Get search results
    results = collection.query(
        query_texts=[term],
        n_results=20
    )
    
    non_toc_results = []
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        # Check if this is NOT table of contents
        doc_lower = doc[:300].lower()
        is_toc = any(toc_term in doc_lower for toc_term in 
                    ["contents", "table of", "article s page", "articles page"])
        
        if not is_toc:
            source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
            page = metadata.get('page', 'N/A')
            
            # Clean and preview
            preview = doc[:200].strip()
            preview = re.sub(r'\s+', ' ', preview)
            
            non_toc_results.append({
                'text': preview,
                'source': source,
                'page': page,
                'score': results['distances'][0][i] if results['distances'] else 0
            })
    
    if non_toc_results:
        print(f"✅ Found {len(non_toc_results)} non-TOC results")
        print()
        
        for i, result in enumerate(non_toc_results[:10]):
            print(f"{i+1}. **{result['source']}**")
            if result['page'] != 'N/A':
                print(f"   Page: {result['page']}")
            print(f"   Text: {result['text']}...")
            print()
    else:
        print(f"❌ No non-TOC content found for '{term}'")
        print("\n⚠️ All results were from table of contents.")

if __name__ == "__main__":
    print("🔍 PDF CONTENT ANALYZER")
    print("="*60)
    
    # Check what real content exists
    extract_all_articles()
    
    # Test specific searches
    print("\n" + "="*60)
    print("🧪 TESTING SPECIFIC SEARCHES")
    print("="*60)
    
    test_terms = ["shall", "right", "power", "duty", "freedom"]
    
    for term in test_terms:
        search_specific_term(term)
        print("-" * 40)