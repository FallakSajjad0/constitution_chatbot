# # # test_simple.py
# # import sys
# # import os
# # sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # print("🧪 Testing RAG with existing ChromaDB...")

# # try:
# #     # Test embeddings first
# #     from embeddings_hf import HuggingFaceEmbeddings
# #     print("1. Testing embeddings...")
# #     emb = HuggingFaceEmbeddings()
# #     test_embed = emb.embed_query("test query")
# #     print(f"   ✅ Embeddings work. Dimension: {len(test_embed)}")
# # except Exception as e:
# #     print(f"   ❌ Embeddings failed: {str(e)}")
# #     exit(1)

# # try:
# #     # Test RAG
# #     from rag_chain import RAGSystem
# #     print("\n2. Initializing RAG system...")
# #     rag = RAGSystem()
# #     rag.initialize()
# #     print("   ✅ RAG initialized")
    
# #     # Test questions
# #     print("\n3. Testing questions:")
    
# #     test_questions = [
# #         "Hello!",
# #         "What is the National Assembly of Pakistan?",
# #         "Who are the members?",
# #         "What does article 25A say?"
# #     ]
    
# #     for question in test_questions:
# #         print(f"\n   🤔 Question: {question}")
# #         print(f"   {'─' * 50}")
# #         answer = rag.answer_question(question)
# #         print(f"   🤖 Answer: {answer[:200]}...")
# #         print(f"   {'─' * 50}")
        
# # except Exception as e:
# #     print(f"   ❌ RAG failed: {str(e)}")
# #     import traceback
# #     traceback.print_exc()

# # test_hf_api.py


# # import os
# # import requests
# # from dotenv import load_dotenv

# # load_dotenv()

# # def test_hf_api():
# #     api_key = os.getenv("HUGGINGFACE_API_KEY")
    
# #     if not api_key:
# #         print("❌ No API key in .env")
# #         return
    
# #     print("🧪 Testing HuggingFace API endpoints...")
    
# #     # Test different endpoints
# #     endpoints = [
# #         {
# #             "name": "Router API",
# #             "url": "https://router.huggingface.co/hf-inference/models",
# #             "payload": {
# #                 "inputs": ["test embedding"],
# #                 "model": "sentence-transformers/all-mpnet-base-v2",
# #                 "task": "feature-extraction"
# #             }
# #         },
# #         {
# #             "name": "Pipeline endpoint",
# #             "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2",
# #             "payload": {"inputs": ["test"]}
# #         },
# #         {
# #             "name": "MiniLM endpoint (fallback)",
# #             "url": "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
# #             "payload": {"inputs": ["test"]}
# #         }
# #     ]
    
# #     headers = {
# #         "Authorization": f"Bearer {api_key}",
# #         "Content-Type": "application/json"
# #     }
    
# #     for endpoint in endpoints:
# #         print(f"\n🔗 Testing {endpoint['name']}...")
        
# #         try:
# #             response = requests.post(
# #                 endpoint['url'],
# #                 json=endpoint['payload'],
# #                 headers=headers,
# #                 timeout=15
# #             )
            
# #             print(f"   Status: {response.status_code}")
            
# #             if response.status_code == 200:
# #                 print("   ✅ Success!")
# #                 result = response.json()
# #                 if isinstance(result, list):
# #                     print(f"   📏 Embedding length: {len(result[0])}")
# #             elif response.status_code == 503:
# #                 print("   ⏳ Model loading (normal)")
# #             elif response.status_code == 401:
# #                 print("   ❌ Invalid API key")
# #             elif response.status_code == 403:
# #                 print("   ❌ Insufficient permissions")
# #             else:
# #                 print(f"   📝 Response: {response.text[:200]}")
                
# #         except Exception as e:
# #             print(f"   ❌ Error: {str(e)}")

# # if __name__ == "__main__":
# #     test_hf_api()



# # test_query.py
# from rag_chain import RAGSystem
# import time

# print("🧪 Testing RAG with document queries...")

# rag = RAGSystem()
# rag.initialize()

# # Test questions about your documents
# test_questions = [
#     "What is Article 25A?",
#     "What does the constitution say about education?",
#     "Who are the members of parliament?",
#     "What is the National Assembly?",
#     "Summarize the main points about fundamental rights"
# ]

# for question in test_questions:
#     print(f"\n{'='*60}")
#     print(f"🤔 Question: {question}")
#     print(f"{'='*60}")
    
#     start_time = time.time()
#     answer = rag.answer_question(question)
#     elapsed = time.time() - start_time
    
#     print(f"\n🤖 Answer ({elapsed:.2f}s):")
#     print(f"{answer}")
#     print(f"\n{'='*60}")




# # # test_groq_fixed.py
# # from llm_online import OnlineLLM

# # print("🧪 Testing fixed Groq LLM...")

# # try:
# #     llm = OnlineLLM()
    
# #     # Simple test
# #     response = llm.invoke("Hello! Can you say 'test successful'?")
# #     print(f"✅ Response: {response}")
    
# #     # Longer test
# #     test_prompt = "Based on this context: Article 25A is about education rights. What is Article 25A?"
# #     response = llm.invoke(test_prompt)
# #     print(f"\n✅ RAG-style response test: {response[:100]}...")
    
# # except Exception as e:
# #     print(f"❌ Error: {str(e)}")

# check_articles.py - Diagnostic tool
import chromadb
import re

def check_specific_article(article_num: str):
    """Check if specific article exists and show its text"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("constitutional_docs")
    
    print(f"\n🔍 CHECKING ARTICLE {article_num}")
    print("="*60)
    
    all_docs = collection.get()
    
    found_articles = []
    
    for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
        # Look for the article
        if (f"Article {article_num}" in doc or 
            f"Art. {article_num}" in doc or 
            f"{article_num}." in doc):
            
            source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
            page = metadata.get('page', 'N/A')
            
            # Extract the article text
            # Look for pattern: Article XX. [text] next Article or end
            pattern = rf"(Article\s+{article_num}[\.\:\-]\s*.*?)(?=Article\s+\d+|$)"
            matches = re.findall(pattern, doc, re.IGNORECASE | re.DOTALL)
            
            if matches:
                article_text = matches[0]
                found_articles.append({
                    'text': article_text,
                    'source': source,
                    'page': page,
                    'doc_index': i
                })
            else:
                # Just show context
                lines = doc.split('\n')
                for j, line in enumerate(lines):
                    if article_num in line or f"Article {article_num}" in line:
                        start = max(0, j-1)
                        end = min(len(lines), j+3)
                        context = '\n'.join(lines[start:end])
                        found_articles.append({
                            'text': context,
                            'source': source,
                            'page': page,
                            'doc_index': i,
                            'type': 'context'
                        })
                        break
    
    if found_articles:
        print(f"✅ Found {len(found_articles)} instances of Article {article_num}")
        print()
        
        # Show the best match (longest text)
        found_articles.sort(key=lambda x: len(x['text']), reverse=True)
        best_match = found_articles[0]
        
        print("📜 **BEST MATCH:**")
        print(f"Source: {best_match['source']}")
        if best_match['page'] != 'N/A':
            print(f"Page: {best_match['page']}")
        print()
        
        # Clean and display text
        text = best_match['text']
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        print("**Text:**")
        print(text[:800])
        if len(text) > 800:
            print("... [truncated]")
        
        # Show other matches
        if len(found_articles) > 1:
            print(f"\n📚 Also found in {len(found_articles)-1} other locations:")
            for i, article in enumerate(found_articles[1:4], 2):
                print(f"{i}. {article['source']} - {article['text'][:100]}...")
        
    else:
        print(f"❌ Article {article_num} not found in exact format")
        print("\n📋 Searching for similar content...")
        
        # Search for similar
        results = collection.query(
            query_texts=[f"Article {article_num}"],
            n_results=5
        )
        
        if results['documents']:
            print("Found similar content:")
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
                print(f"{i+1}. {source}: {doc[:150]}...")
        else:
            print("No similar content found")

def check_all_articles():
    """Check what articles exist in the database"""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("constitutional_docs")
    
    print("\n📊 ALL ARTICLES IN DATABASE")
    print("="*60)
    
    all_docs = collection.get()
    
    articles_found = set()
    
    for doc in all_docs['documents'][:1000]:  # Check first 1000 docs
        # Find all article references
        article_matches = re.findall(r'Article\s+(\d+[A-Z\-]*)', doc, re.IGNORECASE)
        for match in article_matches:
            articles_found.add(match.upper())
    
    print(f"Found {len(articles_found)} unique article references")
    print("\nArticle numbers found:")
    
    # Sort numerically
    def sort_key(x):
        # Extract number part
        num_part = re.match(r'(\d+)', x)
        if num_part:
            return (int(num_part.group(1)), x)
        return (9999, x)
    
    sorted_articles = sorted(list(articles_found), key=sort_key)
    
    for i, article in enumerate(sorted_articles):
        print(f"{article}", end=", ")
        if (i + 1) % 10 == 0:
            print()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🔍 PDF ARTICLE CHECKER")
    print("="*60)
    
    # Check all articles
    check_all_articles()
    
    # Check specific articles
    articles_to_check = ["25A", "25", "19", "14", "8"]
    
    for article in articles_to_check:
        check_specific_article(article)
        print("\n" + "-"*60)