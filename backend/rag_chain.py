# rag_chain.py - FINAL WORKING VERSION
import os
import logging
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Import Chroma
try:
    import chromadb
    from langchain_chroma import Chroma
    logger = logging.getLogger(__name__)
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ Using deprecated Chroma")

from embeddings_local import LocalEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class WorkingPDFAssistant:
    """Working assistant that finds actual article text"""
    
    def __init__(self):
        self.chroma_path = "./chroma_db"
        self.collection_name = "constitutional_docs"
        self.embeddings = None
        self.vector_store = None
        
    def initialize(self):
        """Initialize the system"""
        try:
            self.embeddings = LocalEmbeddings()
            
            if not os.path.exists(self.chroma_path):
                raise FileNotFoundError("ChromaDB not found")
            
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            print("✅ System ready")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            raise
    
    def answer_question(self, question: str) -> str:
        """Answer questions from PDFs"""
        question_lower = question.lower().strip()
        
        # Handle article questions specially
        if "article" in question_lower or "art." in question_lower:
            article_num = self._extract_article_number(question)
            if article_num:
                return self._get_article_content(article_num, question)
        
        # Handle general questions
        return self._get_general_answer(question)
    
    def _get_article_content(self, article_num: str, original_question: str) -> str:
        """Get article content with proper search"""
        print(f"\n🔍 Searching for Article {article_num}...")
        
        # First: Try to find actual article text (not table of contents)
        actual_text = self._find_actual_article_text(article_num)
        
        if actual_text:
            return self._format_article_answer(actual_text, article_num)
        
        # Second: Try to find in detailed sections
        detailed_content = self._find_detailed_content(article_num)
        
        if detailed_content:
            return self._format_detailed_answer(detailed_content, article_num)
        
        # Third: Check if it's in table of contents
        toc_content = self._find_in_table_of_contents(article_num)
        
        if toc_content:
            return self._format_toc_answer(toc_content, article_num, original_question)
        
        # Not found at all
        return self._not_found_response(article_num)
    
    def _find_actual_article_text(self, article_num: str) -> List[Dict]:
        """Find actual article text (not table of contents)"""
        # Search patterns that indicate actual article text
        search_queries = [
            f"{article_num}. ",  # Article with period (indicates start of text)
            f"Article {article_num}:",
            f"Article {article_num} -",
            f"Article {article_num}\n",
            f"{article_num}."  # Just the number with period
        ]
        
        results = []
        
        for query in search_queries:
            docs = self.vector_store.similarity_search(query, k=5)
            
            for doc in docs:
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                
                # Skip table of contents
                if self._is_table_of_contents(content):
                    continue
                
                # Check if this looks like actual article text
                if self._is_actual_article_text(content, article_num):
                    # Extract the article text
                    article_text = self._extract_article_from_content(content, article_num)
                    if article_text:
                        results.append({
                            'text': article_text,
                            'source': source,
                            'page': page,
                            'query': query
                        })
        
        return results
    
    def _find_detailed_content(self, article_num: str) -> List[Dict]:
        """Find detailed content mentioning the article"""
        # Search for the article number in context
        docs = self.vector_store.similarity_search(f"Article {article_num}", k=15)
        
        results = []
        
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            # Skip if it's just table of contents
            if self._is_table_of_contents(content):
                continue
            
            # Check if article is mentioned with some context
            if any(term in content for term in [f"Article {article_num}", f"Art. {article_num}", f"{article_num}. "]):
                # Extract context around the mention
                context = self._extract_context(content, article_num)
                if context and len(context.strip()) > 50:
                    results.append({
                        'text': context,
                        'source': source,
                        'page': page,
                        'type': 'context'
                    })
        
        return results
    
    def _find_in_table_of_contents(self, article_num: str) -> List[Dict]:
        """Find article in table of contents"""
        docs = self.vector_store.similarity_search(f"Article {article_num}", k=10)
        
        results = []
        
        for doc in docs:
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            # Check if this is table of contents
            if self._is_table_of_contents(content):
                # Extract the line mentioning the article
                lines = content.split('\n')
                for line in lines:
                    if any(term in line for term in [f"Article {article_num}", f"Art. {article_num}", f"{article_num}. "]):
                        results.append({
                            'text': line.strip(),
                            'source': source,
                            'page': page,
                            'type': 'toc'
                        })
                        break
        
        return results
    
    def _is_table_of_contents(self, content: str) -> bool:
        """Check if content is table of contents"""
        toc_indicators = [
            "contents", "table of", "article s page", "articles page",
            "preamble", "part i", "chapter", "schedule"
        ]
        
        content_lower = content[:200].lower()  # Check first 200 chars
        
        # If it's very short and contains dots/numbers pattern, it's likely TOC
        if len(content) < 300:
            if re.search(r'\d+\s+\.+\s+\d+', content):  # Pattern like "25 ...... 15"
                return True
        
        # Check for TOC indicators
        indicator_count = sum(1 for indicator in toc_indicators if indicator in content_lower)
        return indicator_count >= 2
    
    def _is_actual_article_text(self, content: str, article_num: str) -> bool:
        """Check if content is actual article text"""
        # Legal language indicators
        legal_indicators = ["shall", "may", "must", "should", "entitled", 
                           "right", "duty", "power", "authority", "prohibited",
                           "provided that", "notwithstanding", "subject to"]
        
        content_lower = content.lower()
        
        # Check for article marker
        if not any(term in content for term in [f"Article {article_num}", f"Art. {article_num}", f"{article_num}. "]):
            return False
        
        # Check for legal language
        legal_count = sum(1 for word in legal_indicators if word in content_lower)
        
        # Should have at least 2 legal terms and not be just TOC
        return legal_count >= 2 and not self._is_table_of_contents(content)
    
    def _extract_article_from_content(self, content: str, article_num: str) -> str:
        """Extract article text from content"""
        # Try different patterns to extract article text
        patterns = [
            rf"(Article\s+{article_num}[\.\:\-]\s*.*?)(?=Article\s+\d+|$)",
            rf"(Art\.\s+{article_num}[\.\:\-]\s*.*?)(?=Art\.\s+\d+|$)",
            rf"({article_num}\.\s+.*?)(?=\d+\.\s+|Article\s+|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the content (it might be the whole article)
        return content.strip()
    
    def _extract_context(self, content: str, article_num: str, lines_before: int = 2, lines_after: int = 4) -> str:
        """Extract context around article mention"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if any(term in line for term in [f"Article {article_num}", f"Art. {article_num}", f"{article_num}. "]):
                start = max(0, i - lines_before)
                end = min(len(lines), i + lines_after + 1)
                return '\n'.join(lines[start:end])
        
        # If not found line by line, return beginning of content
        return content[:500]
    
    def _format_article_answer(self, results: List[Dict], article_num: str) -> str:
        """Format actual article text answer"""
        if not results:
            return ""
        
        best_result = results[0]  # Use the best match
        
        response = f"📜 **ARTICLE {article_num}**\n"
        response += "=" * 50 + "\n\n"
        
        # Clean the text
        text = best_result['text']
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = text.strip()
        
        # Capitalize properly
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        response += f"{text}\n\n"
        
        # Add source
        response += "**Source:** "
        response += best_result['source']
        if best_result['page'] != 'N/A':
            response += f" (Page {best_result['page']})"
        response += "\n\n"
        
        # Add note if there are more results
        if len(results) > 1:
            response += f"*Also found in {len(results)-1} other locations*\n"
        
        return response
    
    def _format_detailed_answer(self, results: List[Dict], article_num: str) -> str:
        """Format detailed content answer"""
        response = f"🔍 **INFORMATION ABOUT ARTICLE {article_num}**\n"
        response += "=" * 50 + "\n\n"
        
        response += f"While I couldn't find the exact text of Article {article_num}, "
        response += "here's relevant information from your documents:\n\n"
        
        for i, result in enumerate(results[:3]):  # Show top 3
            text = result['text']
            source = result['source']
            page = result['page']
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            response += f"**{i+1}. From {source}**"
            if page != 'N/A':
                response += f" (Page {page})"
            response += ":\n"
            
            response += f"> {text}\n\n"
        
        response += "---\n"
        response += "*Note: The full article text might be in a different format or section.*\n"
        
        return response
    
    def _format_toc_answer(self, results: List[Dict], article_num: str, original_question: str) -> str:
        """Format table of contents answer"""
        response = f"📋 **ARTICLE {article_num} - TABLE OF CONTENTS**\n"
        response += "=" * 50 + "\n\n"
        
        response += f"Article {article_num} is listed in your documents, but I could only find it in the table of contents.\n\n"
        
        response += "**Where it appears:**\n"
        for i, result in enumerate(results[:3]):
            text = result['text'].strip()
            source = result['source']
            
            response += f"{i+1}. **{source}**: {text}\n"
        
        response += "\n**What this means:**\n"
        response += "✓ Article exists in the document structure\n"
        response += f"✗ Actual text of Article {article_num} not found in searchable content\n"
        response += "✗ Might be in scanned/images or different formatting\n\n"
        
        response += "**Suggestions:**\n"
        response += "1. Check the actual PDF file for Article {article_num}\n"
        response += "2. The article might be on a scanned page (not searchable text)\n"
        response += "3. Try searching for related terms instead\n"
        
        return response
    
    def _not_found_response(self, article_num: str) -> str:
        """Response when article is not found at all"""
        response = f"❌ **ARTICLE {article_num} NOT FOUND**\n"
        response += "=" * 50 + "\n\n"
        
        response += f"I searched your PDF documents but could not find **Article {article_num}** in any form.\n\n"
        
        response += "**Possible reasons:**\n"
        response += "1. Article doesn't exist in your uploaded PDFs\n"
        response += "2. PDF contains only table of contents, not full text\n"
        response += "3. Text is in images/scanned pages (not searchable)\n"
        response += "4. Different numbering system used\n\n"
        
        response += "**Try asking about:**\n"
        
        # Suggest similar articles that DO exist
        similar_articles = self._find_similar_articles(article_num)
        if similar_articles:
            response += "Similar articles found in your PDFs:\n"
            for article in similar_articles[:5]:
                response += f"• Article {article}\n"
        else:
            response += "• Fundamental rights\n• Constitutional provisions\n• Legal articles\n"
        
        return response
    
    def _find_similar_articles(self, article_num: str) -> List[str]:
        """Find similar articles that exist"""
        try:
            # Get all documents and extract article numbers
            all_docs = self.vector_store.similarity_search("Article", k=50)
            
            articles_found = set()
            
            for doc in all_docs:
                content = doc.page_content
                # Extract article numbers
                matches = re.findall(r'Article\s+(\d+[A-Z]*)', content, re.IGNORECASE)
                for match in matches:
                    if match != article_num:  # Don't include the one we're looking for
                        articles_found.add(match)
            
            return sorted(list(articles_found))
        except:
            return []
    
    def _get_general_answer(self, question: str) -> str:
        """Answer general questions"""
        print(f"🔍 Searching for: '{question}'")
        
        docs = self.vector_store.similarity_search(question, k=10)
        
        if not docs:
            return f"❌ No information found about '{question}'"
        
        # Filter out table of contents
        relevant_docs = []
        for doc in docs:
            content = doc.page_content
            if not self._is_table_of_contents(content):
                relevant_docs.append(doc)
        
        if not relevant_docs:
            # If only TOC found, say so
            return f"📋 Only found '{question}' in table of contents, not detailed text."
        
        response = f"📚 **INFORMATION ABOUT: {question}**\n"
        response += "=" * 50 + "\n\n"
        
        for i, doc in enumerate(relevant_docs[:5]):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            
            # Clean content
            content = re.sub(r'\s+', ' ', content)
            content = content.strip()
            
            if len(content) > 300:
                content = content[:297] + "..."
            
            response += f"**{i+1}. {source}**"
            if page != 'N/A':
                response += f" (Page {page})"
            response += ":\n"
            
            response += f"> {content}\n\n"
        
        response += f"*Found {len(relevant_docs)} relevant sections*\n"
        
        return response
    
    def _extract_article_number(self, text: str) -> str:
        """Extract article number from text"""
        patterns = [
            r'article\s+(\d+[A-Z\-]*)',
            r'art\.\s*(\d+[A-Z\-]*)',
            r'\b(\d+[A-Z\-])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        return ""

# Global instance
assistant = WorkingPDFAssistant()

def answer_question(question: str) -> str:
    """Public function"""
    if assistant.vector_store is None:
        assistant.initialize()
    
    return assistant.answer_question(question)


# SPECIAL: Search for actual article text
def find_real_article(article_num: str) -> str:
    """Find real article text (not table of contents)"""
    import chromadb
    
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("constitutional_docs")
    
    print(f"\n🔍 FINDING REAL TEXT OF ARTICLE {article_num}")
    print("="*60)
    
    # Get all documents
    all_docs = collection.get()
    
    real_texts = []
    
    for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
        # Skip if it's table of contents
        if "contents" in doc[:200].lower() or "article s page" in doc.lower():
            continue
        
        # Look for article text patterns
        patterns = [
            rf"Article\s+{article_num}[\.\:\-]\s+(.*?)(?=Article\s+\d+)",
            rf"{article_num}\.\s+(.*?)(?=\d+\.\s+)",
            rf"Art\.\s+{article_num}[\.\:\-]\s+(.*?)(?=Art\.\s+\d+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, doc, re.IGNORECASE | re.DOTALL)
            if matches:
                source = metadata.get('source', 'Unknown') if metadata else 'Unknown'
                page = metadata.get('page', 'N/A')
                
                real_texts.append({
                    'text': matches[0].strip(),
                    'source': source,
                    'page': page,
                    'doc_index': i
                })
                break
    
    if real_texts:
        print(f"✅ Found {len(real_texts)} instances of real article text")
        print()
        
        # Show the longest/best match
        real_texts.sort(key=lambda x: len(x['text']), reverse=True)
        best = real_texts[0]
        
        print(f"📜 **ARTICLE {article_num}:**")
        print("-" * 50)
        
        text = best['text']
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        print(text[:800])
        if len(text) > 800:
            print("... [continued]")
            print(text[800:1600] if len(text) > 1600 else text[800:])
        
        print("-" * 50)
        print(f"\n📁 Source: {best['source']}")
        if best['page'] != 'N/A':
            print(f"📄 Page: {best['page']}")
        
        return text
    else:
        print(f"❌ No real article text found for Article {article_num}")
        print("\n⚠️ Your PDFs might only contain table of contents, not full article text.")
        return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📄 WORKING PDF ASSISTANT")
    print("="*60)
    
    assistant.initialize()
    
    # Test with your questions
    test_questions = [
        "What does Article 25A say?",
        "Article 19",
        "What is freedom of speech?",
        "Explain Article 14",
        "What are fundamental rights?"
    ]
    
    for i, question in enumerate(test_questions):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {question}")
        print(f"{'='*60}")
        
        answer = answer_question(question)
        print(f"\n{answer}")