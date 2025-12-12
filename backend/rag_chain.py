# backend/rag_chain.py
import os
import sys
import logging
import re
from typing import List, Dict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Chroma
try:
    import chromadb
    from langchain_chroma import Chroma
    logger = logging.getLogger(__name__)
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger = logging.getLogger(__name__)
    logger.warning("Warning: Using deprecated Chroma")

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env from project root with better error handling
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    try:
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"Loaded environment from: {env_path}")
        
        # Test if API keys are loaded
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and groq_key.startswith("gsk_"):
            logger.info(f"GROQ_API_KEY found (starts with gsk_)")
        elif groq_key:
            logger.warning(f"GROQ_API_KEY found but doesn't look like a Groq key")
        else:
            logger.warning(f"GROQ_API_KEY not found in .env")
            
    except Exception as e:
        logger.error(f"Failed to load .env file: {str(e)}")
        logger.info(f"Check .env format. Example:")
        logger.info(f"   GROQ_API_KEY=gsk_your_key_here")
        logger.info(f"   OPENAI_API_KEY=sk-your-key-here")
else:
    logger.warning(f".env file not found at: {env_path}")
    logger.info(f"Create a .env file with your API keys")

# Import embeddings
from embeddings_local import LocalEmbeddings

# Initialize Groq availability flag
GROQ_AVAILABLE = False
llm_instance = None

# Try to import and initialize Groq LLM
try:
    # First try to install groq if not available
    try:
        import groq
    except ImportError:
        logger.warning("Groq module not installed. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
        import groq
        logger.info("Groq module installed successfully")
    
    # Now try to import OnlineLLM
    from llm_online import OnlineLLM
    
    # Check if API key exists
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and groq_api_key.startswith("gsk_"):
        try:
            llm_instance = OnlineLLM(provider="groq")
            GROQ_AVAILABLE = True
            logger.info("Groq LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq LLM: {str(e)}")
            logger.info("Check if your GROQ_API_KEY is valid at: https://console.groq.com")
    else:
        logger.warning("GROQ_API_KEY not found or invalid in .env file")
        logger.info("Get a free API key from: https://console.groq.com")
        
except Exception as e:
    logger.warning(f"Groq LLM not available: {str(e)}")
    GROQ_AVAILABLE = False

class GroqConstitutionAssistant:
    """Constitutional Assistant with Groq API for enhanced reasoning"""
    
    def __init__(self):
        self.chroma_path = "./chroma_db"
        self.collection_name = "pakistan_constitution"  
        self.embeddings = None
        self.vector_store = None
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use global llm_instance
        self.llm = llm_instance
        
        # Define greetings and responses
        self.greetings = self._initialize_greetings() 
        
        # Enhanced legal keywords for better matching
        self.legal_keywords = [
            "article", "articles", "constitution", "fundamental right", "freedom",
            "right to", "duty", "power", "president", "parliament", "judiciary",
            "amendment", "provision", "shall", "may", "must", "entitled",
            "equality", "education", "speech", "expression", "religion", "assembly",
            "association", "property", "trade", "profession", "life", "liberty",
            "dignity", "privacy", "security", "arrest", "detention", "trial",
            "compensation", "citizenship", "vote", "election", "government",
            "federal", "provincial", "local", "council", "senate", "national",
            "assembly", "supreme court", "high court", "judge", "justice"
        ]
        
        # Enhanced article patterns for accurate extraction
        self.article_patterns = [
            r'article\s+(\d+[A-Z]*)',
            r'art\.\s*(\d+[A-Z]*)',
            r'article\s+(\d+)\s*\([a-zA-Z]+\)',
            r'art\.\s*(\d+)\s*\([a-zA-Z]+\)',
            r'\b(\d+[A-Z])\b',
        ]
        
        # Question analysis patterns
        self.question_types = {
            "definition": ["what is", "define", "meaning of", "explain", "describe"],
            "procedure": ["how to", "how is", "process of", "procedure for", "steps to"],
            "comparison": ["difference between", "compare", "versus", "vs", "similar to"],
            "reasoning": ["why", "reason for", "cause of", "purpose of"],
            "existence": ["is there", "does the", "has", "have", "can"],
            "specific": ["article", "section", "clause", "act", "law"],
            "rights": ["right", "freedom", "entitle", "guarantee", "protect"]
        }
        
    def _initialize_greetings(self):
        """Initialize comprehensive greetings dictionary"""
        return {
            'hello': "Hello! I'm your Pakistan Constitution Assistant. I can answer questions about constitutional articles, rights, and legal provisions. How can I help you today?",
            'hi': "Hi there! I'm here to help you understand Pakistan's Constitution. What would you like to know?",
            'hey': "Hello! Ready to explore the Constitution of Pakistan with you.",
            'good morning': "Good morning! I'm here to help with constitutional questions.",
            'good afternoon': "Good afternoon! What constitutional provision would you like to learn about?",
            'good evening': "Good evening! I can help you understand Pakistan's Constitution.",
            'thank you': "You're welcome! Feel free to ask more questions about the Constitution.",
            'thanks': "You're welcome! Happy to assist with constitutional matters.",
            'bye': "Goodbye! Have a great day.",
            'goodbye': "Goodbye! Come back if you have more constitutional questions.",
            'how are you': "I'm functioning well, thank you! Ready to answer your constitutional queries.",
            'what is your name': "I'm the Pakistan Constitution Assistant, specializing in constitutional law and provisions.",
            'who are you': "I'm an AI assistant trained on Pakistan's Constitution. I can help explain articles, rights, and legal provisions.",
            'help': """I can help you with:
• Specific articles (e.g., Article 25A, Article 19, Article 14)
• Constitutional rights and freedoms
• Legal definitions and concepts
• Government structure and powers
• Historical context of constitutional provisions

Just ask your question in plain English!""",
        }
        
    def is_greeting(self, question: str) -> bool:
        """Check if the question is a greeting"""
        question_lower = question.lower().strip()
        
        if question_lower in self.greetings:
            return True
        
        for greeting in self.greetings.keys():
            if greeting in question_lower and len(greeting) > 2:
                return True
        
        return False
    
    def get_greeting_response(self, question: str) -> str:
        """Get appropriate greeting response"""
        question_lower = question.lower().strip()
        
        if question_lower in self.greetings:
            return self.greetings[question_lower]
        
        for greeting, response in self.greetings.items():
            if greeting in question_lower and len(greeting) > 2:
                return response
        
        return "Hello! I'm your Pakistan Constitution Assistant. How can I help you today?"
    
    def initialize(self):
        """Initialize the system"""
        try:
            self.embeddings = LocalEmbeddings()
            
            if not os.path.exists(self.chroma_path):
                raise FileNotFoundError(f"ChromaDB database not found at: {os.path.abspath(self.chroma_path)}")
            
            logger.info(f"Loading vector store from: {self.chroma_path}")
            logger.info(f"Collection name: {self.collection_name}")
            
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Verify database is populated
            try:
                collection_stats = self.vector_store._collection.count()
            except Exception as e:
                # Try alternative method to get count
                logger.warning(f"Could not get count via _collection: {e}")
                # Get all documents to count
                all_docs = self.vector_store.get()
                collection_stats = len(all_docs['ids']) if 'ids' in all_docs else 0
            
            if collection_stats == 0:
                raise ValueError(f"Database is empty. No documents found in collection '{self.collection_name}'.")
            
            logger.info(f"System initialized successfully (Session: {self.session_id})")
            logger.info(f"Database contains {collection_stats} document chunks")
            
            if self.llm:
                logger.info(f"Groq AI Enhancement: Available")
            else:
                logger.info(f"Groq AI Enhancement: Not available (using local data only)")
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise
    
    def answer_question(self, question: str) -> str:
        """Answer questions with enhanced reasoning using Groq"""
        question_lower = question.lower().strip()
        
        # Check if it's a greeting first
        if self.is_greeting(question):
            return self.get_greeting_response(question)
        
        try:
            logger.info(f"Processing question: {question[:50]}...")
            
            # Analyze question type
            question_type = self._analyze_question_type(question)
            
            # Extract article number if present
            article_num = self._extract_article_number(question)
            
            # Get relevant context from vector store using smart retrieval
            context = self._get_smart_context(question, question_type, article_num)
            
            # If no context found
            if not context:
                return self._no_results_response(question)
            
            # If an article number is found and we have article-specific content
            if article_num and self._is_article_specific_query(question):
                article_context = self._search_article_content(article_num)
                if article_context:
                    return self._format_article_response(article_num, article_context)
            
            # If Groq is available, use it for enhanced response
            if self.llm:
                return self._generate_enhanced_response(question, context, question_type)
            else:
                # Fallback to local response
                return self._generate_local_response(question, context, question_type, article_num)
                
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return self._error_response(question, str(e))
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of question"""
        question_lower = question.lower()
        
        for q_type, keywords in self.question_types.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return q_type
        
        return "general"
    
    def _is_article_specific_query(self, question: str) -> bool:
        """Check if this is specifically asking for an article"""
        question_lower = question.lower()
        article_num = self._extract_article_number(question)
        
        if article_num:
            return True
        
        # Check for article-specific patterns
        article_patterns = ["article", "art.", "section", "clause"]
        for pattern in article_patterns:
            if pattern in question_lower:
                return True
        
        return False
    
    def _get_smart_context(self, question: str, question_type: str, article_num: str, k: int = 7) -> str:
        """Get relevant context using smart retrieval strategies"""
        try:
            # Generate multiple search queries based on question type
            search_queries = self._generate_search_queries(question, question_type, article_num)
            
            all_docs = []
            seen_content = set()
            
            # Search with each query
            for query in search_queries:
                try:
                    docs = self.vector_store.similarity_search(query, k=min(3, k))
                    
                    for doc in docs:
                        content_hash = hash(doc.page_content[:200])
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            
                            # Score relevance based on question type and article number
                            relevance_score = self._calculate_relevance_score(doc, question, question_type, article_num)
                            all_docs.append((relevance_score, doc))
                except Exception as e:
                    logger.warning(f"Error in search query '{query}': {str(e)}")
                    continue
            
            # Sort by relevance score
            all_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Take top documents
            top_docs = [doc for score, doc in all_docs[:k]]
            
            context_parts = []
            for i, doc in enumerate(top_docs):
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown Document')
                page = doc.metadata.get('page', 'N/A')
                
                # Skip table of contents
                if self._is_table_of_contents(content):
                    continue
                
                # Clean and trim content
                content = self._clean_context_content(content, question, article_num)
                
                # Add to context
                context_parts.append(f"[Document {i+1}] Source: {source}, Page: {page}\nContent: {content}")
            
            if context_parts:
                return "\n\n".join(context_parts)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return ""
    
    def _generate_search_queries(self, question: str, question_type: str, article_num: str) -> List[str]:
        """Generate multiple search queries for better retrieval"""
        queries = [question]  # Always include original question
        
        # Add article-specific queries if article number exists
        if article_num:
            queries.append(f"Article {article_num}")
            queries.append(f"Article {article_num} constitution")
            queries.append(f"Article {article_num} of constitution")
        
        # Add question type specific queries
        if question_type == "definition":
            key_terms = self._extract_key_terms(question)
            for term in key_terms[:3]:
                queries.append(f"definition {term} constitution")
                queries.append(f"{term} meaning constitution")
        
        elif question_type == "rights":
            queries.append("fundamental rights constitution")
            queries.append("rights constitution Pakistan")
        
        elif question_type == "procedure":
            queries.append("procedure constitution Pakistan")
            queries.append("how constitution")
        
        # Add constitutional context queries
        queries.append("constitution Pakistan")
        queries.append("constitutional law")
        
        # Add key terms as queries
        key_terms = self._extract_key_terms(question)
        for term in key_terms[:3]:
            if term not in queries:
                queries.append(term)
        
        # Remove duplicates and limit
        unique_queries = []
        seen = set()
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries[:10]
    
    def _calculate_relevance_score(self, doc, question: str, question_type: str, article_num: str) -> float:
        """Calculate relevance score for a document"""
        content = doc.page_content.lower()
        question_lower = question.lower()
        score = 0.0
        
        # Base score
        score += 0.2
        
        # Article match bonus
        if article_num:
            if f"article {article_num}" in content or f"art. {article_num}" in content:
                score += 0.5
        
        # Question term matches
        question_terms = self._extract_key_terms(question)
        for term in question_terms:
            if term in content:
                score += 0.1
        
        # Question type specific bonuses
        if question_type == "definition" and ("definition" in content or "means" in content):
            score += 0.2
        elif question_type == "procedure" and ("procedure" in content or "process" in content):
            score += 0.2
        elif question_type == "rights" and ("right" in content or "entitled" in content):
            score += 0.2
        
        # Penalize table of contents
        if self._is_table_of_contents(content):
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _clean_context_content(self, content: str, question: str, article_num: str) -> str:
        """Clean and focus context content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Try to extract most relevant parts
        if article_num:
            # For article-specific queries, try to get text around the article
            article_pattern = rf'(?:article\s+{article_num}|art\.\s+{article_num})[\.\:\-\s]+(.*?)(?=(?:article|art\.)\s+\d+|\Z)'
            match = re.search(article_pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 100:
                    return extracted[:500] + "..."
        
        # General truncation
        if len(content) > 500:
            # Try to find a good breaking point
            sentences = content.split('. ')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
            else:
                return content[:500] + "..."
        
        return content
    
    def _generate_enhanced_response(self, question: str, context: str, question_type: str) -> str:
        """Generate enhanced response using Groq with improved prompt engineering"""
        try:
            logger.info("Generating enhanced response with Groq...")
            
            # Extract article number from question
            article_num = self._extract_article_number(question)
            
            # Prepare enhanced prompt
            prompt = self._build_enhanced_prompt(question, context, question_type, article_num)
            
            # Get response from Groq
            groq_response = self.llm.invoke(prompt)
            
            # Post-process the response
            groq_response = self._clean_response(groq_response)
            
            # Format the final response
            response = self._format_final_response(groq_response, article_num, question)
            
            logger.info("Groq response generated successfully")
            return response
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            logger.info("Falling back to local response...")
            # Fallback to local response
            return self._generate_local_response(question, context, question_type, article_num)
    
    def _build_enhanced_prompt(self, question: str, context: str, question_type: str, article_num: str) -> str:
        """Build enhanced prompt for Groq"""
        prompt = f"""You are a constitutional law expert for Pakistan. Your task is to answer questions based EXACTLY on the provided constitutional context.

QUESTION: {question}

QUESTION TYPE: {question_type.upper()}
"""
        
        if article_num:
            prompt += f"PRIMARY ARTICLE: {article_num}\n\n"
        
        prompt += f"""CONSTITUTIONAL CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer based ONLY on the constitutional context provided
2. If specific Articles are mentioned, cite them exactly (e.g., Article {article_num if article_num else 'XX'})
3. Structure your answer clearly with appropriate headings if helpful
4. If information is incomplete in the context, acknowledge limitations but provide available information
5. Maintain a professional, academic tone appropriate for constitutional law
6. Include relevant constitutional principles and interpretations where applicable
7. End with a brief summary of key constitutional points

CONSTITUTIONAL EXPERT ANSWER:"""
        
        return prompt
    
    def _format_final_response(self, groq_response: str, article_num: str, question: str) -> str:
        """Format the final response"""
        # Extract title from response or create one
        title = self._extract_title_from_response(groq_response, article_num, question)
        
        # Format response
        response = f"Title: {title}\n\n"
        response += f"Full Article Text: {groq_response}\n\n"
        
        # Add source reference
        mentioned_articles = self._extract_mentioned_articles(groq_response)
        if article_num and f"Article {article_num}" not in mentioned_articles:
            mentioned_articles.append(f"Article {article_num}")
        
        if mentioned_articles:
            response += f"Source: Constitution of Pakistan ({', '.join(mentioned_articles)})"
        else:
            response += f"Source: Constitution of Pakistan"
        
        return response
    
    def _extract_title_from_response(self, response: str, article_num: str, question: str) -> str:
        """Extract title from response or create one"""
        if article_num:
            # Try to find a title in the response
            lines = response.split('\n')
            for line in lines[:3]:
                line = line.strip()
                if line and not line.startswith('*') and len(line) < 100:
                    # Check if it looks like a title
                    if line.endswith(':') or line[0].isupper():
                        return line
        
        # Create title from question
        if article_num:
            return f"Article {article_num} Analysis"
        else:
            # Create meaningful title from question
            question_words = question.split()[:5]
            title = ' '.join(question_words)
            if len(title) > 40:
                title = title[:37] + "..."
            return f"Constitutional Analysis: {title}"
    
    def _extract_mentioned_articles(self, text: str) -> List[str]:
        """Extract all mentioned articles from text"""
        articles = set()
        for pattern in self.article_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                articles.add(f"Article {match.upper()}")
        return sorted(list(articles))
    
    def _generate_local_response(self, question: str, context: str, question_type: str, article_num: str) -> str:
        """Generate local response without Groq"""
        logger.info("Generating local response...")
        
        # Create title
        if article_num:
            title = f"Article {article_num} Analysis"
        else:
            title = f"Constitutional Analysis"
        
        # Format response
        response = f"Title: {title}\n\n"
        
        # Add question analysis
        response += f"Question: {question}\n\n"
        response += f"Question Type: {question_type}\n\n"
        
        # Add context summary
        if context:
            response += f"Relevant Context Found: {len(context.split('[Document'))} document sections\n\n"
        
        # Add analysis based on question type
        response += f"Analysis:\n\n"
        
        if article_num:
            response += f"This question references Article {article_num} of the Constitution of Pakistan.\n\n"
        
        if question_type == "definition":
            response += f"The Constitution provides definitions and explanations for various legal concepts and terms.\n"
        elif question_type == "procedure":
            response += f"Constitutional procedures establish the framework for government operations and legal processes.\n"
        elif question_type == "rights":
            response += f"Fundamental rights are enshrined in the Constitution to protect individual freedoms and ensure justice.\n"
        else:
            response += f"The constitutional framework provides comprehensive guidelines for governance, rights, and legal processes.\n"
        
        response += f"\nFull Article Text: Based on the constitutional context provided, the relevant provisions address the query comprehensively.\n\n"
        response += f"Source: Constitution of Pakistan"
        
        return response
    
    def _search_article_content(self, article_num: str) -> str:
        """Search for specific article content with more comprehensive results"""
        try:
            # Search for article with multiple queries
            queries = [
                f"Article {article_num}",
                f"Art. {article_num}",
                f"Article {article_num}:",
                f"Article {article_num}.",
                f"Article {article_num} -"
            ]
            
            article_texts = []
            
            for query in queries:
                docs = self.vector_store.similarity_search(query, k=5)
                
                for doc in docs:
                    content = doc.page_content
                    if self._is_article_content(content, article_num):
                        article_text = self._extract_article_text(content, article_num)
                        if article_text and article_text not in article_texts:
                            article_texts.append(article_text)
            
            # If we found article content, return it
            if article_texts:
                return "\n\n".join(article_texts)
            
            # If no direct article found, try to get relevant context
            docs = self.vector_store.similarity_search(f"Article {article_num}", k=3)
            if docs:
                return "\n\n".join([doc.page_content for doc in docs])
            
            return ""
            
        except Exception as e:
            logger.error(f"Error searching article: {str(e)}")
            return ""
    
    def _format_article_response(self, article_num: str, context: str) -> str:
        """Format article-specific response"""
        try:
            # Extract title from context
            title = self._extract_article_title(context, article_num)
            
            # Clean up the title
            title = title.strip()
            
            # Get the full article text
            full_text = context.strip()
            
            # Clean the full text
            full_text = self._clean_article_text(full_text)
            
            # Format the response
            response = f"Title: {title}\n\n"
            response += f"Full Article Text: {full_text}\n\n"
            response += f"Source: Constitution of Pakistan (Article {article_num})"
            
            return response
            
        except Exception as e:
            logger.error(f"Error formatting article response: {str(e)}")
            # Fallback to simple format
            return f"Title: Article {article_num}\n\nFull Article Text: {context}\n\nSource: Constitution of Pakistan (Article {article_num})"
    
    def _extract_article_title(self, text: str, article_num: str) -> str:
        """Extract article title from text"""
        # Look for title after article number
        patterns = [
            rf'article\s+{article_num}[\.\:\-\s]+(.*?\.)(?:\s|$)',
            rf'art\.\s+{article_num}[\.\:\-\s]+(.*?\.)(?:\s|$)',
            rf'{article_num}\.\s*(.*?\.)(?:\s|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                # Clean up the title
                title = re.sub(r'^\W+', '', title)
                title = re.sub(r'\s+', ' ', title)
                if title:
                    return title
        
        # If no title found, use first sentence
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if sentences:
            first_sentence = sentences[0]
            # Remove any leading article reference
            first_sentence = re.sub(rf'^(article\s+{article_num}|art\.\s+{article_num})[\.\:\-\s]*', '', first_sentence, flags=re.IGNORECASE)
            if first_sentence.strip():
                return first_sentence.strip()
        
        # Default title
        return f"Article {article_num}"
    
    def _clean_article_text(self, text: str) -> str:
        """Clean up article text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Remove page numbers and other artifacts
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text)
        text = re.sub(r'-\s*\d+\s*-', '', text)
        
        return text.strip()
    
    def _extract_article_number(self, text: str) -> str:
        """Extract article number from text"""
        for pattern in self.article_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                article_num = match.group(1).upper()
                article_num = re.sub(r'\s+', '', article_num)
                return article_num
        
        return ""
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        key_terms = []
        for word in words:
            if word not in stop_words and word in self.legal_keywords:
                key_terms.append(word)
        
        # If no legal keywords found, return general key terms
        if not key_terms:
            for word in words:
                if word not in stop_words:
                    key_terms.append(word)
        
        return list(set(key_terms))[:5]
    
    def _is_article_content(self, content: str, article_num: str) -> bool:
        """Check if content is article content"""
        patterns = [
            rf'article\s+{article_num}[\.\:\-]',
            rf'art\.\s+{article_num}[\.\:\-]',
            rf'{article_num}\.\s+',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                legal_markers = ["shall", "may", "must", "entitled", "right", "duty", "power"]
                content_lower = content.lower()
                marker_count = sum(1 for marker in legal_markers if marker in content_lower)
                
                return marker_count >= 1 and not self._is_table_of_contents(content)
        
        return False
    
    def _extract_article_text(self, content: str, article_num: str) -> str:
        """Extract article text"""
        patterns = [
            rf'(?:article\s+{article_num}|art\.\s+{article_num})[\.\:\-]\s*(.*?)(?=(?:article|art\.)\s+\d+|\Z)',
            rf'{article_num}\.\s*(.*?)(?=\d+\.\s+|$|\n\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                extracted = re.sub(r'\s+', ' ', extracted)
                return extracted
        
        return content[:500].strip()
    
    def _is_table_of_contents(self, content: str) -> bool:
        """Check if content is table of contents"""
        if len(content) < 100:
            return False
        
        content_lower = content[:300].lower()
        
        toc_indicators = [
            "contents", "table of", "preamble", "part i", "chapter i",
            "schedule", "index", "article page", "articles page",
            "......", "........", "....."
        ]
        
        indicator_count = sum(1 for indicator in toc_indicators if indicator in content_lower)
        
        if re.search(r'\d+\s+\.+\s+\d+', content):
            return True
        
        return indicator_count >= 2
    
    def _clean_response(self, response: str) -> str:
        """Clean up the response"""
        # Remove any markdown formatting
        response = re.sub(r'#+\s*', '', response)
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
        response = re.sub(r'\*(.*?)\*', r'\1', response)
        
        # Clean up whitespace
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = response.strip()
        
        return response
    
    def _no_results_response(self, question: str) -> str:
        """No results response"""
        response = f"No Direct Constitutional Results Found\n\n"
        response += f"Query: '{question}'\n\n"
        response += f"Suggestions for Better Results:\n"
        response += f"• Search by specific Article number (e.g., 'Article 25A', 'Article 19')\n"
        response += f"• Ask about fundamental rights or government structure\n"
        response += f"• Use constitutional terminology\n"
        response += f"• Check if the PDFs have been properly ingested\n\n"
        response += f"Common Constitutional Topics:\n"
        response += f"• Fundamental Rights (Articles 8-28)\n"
        response += f"• Principles of Policy (Articles 29-40)\n"
        response += f"• The Federation (Articles 41-100)\n"
        response += f"• The President and Parliament\n"
        response += f"• The Judiciary\n"
        
        return response
    
    def _error_response(self, question: str, error_msg: str) -> str:
        """Error response"""
        response = f"System Error\n\n"
        response += f"Query: {question}\n"
        response += f"Error: {error_msg[:100]}\n\n"
        response += f"Troubleshooting:\n"
        response += f"1. Ensure ChromaDB is initialized (run ingest.py)\n"
        response += f"2. Check if PDFs are in the data directory\n"
        response += f"3. Verify the .env file has correct API keys\n"
        response += f"4. Check logs for detailed error information\n"
        
        return response

# Global instance for compatibility
assistant = GroqConstitutionAssistant()

def answer_question(question: str) -> str:
    """Public function for external use"""
    if assistant.vector_store is None:
        assistant.initialize()
    
    return assistant.answer_question(question)

# For backward compatibility
def find_real_article(article_num: str) -> str:
    """Legacy function - redirects to new system"""
    return answer_question(f"Article {article_num}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Pakistan Constitution Assistant with Groq API")
    print("="*60 + "\n")
    
    assistant.initialize()
    
    print(f"Try these queries:")
    print("1. Article 25A - Right to education")
    print("2. What are fundamental rights?")
    print("3. How is the President elected?")
    print("4. Explain the role of Supreme Court")
    print("5. What is Article 19 about?")
    print("\nSay hello or ask for help")
    print("\n" + "="*60 + "\n")
    
    # Interactive mode
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            answer = answer_question(question)
            print(f"\n{answer}\n")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")