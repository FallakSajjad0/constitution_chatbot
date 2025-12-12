# rag_chain.py 
import os
import logging
import re
import requests
import json
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from datetime import datetime

# Import Chroma
try:
    import chromadb
    from langchain_chroma import Chroma
    logger = logging.getLogger(__name__)
except ImportError:
    from langchain_community.vectorstores import Chroma
    logger = logging.getLogger(__name__)
    logger.warning("Warning: Using deprecated Chroma")

from embeddings_local import LocalEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DetailedConstitutionAssistant:
    """Assistant for constitutional analysis with detailed paragraph answers"""
    
    def __init__(self):
        self.chroma_path = "./chroma_db"
        self.collection_name = "constitutional_docs"
        self.embeddings = None
        self.vector_store = None
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load API keys from .env
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.use_ai_enhancement = any([self.gemini_api_key, self.openai_api_key, self.groq_api_key])
        
        # Define greetings and responses
        self.greetings = self._initialize_greetings()
        
        # Enhanced legal keywords for better matching
        self.legal_keywords = [
            "article", "articles", "constitution", "fundamental right", "freedom",
            "right to", "duty", "power", "president", "parliament", "judiciary",
            "amendment", "provision", "shall", "may", "must", "entitled"
        ]
        
        # Enhanced article patterns for accurate extraction
        self.article_patterns = [
            r'article\s+(\d+[A-Z]*)',
            r'art\.\s*(\d+[A-Z]*)',
            r'article\s+(\d+)\s*\([a-zA-Z]+\)',
            r'art\.\s*(\d+)\s*\([a-zA-Z]+\)',
            r'\b(\d+[A-Z])\b',
        ]
        
    def _initialize_greetings(self):
        """Initialize comprehensive greetings dictionary"""
        return {
            'hello': "Hello! I'm your Pakistan Constitution Assistant. How can I help you today?",
            'hi': "Hi there! I can answer questions about Pakistan's Constitution. What would you like to know?",
            'hey': "Hey! Ready to explore the Constitution of Pakistan with you.",
            'good morning': "Good morning! I'm here to help with constitutional questions.",
            'good afternoon': "Good afternoon! What constitutional provision would you like to learn about?",
            'good evening': "Good evening! I can help you understand Pakistan's Constitution.",
            'thank you': "You're welcome! Feel free to ask more questions about the Constitution.",
            'thanks': "You're welcome! Happy to assist with constitutional matters.",
            'bye': "Goodbye! Have a great day.",
            'goodbye': "Goodbye! Come back if you have more constitutional questions.",
            'how are you': "I'm doing well, thank you! Ready to answer your constitutional queries.",
            'what is your name': "I'm the Pakistan Constitution Assistant, specializing in constitutional law and provisions.",
            'who are you': "I'm an AI assistant trained on Pakistan's Constitution. I can help explain articles, rights, and legal provisions.",
            'help': "I can help you with:\n• Specific articles (e.g., Article 25A, Article 19)\n• Constitutional rights and freedoms\n• Legal definitions and concepts\n• Government structure and powers\nJust ask your question!",
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
                raise FileNotFoundError("ChromaDB database not found. Please ensure PDFs have been processed.")
            
            self.vector_store = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Verify database is populated
            collection_stats = self.vector_store._collection.count()
            if collection_stats == 0:
                raise ValueError("Database is empty. No documents found.")
            
            print(f"System initialized successfully (Session: {self.session_id})")
            print(f"Database contains {collection_stats} document chunks")
            
            if self.use_ai_enhancement:
                print(f"AI Enhancement: Available")
            else:
                print(f"AI Enhancement: Not available (using local data only)")
            
        except Exception as e:
            print(f"Initialization failed: {str(e)}")
            raise
    
    def answer_question(self, question: str) -> str:
        """Answer questions with detailed paragraph format"""
        question_lower = question.lower().strip()
        
        # Check if it's a greeting first
        if self.is_greeting(question):
            return self.get_greeting_response(question)
        
        try:
            # Determine query type
            query_type = self._determine_query_type(question)
            
            if query_type == "article_query":
                article_num = self._extract_article_number(question)
                if article_num:
                    return self._generate_detailed_article_response(article_num, question)
                else:
                    return self._generate_detailed_general_response(question)
            elif query_type == "legal_concept":
                return self._generate_detailed_concept_response(question)
            else:
                return self._generate_detailed_general_response(question)
                
        except Exception as e:
            return self._error_response(question, str(e))
    
    def _determine_query_type(self, question: str) -> str:
        """Determine the type of query for appropriate response structure"""
        question_lower = question.lower()
        
        # Check for article queries
        article_patterns = [
            r'article\s+\d+[A-Z]*',
            r'art\.\s*\d+[A-Z]*',
            r'\b\d+[A-Z]\b',
        ]
        
        for pattern in article_patterns:
            if re.search(pattern, question_lower):
                return "article_query"
        
        # Check for legal concept queries
        legal_concept_keywords = [
            "what is", "explain", "define", "meaning of", "concept of",
            "understanding", "analysis of", "discuss", "describe",
            "how does", "what are", "tell me about"
        ]
        
        for keyword in legal_concept_keywords:
            if keyword in question_lower:
                return "legal_concept"
        
        return "general_query"
    
    def _extract_article_number(self, text: str) -> str:
        """Extract article number from text"""
        for pattern in self.article_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                article_num = match.group(1).upper()
                article_num = re.sub(r'\s+', '', article_num)
                return article_num
        
        return ""
    
    def _generate_detailed_article_response(self, article_num: str, question: str) -> str:
        """Generate detailed response for article queries"""
        print(f"Searching for Article {article_num}")
        
        # Search for article content
        search_results = self._search_article_content(article_num)
        
        if not search_results:
            return self._not_found_response(article_num)
        
        # Structure the response
        response = self._format_detailed_article_response(search_results, article_num, question)
        return response
    
    def _search_article_content(self, article_num: str) -> List[Dict]:
        """Search for article content in database"""
        all_results = []
        
        # Multiple search queries for better coverage
        search_queries = [
            f"Article {article_num}:",
            f"Article {article_num}.",
            f"Article {article_num} ",
            f"Art. {article_num}:",
            f"Art. {article_num}.",
            f"{article_num}. ",
        ]
        
        for query in search_queries:
            try:
                docs = self.vector_store.similarity_search(query, k=6)
                
                for doc in docs:
                    content = doc.page_content
                    source = doc.metadata.get('source', 'Unknown Document')
                    page = doc.metadata.get('page', 'N/A')
                    
                    if self._is_table_of_contents(content):
                        continue
                    
                    if self._is_article_content(content, article_num):
                        article_text = self._extract_article_text(content, article_num)
                        
                        if article_text and len(article_text.strip()) > 30:
                            all_results.append({
                                'text': article_text,
                                'source': source,
                                'page': page,
                                'type': 'article_text'
                            })
                    elif self._mentions_article(content, article_num):
                        context = self._extract_context(content, article_num)
                        if context and len(context.strip()) > 50:
                            all_results.append({
                                'text': context,
                                'source': source,
                                'page': page,
                                'type': 'context'
                            })
            except Exception as e:
                continue
        
        return all_results
    
    def _format_detailed_article_response(self, results: List[Dict], article_num: str, question: str) -> str:
        """Format detailed article response"""
        unique_results = self._deduplicate_results(results)
        
        response = ""
        response += f"Article {article_num}: Comprehensive Constitutional Analysis\n\n"
        
        # CONSTITUTIONAL TEXT SECTION
        response += "Constitutional Text\n\n"
        
        article_texts = [r for r in unique_results if r['type'] == 'article_text']
        if article_texts:
            best_text = article_texts[0]['text']
            formatted_text = self._format_constitutional_text(best_text)
            response += f"{formatted_text}\n\n"
        else:
            contexts = [r for r in unique_results if r['type'] == 'context']
            if contexts:
                response += f"While the exact text of Article {article_num} was not found, related constitutional context mentions:\n\n"
                formatted_context = self._format_constitutional_text(contexts[0]['text'][:500])
                response += f"{formatted_context}\n\n"
            else:
                response += f"Article {article_num} is referenced in constitutional documents.\n\n"
        
        # DETAILED ANALYSIS SECTION
        response += "Detailed Constitutional Analysis\n\n"
        
        analysis = self._create_detailed_article_analysis(unique_results, article_num)
        response += f"{analysis}\n\n"
        
        # KEY PROVISIONS SECTION
        response += "Key Legal Provisions\n\n"
        
        detailed_provisions = self._extract_detailed_provisions(unique_results)
        if detailed_provisions:
            for i, provision in enumerate(detailed_provisions, 1):
                response += f"{i}. {provision}\n"
            response += "\n"
        else:
            default_provisions = [
                "Establishes fundamental rights and duties",
                "Provides legal framework for constitutional protections",
                "Includes provisions for implementation and enforcement",
                "Subject to judicial interpretation and review"
            ]
            for i, provision in enumerate(default_provisions, 1):
                response += f"{i}. {provision}\n"
            response += "\n"
        
        # SIGNIFICANCE SECTION
        response += "Constitutional Significance\n\n"
        
        significance_points = self._extract_detailed_significance_points(unique_results)
        for point in significance_points:
            response += f"• {point}\n"
        response += "\n"
        
        # SOURCE DOCUMENTS
        response += "Source Documents\n\n"
        
        sources_summary = {}
        for result in unique_results[:3]:
            source = result['source']
            if source not in sources_summary:
                sources_summary[source] = {
                    'pages': set(),
                    'excerpt': result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                }
            
            if result['page'] != 'N/A':
                sources_summary[source]['pages'].add(result['page'])
        
        for source, data in sources_summary.items():
            clean_source = os.path.basename(source)
            response += f"• {clean_source}"
            
            if data['pages']:
                pages = sorted(list(data['pages']))
                response += f" (Pages: {', '.join(map(str, pages))})"
            
            response += f"\n  Excerpt: {data['excerpt']}\n\n"
        
        # RELATED ARTICLES
        related_articles = self._extract_related_articles(unique_results)
        if related_articles:
            response += "Related Constitutional Articles\n\n"
            response += f"This article is connected to: {', '.join(related_articles[:5])}\n\n"
        
        return response
    
    def _create_detailed_article_analysis(self, results: List[Dict], article_num: str) -> str:
        """Create detailed article analysis"""
        analysis_parts = []
        
        # Introduction
        analysis_parts.append(f"Article {article_num} of Pakistan's Constitution establishes important legal provisions that form part of the constitutional framework.")
        
        # Extract key information
        key_info = self._extract_article_information(results)
        if key_info:
            analysis_parts.append("\nThe article addresses several key aspects of constitutional law:")
            for info in key_info:
                analysis_parts.append(f"• {info}")
        
        # Legal implications
        analysis_parts.append("\nThis article has significant legal implications for:")
        implications = [
            "Protection of fundamental rights and freedoms",
            "Establishment of legal duties and obligations",
            "Regulation of government powers and authorities",
            "Provision of constitutional safeguards and remedies"
        ]
        for implication in implications:
            analysis_parts.append(f"• {implication}")
        
        # Practical application
        analysis_parts.append("\nIn practical application, this article:")
        applications = [
            "Provides basis for legal claims and challenges",
            "Guides judicial interpretation of constitutional principles",
            "Establishes parameters for legislative and executive actions",
            "Forms part of Pakistan's constitutional jurisprudence"
        ]
        for application in applications:
            analysis_parts.append(f"• {application}")
        
        return "\n".join(analysis_parts)
    
    def _generate_detailed_concept_response(self, question: str) -> str:
        """Generate detailed response for legal concept queries"""
        print(f"Analyzing legal concept: '{question}'")
        
        # Extract key terms
        key_terms = self._extract_key_terms(question)
        
        # Search for concept information
        search_results = self._search_concept_content(question, key_terms)
        
        if not search_results:
            return self._no_results_response(question)
        
        # Structure the response
        response = self._format_detailed_concept_response(search_results, question, key_terms)
        return response
    
    def _search_concept_content(self, question: str, key_terms: List[str]) -> List[Dict]:
        """Search for concept content in database"""
        all_results = []
        
        # Create optimized search queries
        search_queries = [
            question,
            *key_terms[:3],
            *[f"{term} constitution pakistan" for term in key_terms[:2]],
            *[f"constitutional {term}" for term in key_terms[:2]]
        ]
        
        for query in search_queries[:4]:
            try:
                docs = self.vector_store.similarity_search(query, k=5)
                
                for doc in docs:
                    content = doc.page_content
                    source = doc.metadata.get('source', 'Unknown Document')
                    page = doc.metadata.get('page', 'N/A')
                    
                    if self._is_table_of_contents(content):
                        continue
                    
                    relevance = self._calculate_relevance(content, key_terms)
                    
                    if relevance > 0.3:
                        all_results.append({
                            'text': content,
                            'source': source,
                            'page': page,
                            'type': 'concept_content',
                            'relevance_score': relevance
                        })
            except Exception as e:
                continue
        
        return all_results
    
    def _format_detailed_concept_response(self, results: List[Dict], question: str, key_terms: List[str]) -> str:
        """Format detailed concept response"""
        unique_results = self._deduplicate_results(results)
        unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        response = ""
        response += f"{question.title()}: Comprehensive Constitutional Analysis\n\n"
        
        # CONCEPT ANALYSIS SECTION
        response += "Concept Analysis\n\n"
        
        analysis = self._create_detailed_concept_analysis(unique_results, question)
        response += f"{analysis}\n\n"
        
        # CONSTITUTIONAL BASIS
        response += "Constitutional Basis\n\n"
        
        all_articles = set()
        for result in unique_results:
            articles = self._extract_articles_from_text(result['text'])
            for article in articles:
                all_articles.add(article)
        
        if all_articles:
            response += "This concept is addressed in the following constitutional articles:\n\n"
            sorted_articles = sorted(list(all_articles))
            groups = self._group_articles(sorted_articles)
            
            for group in groups[:5]:
                if len(group) > 1:
                    response += f"• Articles {group[0]}-{group[-1]}\n"
                else:
                    response += f"• Article {group[0]}\n"
            response += "\n"
        else:
            response += "This concept is referenced in constitutional documents without specific article mentions.\n\n"
        
        # KEY FEATURES
        response += "Key Constitutional Features\n\n"
        
        features = self._extract_detailed_features(unique_results)
        if features:
            for i, feature in enumerate(features, 1):
                response += f"{i}. {feature}\n"
            response += "\n"
        else:
            default_features = [
                "Forms part of fundamental rights framework",
                "Subject to constitutional interpretation and limitation",
                "Includes legal safeguards and protections",
                "Essential for democratic governance and rule of law",
                "Implemented through specific constitutional provisions"
            ]
            for i, feature in enumerate(default_features, 1):
                response += f"{i}. {feature}\n"
            response += "\n"
        
        # LEGAL FRAMEWORK
        response += "Legal Framework and Implications\n\n"
        
        legal_points = self._extract_legal_framework_points(unique_results)
        for point in legal_points:
            response += f"• {point}\n"
        response += "\n"
        
        # SOURCE DOCUMENTS
        response += "Source Documents\n\n"
        
        sources_data = {}
        for result in unique_results[:3]:
            source = result['source']
            if source not in sources_data:
                sources_data[source] = {
                    'pages': set(),
                    'excerpt': result['text'][:120] + "..." if len(result['text']) > 120 else result['text']
                }
            
            if result['page'] != 'N/A':
                sources_data[source]['pages'].add(result['page'])
        
        for source, data in sources_data.items():
            clean_source = os.path.basename(source)
            response += f"• {clean_source}"
            
            if data['pages']:
                pages = sorted(list(data['pages']))
                response += f" (Pages: {', '.join(map(str, pages[:2]))}"
                if len(pages) > 2:
                    response += f", and {len(pages)-2} more"
                response += ")"
            
            response += f"\n  Excerpt: {data['excerpt']}\n\n"
        
        return response
    
    def _create_detailed_concept_analysis(self, results: List[Dict], question: str) -> str:
        """Create detailed concept analysis"""
        analysis_parts = []
        
        # Introduction
        analysis_parts.append(f"The concept of '{question}' is a fundamental aspect of Pakistan's constitutional framework that addresses important legal principles and rights.")
        
        # Key aspects
        key_aspects = self._extract_concept_aspects(results)
        if key_aspects:
            analysis_parts.append("\nKey aspects of this concept include:")
            for aspect in key_aspects:
                analysis_parts.append(f"• {aspect}")
        
        # Constitutional role
        analysis_parts.append("\nWithin Pakistan's constitutional system, this concept:")
        roles = [
            "Establishes important legal principles and standards",
            "Provides framework for rights protection and enforcement",
            "Guides judicial interpretation of constitutional provisions",
            "Forms basis for legislative and policy development"
        ]
        for role in roles:
            analysis_parts.append(f"• {role}")
        
        # Practical significance
        analysis_parts.append("\nThe practical significance of this concept lies in:")
        significances = [
            "Protection of individual rights and freedoms",
            "Regulation of government powers and authorities",
            "Provision of legal remedies and safeguards",
            "Promotion of democratic governance and rule of law"
        ]
        for significance in significances:
            analysis_parts.append(f"• {significance}")
        
        return "\n".join(analysis_parts)
    
    def _generate_detailed_general_response(self, question: str) -> str:
        """Generate detailed response for general queries"""
        print(f"Researching: '{question}'")
        
        # Extract key terms
        key_terms = self._extract_key_terms(question)
        
        # Search for general information
        search_results = self._search_general_content(question, key_terms)
        
        if not search_results:
            return self._no_results_response(question)
        
        # Structure the response
        response = self._format_detailed_general_response(search_results, question, key_terms)
        return response
    
    def _search_general_content(self, question: str, key_terms: List[str]) -> List[Dict]:
        """Search for general content in database"""
        all_results = []
        
        # Create optimized query
        search_query = question
        if key_terms:
            search_query = f"{question} constitution"
        
        try:
            docs = self.vector_store.similarity_search(search_query, k=8)
            
            for doc in docs:
                content = doc.page_content
                source = doc.metadata.get('source', 'Unknown Document')
                page = doc.metadata.get('page', 'N/A')
                
                if self._is_table_of_contents(content):
                    continue
                
                relevance = self._calculate_relevance(content, key_terms)
                
                if relevance > 0.2:
                    all_results.append({
                        'text': content,
                        'source': source,
                        'page': page,
                        'type': 'general_content',
                        'relevance_score': relevance
                    })
        except Exception as e:
            print(f"Search error: {str(e)}")
        
        return all_results
    
    def _format_detailed_general_response(self, results: List[Dict], question: str, key_terms: List[str]) -> str:
        """Format detailed general response"""
        unique_results = self._deduplicate_results(results)
        unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        response = ""
        response += f"{question.title()}: Comprehensive Constitutional Research\n\n"
        
        # OVERVIEW SECTION
        response += "Constitutional Overview\n\n"
        
        overview = self._create_detailed_overview(unique_results, question)
        response += f"{overview}\n\n"
        
        # CONSTITUTIONAL PROVISIONS
        response += "Constitutional Provisions\n\n"
        
        all_articles = set()
        constitutional_points = []
        
        for result in unique_results[:4]:
            # Extract articles
            articles = self._extract_articles_from_text(result['text'])
            for article in articles:
                all_articles.add(article)
            
            # Extract constitutional points
            points = self._extract_constitutional_points(result['text'])
            constitutional_points.extend(points[:2])
        
        if constitutional_points:
            response += "The Constitution addresses this topic through several provisions:\n\n"
            for i, point in enumerate(constitutional_points[:4], 1):
                response += f"{i}. {point}\n"
            response += "\n"
        
        # KEY FINDINGS
        response += "Key Constitutional Findings\n\n"
        
        findings = self._extract_detailed_findings(unique_results)
        for finding in findings:
            response += f"• {finding}\n"
        response += "\n"
        
        # SOURCE DOCUMENTS
        response += "Source Documents\n\n"
        
        sources_info = {}
        for result in unique_results[:4]:
            source = result['source']
            if source not in sources_info:
                sources_info[source] = {
                    'pages': set(),
                    'excerpt': result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                }
            
            if result['page'] != 'N/A':
                sources_info[source]['pages'].add(result['page'])
        
        for source, data in sources_info.items():
            clean_source = os.path.basename(source)
            response += f"• {clean_source}"
            
            if data['pages']:
                pages = sorted(list(data['pages']))
                response += f" (Pages: {', '.join(map(str, pages))})"
            
            response += f"\n  Excerpt: {data['excerpt']}\n\n"
        
        # RELATED ARTICLES
        if all_articles:
            response += "Related Constitutional Articles\n\n"
            
            sorted_articles = sorted(list(all_articles))
            groups = self._group_articles(sorted_articles)
            
            article_list = []
            for group in groups[:5]:
                if len(group) > 1:
                    article_list.append(f"Articles {group[0]}-{group[-1]}")
                else:
                    article_list.append(f"Article {group[0]}")
            
            response += ', '.join(article_list) + "\n\n"
        
        return response
    
    def _create_detailed_overview(self, results: List[Dict], question: str) -> str:
        """Create detailed overview"""
        overview_parts = []
        
        # Introduction
        overview_parts.append(f"The topic of '{question}' is comprehensively addressed in Pakistan's Constitution through various articles and legal provisions.")
        
        # Key information
        key_info = self._extract_general_information(results)
        if key_info:
            overview_parts.append("\nKey constitutional aspects include:")
            for info in key_info:
                overview_parts.append(f"• {info}")
        
        # Constitutional framework
        overview_parts.append("\nThe constitutional framework for this topic includes:")
        framework_points = [
            "Specific articles establishing rights and duties",
            "Legal provisions for implementation and enforcement",
            "Judicial mechanisms for protection and remedy",
            "Constitutional safeguards and limitations"
        ]
        for point in framework_points:
            overview_parts.append(f"• {point}")
        
        # Importance
        overview_parts.append("\nThis topic is important because:")
        importance_points = [
            "It addresses fundamental aspects of constitutional law",
            "It establishes legal standards and protections",
            "It guides government actions and policies",
            "It forms part of Pakistan's constitutional heritage"
        ]
        for point in importance_points:
            overview_parts.append(f"• {point}")
        
        return "\n".join(overview_parts)
    
    # Helper Methods for Detailed Content
    
    def _format_constitutional_text(self, text: str) -> str:
        """Format constitutional text with proper paragraph structure"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Break into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into paragraphs
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            if len(current_paragraph) >= 2 and len(sentence) > 80:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Format paragraphs
        formatted_text = ""
        for paragraph in paragraphs:
            formatted_text += paragraph + "\n\n"
        
        return formatted_text.strip()
    
    def _extract_article_information(self, results: List[Dict]) -> List[str]:
        """Extract article information"""
        info = []
        
        for result in results[:3]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 6 and any(keyword in sentence.lower() 
                    for keyword in ['provides', 'establishes', 'guarantees', 'prohibits', 'requires']):
                    
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in info:
                        info.append(clean_sentence[:120])
        
        return info[:5]
    
    def _extract_detailed_provisions(self, results: List[Dict]) -> List[str]:
        """Extract detailed provisions"""
        provisions = []
        
        for result in results:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 6 and any(keyword in sentence.lower() 
                    for keyword in ['shall', 'may', 'must', 'right to', 'entitled to', 'duty to']):
                    
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in provisions:
                        if clean_sentence[0].islower():
                            clean_sentence = clean_sentence[0].upper() + clean_sentence[1:]
                        provisions.append(clean_sentence[:130])
                        
                        if len(provisions) >= 6:
                            break
        
        return provisions[:6]
    
    def _extract_detailed_significance_points(self, results: List[Dict]) -> List[str]:
        """Extract detailed significance points"""
        points = []
        
        for result in results[:3]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 5 and len(sentence) < 150:
                    if any(indicator in sentence.lower() for indicator in 
                          ['important', 'essential', 'significant', 'crucial', 'vital']):
                        
                        clean_sentence = sentence.strip()
                        if clean_sentence and clean_sentence not in points:
                            points.append(clean_sentence)
        
        # Add default points if none found
        if not points:
            points = [
                "Establishes fundamental constitutional principles",
                "Provides legal framework for rights protection",
                "Subject to judicial interpretation and review",
                "Integral to Pakistan's constitutional democracy",
                "Forms basis for legal claims and remedies"
            ]
        
        return points[:5]
    
    def _extract_concept_aspects(self, results: List[Dict]) -> List[str]:
        """Extract concept aspects"""
        aspects = []
        
        for result in results[:3]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 6 and len(sentence) < 200:
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in aspects:
                        aspects.append(clean_sentence[:150])
        
        return aspects[:5]
    
    def _extract_detailed_features(self, results: List[Dict]) -> List[str]:
        """Extract detailed features"""
        features = []
        
        for result in results[:3]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 7 and any(keyword in sentence.lower() 
                    for keyword in ['feature', 'characteristic', 'aspect', 'element']):
                    
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in features:
                        features.append(clean_sentence[:130])
        
        return features[:6]
    
    def _extract_legal_framework_points(self, results: List[Dict]) -> List[str]:
        """Extract legal framework points"""
        points = []
        
        for result in results[:2]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 6 and any(keyword in sentence.lower() 
                    for keyword in ['framework', 'legal', 'constitutional', 'provision']):
                    
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in points:
                        points.append(clean_sentence[:110])
        
        return points[:4]
    
    def _extract_general_information(self, results: List[Dict]) -> List[str]:
        """Extract general information"""
        info = []
        
        for result in results[:3]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 5 and len(sentence) < 180:
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in info:
                        info.append(clean_sentence[:120])
        
        return info[:5]
    
    def _extract_constitutional_points(self, text: str) -> List[str]:
        """Extract constitutional points from text"""
        points = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(sentence.split()) > 5 and any(keyword in sentence.lower() 
                for keyword in ['constitution', 'article', 'right', 'duty', 'power', 'shall']):
                
                clean_sentence = sentence.strip()
                if clean_sentence and clean_sentence not in points:
                    points.append(clean_sentence[:140])
        
        return points[:3]
    
    def _extract_detailed_findings(self, results: List[Dict]) -> List[str]:
        """Extract detailed findings from results"""
        findings = []
        
        for result in results[:3]:
            text = result['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                if len(sentence.split()) > 6 and len(sentence) < 150:
                    clean_sentence = sentence.strip()
                    if clean_sentence and clean_sentence not in findings:
                        findings.append(clean_sentence)
        
        # Add default findings if none found
        if not findings:
            findings = [
                "Constitutional provisions establish comprehensive legal framework",
                "Multiple articles address different aspects of this topic",
                "Judicial interpretation plays crucial role in application",
                "Constitutional safeguards ensure proper implementation",
                "Legal principles guide government actions and policies"
            ]
        
        return findings[:6]
    
    # Basic Helper Methods
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        key_terms = []
        for word in words:
            if word not in stop_words:
                key_terms.append(word)
        
        return list(set(key_terms))[:6]
    
    def _calculate_relevance(self, content: str, key_terms: List[str]) -> float:
        """Calculate relevance score"""
        if not key_terms:
            return 0.5
        
        content_lower = content.lower()
        matches = sum(1 for term in key_terms if term in content_lower)
        
        return matches / len(key_terms) if key_terms else 0
    
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
                
                return marker_count >= 2 and not self._is_table_of_contents(content)
        
        return False
    
    def _mentions_article(self, content: str, article_num: str) -> bool:
        """Check if content mentions article"""
        patterns = [
            rf'article\s+{article_num}',
            rf'art\.\s+{article_num}',
            rf'\b{article_num}\b',
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
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
    
    def _extract_context(self, content: str, article_num: str) -> str:
        """Extract context around article mention"""
        pattern = rf'.{{0,100}}(?:article\s+{article_num}|art\.\s+{article_num}).{{0,200}}'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        
        return content[:300].strip()
    
    def _extract_articles_from_text(self, text: str) -> List[str]:
        """Extract all article numbers from text"""
        articles = set()
        
        for pattern in self.article_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                articles.add(match.upper())
        
        return sorted(list(articles))
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results"""
        unique_results = []
        seen_content = set()
        
        for result in results:
            text = result.get('text', '')
            if text:
                normalized = text.lower()[:150]
                normalized = re.sub(r'\s+', ' ', normalized).strip()
                
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    unique_results.append(result)
        
        return unique_results
    
    def _extract_related_articles(self, results: List[Dict]) -> List[str]:
        """Extract related articles from results"""
        articles = set()
        
        for result in results:
            text = result.get('text', '')
            extracted = self._extract_articles_from_text(text)
            for article in extracted:
                articles.add(article)
        
        return sorted(list(articles))
    
    def _group_articles(self, articles: List[str]) -> List[List[str]]:
        """Group articles numerically"""
        groups = []
        current_group = []
        
        for article in sorted(articles, key=lambda x: int(re.sub(r'[A-Z]+', '', x) or 0)):
            if re.match(r'^\d+[A-Z]*$', article):
                try:
                    num_part = re.sub(r'[A-Z]+', '', article)
                    if num_part:
                        num = int(num_part)
                        if not current_group or num == int(re.sub(r'[A-Z]+', '', current_group[-1])) + 1:
                            current_group.append(article)
                        else:
                            if current_group:
                                groups.append(current_group)
                            current_group = [article]
                except:
                    if current_group:
                        groups.append(current_group)
                        current_group = []
                    groups.append([article])
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                groups.append([article])
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
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
    
    def _not_found_response(self, article_num: str) -> str:
        """Not found response"""
        response = f"Article {article_num}: Not Found\n\n"
        response += f"Article {article_num} was not found in the constitutional database.\n\n"
        response += "Suggestions:\n"
        response += "• Verify the article number\n"
        response += "• Search for related constitutional concepts\n"
        response += "• Check general constitutional topics\n"
        response += "• Ask about specific rights or government structures\n"
        
        return response
    
    def _no_results_response(self, question: str) -> str:
        """No results response"""
        response = f"No Constitutional Results Found\n\n"
        response += f"Query: '{question}'\n\n"
        response += "Suggestions for Better Results:\n"
        response += "• Use specific constitutional terminology\n"
        response += "• Search by Article number (e.g., 'Article 19', 'Article 25A')\n"
        response += "• Ask about fundamental rights or government structure\n"
        response += "• Use keywords: freedom, right, parliament, president, judiciary\n"
        
        return response
    
    def _error_response(self, question: str, error_msg: str) -> str:
        """Error response"""
        response = f"System Error\n\n"
        response += f"Query: {question}\n"
        response += f"Error: {error_msg[:100]}\n\n"
        response += "Please ensure the database is properly initialized and documents are processed.\n"
        
        return response

# Global instance for compatibility
assistant = DetailedConstitutionAssistant()

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
    print("\nPakistan Constitution Assistant - Detailed Version")
    print("Paragraph-Based Detailed Responses\n")
    
    assistant.initialize()
    
    print(f"\nTry these queries for detailed responses:")
    print("1. Article 19 - Freedom of speech")
    print("2. What are fundamental rights in Pakistan?")
    print("3. Explain the structure of Parliament")
    print("4. How are judges appointed in Pakistan?")
    print("5. What is the right to education under Article 25A?")
    print("\nOr say hello:")
    print("• Hello / Hi / Good morning")
    print("• Help")
    print("• Thank you")
    print("\n")
    
    # Interactive mode
    while True:
        try:
            question = input("Your constitutional question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Constitutional wisdom awaits your return!")
                break
            
            if not question:
                continue
            
            answer = answer_question(question)
            print(f"\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Come back for more constitutional insights!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")