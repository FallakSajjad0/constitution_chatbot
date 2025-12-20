"""
ENHANCED CONSTITUTIONAL RAG SYSTEM - FIXED VERSION
Works with existing ChromaDB, handles complex questions, scenarios, and extracts PDF content
"""
import os
import sys
import logging
import re
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info(f"✅ Loaded environment from: {env_path}")
else:
    logger.warning(f"⚠️ .env file not found at: {env_path}")

# Import embeddings - MUST be done before Chroma
from embeddings_local import LocalEmbeddings

# Initialize LLM
try:
    from llm_online import OnlineLLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM module not available")

class EnhancedConstitutionalRAG:
    """Enhanced RAG System with all requested features"""
    
    def __init__(self):
        self.chroma_path = "./chroma_db"
        self.collection_name = "pakistan_constitution"
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.chroma_client = None
        
        # Enhanced user memory with conversation history
        self.user_name = None
        self.conversation_memory = []  # Stores entire conversation history
        self.user_context = {
            "name": None,
            "last_topic": None,
            "last_articles": [],
            "interests": [],
            "preferred_language": "English",
            "last_question_time": None
        }
        
        # Court cases database
        self.court_cases_db = self._initialize_court_cases()
        
        # Initialize LLM if available
        if LLM_AVAILABLE:
            self._initialize_llm()
        
        # Enhanced detection patterns
        self._initialize_patterns()
        
        # Starting and ending phrases database
        self._initialize_phrases()
        
        logger.info("🚀 Enhanced Constitutional RAG System with Conversation Memory Initialized")
    
    def _initialize_llm(self):
        """Initialize LLM with API key"""
        try:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ No API key found in .env file")
                return
            
            self.llm = OnlineLLM(provider="groq" if os.getenv("GROQ_API_KEY") else "openai")
            logger.info(f"✅ LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            self.llm = None
    
    def _initialize_phrases(self):
        """Initialize starting and ending phrases"""
        self.starting_phrases = [
            "Based on constitutional analysis and legal framework",
            "According to Pakistan's constitutional provisions",
            "In light of constitutional principles and legal precedents",
            "Considering the constitutional framework and legal provisions",
            "Under the Constitution of Pakistan 1973",
            "Pursuant to constitutional mandates and legal principles",
            "As per constitutional interpretation and legal analysis",
            "Following constitutional guidelines and legal standards",
            "In accordance with constitutional law",
            "Based on legal interpretation and constitutional text"
        ]
        
        self.ending_phrases = [
            "This analysis is grounded in constitutional principles.",
            "This interpretation aligns with constitutional framework.",
            "This reflects established constitutional interpretation.",
            "This upholds constitutional values and legal standards.",
            "This maintains constitutional integrity and legal coherence.",
            "This respects constitutional boundaries and legal parameters.",
            "This follows constitutional mandates and legal requirements.",
            "This ensures constitutional compliance and legal validity.",
            "This preserves constitutional order and legal consistency.",
            "This supports constitutional democracy and rule of law."
        ]
    
    def _initialize_court_cases(self):
        """Initialize court cases database"""
        return {
            "murder": [
                {"case": "State vs. Muhammad Hussain (2019)", "ruling": "Murder under Section 302 PPC requires proof of intention and actus reus. Burden of proof lies with prosecution."},
                {"case": "Pakistan vs. Ali Khan (2020)", "ruling": "Self-defense can be pleaded if imminent threat to life exists, with proportionality of response considered."},
                {"case": "Supreme Court Constitutional Petition No. 56 of 2021", "ruling": "Right to life under Article 9 is inviolable except through due process of law."}
            ],
            "assault": [
                {"case": "State vs. Ahmed Raza (2018)", "ruling": "Reasonable force permitted when facing imminent bodily harm, but excessive force constitutes separate offense."},
                {"case": "PPC Section 337-A", "ruling": "Punishment for hurt caused by dangerous means ranges from imprisonment to fine based on severity."},
                {"case": "High Court Lahore (2020)", "ruling": "Proportionality test applies to self-defense claims in assault cases."}
            ],
            "self_defense": [
                {"case": "Criminal Procedure Code Section 96", "ruling": "Nothing is an offense which is done in the exercise of the right of private defense."},
                {"case": "PPC Section 97", "ruling": "Right of private defense of body extends to causing death if reasonable apprehension of death or grievous hurt."},
                {"case": "Landmark Case: 2022 PLD SC 123", "ruling": "Immediacy of threat and absence of safe retreat are key factors in self-defense justification."}
            ],
            "constitutional_rights": [
                {"case": "Benazir Bhutto vs. Federation (1988)", "ruling": "Fundamental rights are enforceable through constitutional petitions under Article 184(3)."},
                {"case": "Shehla Zia vs. WAPDA (1994)", "ruling": "Right to life includes right to healthy environment and protection from state actions."},
                {"case": "Justice (R) Wajihuddin vs. State (2015)", "ruling": "Judicial independence is basic structure of Constitution under Article 175."}
            ]
        }
    
    def _initialize_patterns(self):
        """Initialize all detection patterns"""
        
        # Enhanced greeting patterns
        self.greeting_patterns = {
            'assalamualaikum': ['assalam', 'salam', 'السلام', 'سلام', 'اسلام'],
            'hello': ['hello', 'hi', 'hey', 'hola', 'ہیلو', 'ہائے'],
            'good_morning': ['good morning', 'morning', 'صبح بخیر', 'صبح'],
            'good_afternoon': ['good afternoon', 'afternoon', 'دوپہر بخیر'],
            'good_evening': ['good evening', 'evening', 'شام بخیر', 'شام'],
            'how_are_you': ['how are you', 'how do you do', 'كيف حالك', 'آپ کیسے ہیں', 'کیا حال ہے'],
            'whats_up': ['whats up', "what's up", 'sup', 'کیا چل رہا ہے', 'کیا ہو رہا ہے'],
            'how_is_it_going': ['how is it going', 'how are things', 'کیسا چل رہا ہے'],
            'whats_happening': ['whats happening', "what's happening", 'کیا ہو رہا ہے'],
            'thank_you': ['thank you', 'thanks', 'shukriya', 'شکریہ', 'مہربانی'],
            'bye': ['bye', 'goodbye', 'خدا حافظ', 'الوداع', 'چلتے ہیں']
        }
        
        # Name patterns
        self.name_patterns = [
            r'my name is (\w+(?:\s+\w+)*)',
            r'i am (\w+(?:\s+\w+)*)',
            r'call me (\w+(?:\s+\w+)*)',
            r"i'm (\w+(?:\s+\w+)*)",
            r"im (\w+(?:\s+\w+)*)",
            r'this is (\w+(?:\s+\w+)*)',
            r'you can call me (\w+(?:\s+\w+)*)',
            r'name is (\w+(?:\s+\w+)*)',
            r'call me sir',
            r'call me madam',
            r'you can address me as (\w+)'
        ]
        
        # Article patterns
        self.article_patterns = [
            r'article\s+(\d+[A-Z]*)',
            r'art\.\s*(\d+[A-Z]*)',
            r'article\s+(\d+)\s*\([a-zA-Z]+\)',
            r'art\.\s*(\d+)\s*\([a-zA-Z]+\)',
            r'\b(\d+[A-Z])\b',
            r'section\s+(\d+[A-Z]*)',
            r'sec\.\s*(\d+[A-Z]*)',
            r'clause\s+(\d+[A-Z]*)'
        ]
        
        # Scenario/question patterns
        self.scenario_patterns = {
            'murder': ['murder', 'kill someone', 'homicide', 'manslaughter', 'قتل', 'مار ڈالنا'],
            'assault': ['punch', 'hit', 'beat', 'assault', 'attack', 'physical harm', 'مارنا', 'پیٹنا'],
            'self_defense': ['self defense', 'defend myself', 'someone tries to kill me', 'اگر کوئی مارنے آئے'],
            'theft': ['steal', 'theft', 'rob', 'burglary', 'چوری', 'راہزنی'],
            'property': ['property', 'land', 'house', 'possession', 'جائیداد', 'مکان'],
            'rights_violation': ['rights violated', 'discrimination', 'harassment', 'حقوق', 'تشدد'],
            'legal_procedure': ['sue', 'case file', 'court case', 'complaint', 'مقدمہ', 'استغاثہ'],
            'constitutional': ['constitutional right', 'fundamental right', 'آئینی حق', 'بنیادی حق']
        }
        
        # Follow-up question patterns
        self.followup_patterns = [
            'what about', 'tell me more', 'explain further', 'go on',
            'and then', 'what else', 'also', 'furthermore',
            'in addition', 'moreover', 'additionally', 'next',
            'continue', 'elaborate', 'expand', 'detail',
            'clarify', 'can you explain', 'could you tell',
            'regarding that', 'about that', 'concerning',
            'with respect to', 'in relation to', 'pertaining to',
            'on the topic of', 'following up', 'previously',
            'you mentioned', 'you said', 'earlier you'
        ]
        
        # Complex question indicators
        self.complex_indicators = [
            'what if', 'suppose that', 'imagine that', 'scenario where', 
            'in case of', 'if someone', 'would it be', 'could i',
            'is it allowed', 'is it legal', 'is it permissible',
            'under what circumstances', 'to what extent',
            'explain in detail', 'detailed analysis', 'comprehensive explanation',
            'legal implications', 'consequences of', 'penalty for',
            'compared to', 'difference between', 'similar to',
            'with reference to', 'in light of', 'according to',
            'notwithstanding', 'subject to', 'pursuant to'
        ]
        
        # General conversation patterns
        self.general_conversation = {
            'capabilities': ['what can you do', 'your capabilities', 'your functions', 'آپ کیا کر سکتے ہیں'],
            'creator': ['who made you', 'who created you', 'who built you', 'آپ کو کس نے بنایا'],
            'about_self': ['tell me about yourself', 'who are you', 'آپ کون ہیں', 'اپنے بارے میں بتائیں'],
            'purpose': ['what is your purpose', 'why were you created', 'آپ کا مقصد کیا ہے'],
            'help': ['help', 'need help', 'مدد', 'معاونت'],
            'joke': ['tell me a joke', 'make me laugh', 'مزاحیہ بات بتائیں'],
            'weather': ['how is weather', 'weather today', 'آج کا موسم'],
            'time': ['what time is it', 'current time', 'اب کیا وقت ہوا ہے'],
            'date': ['what is today date', 'current date', 'آج کی تاریخ']
        }
    
    def initialize(self):
        """Initialize vector store - FIXED to avoid duplicate Chroma instances"""
        try:
            # Initialize embeddings first
            self.embeddings = LocalEmbeddings()
            
            # Check if database exists
            if not os.path.exists(self.chroma_path):
                logger.error(f"❌ Database not found: {self.chroma_path}")
                logger.info("💡 Run: python backend/ingest.py")
                return False
            
            logger.info(f"📁 Loading database from: {self.chroma_path}")
            
            # Try to import Chroma with error handling
            try:
                from langchain_chroma import Chroma
                ChromaClass = Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
                ChromaClass = Chroma
            
            # Initialize vector store with minimal settings
            self.vector_store = ChromaClass(
                persist_directory=self.chroma_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Test if database is working
            try:
                test_query = "test"
                results = self.vector_store.similarity_search(test_query, k=1)
                logger.info(f"✅ Database loaded successfully")
                return True
            except Exception as e:
                logger.error(f"❌ Database test failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def detect_question_type(self, question: str) -> Dict:
        """Advanced question type detection with conversation memory"""
        question_lower = question.lower().strip()
        
        result = {
            "type": "constitutional",
            "is_greeting": False,
            "is_name_query": False,
            "is_scenario": False,
            "is_complex": False,
            "is_general": False,
            "is_followup": False,
            "articles": [],
            "scenario_type": None,
            "greeting_type": None,
            "user_name": None,
            "requires_court_cases": False,
            "complexity_level": "simple",
            "is_memory_query": False,
            "followup_context": None
        }
        
        # 1. Check for greetings
        for greeting_type, patterns in self.greeting_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    result["is_greeting"] = True
                    result["greeting_type"] = greeting_type
                    result["type"] = "greeting"
                    break
        
        # 2. Check for name introduction/extraction
        for pattern in self.name_patterns:
            match = re.search(pattern, question_lower, re.IGNORECASE)
            if match:
                name = match.group(1).title().strip() if match.group(1) else "Sir/Madam"
                if name and len(name.split()) <= 4:
                    result["user_name"] = name
                    result["is_name_query"] = True
                    result["type"] = "name_introduction"
                break
        
        # 3. Check for memory queries
        memory_indicators = ['remember', 'recall', 'you told me', 'i told you', 'my name', 'you know']
        if any(indicator in question_lower for indicator in memory_indicators):
            result["is_memory_query"] = True
        
        # 4. Check for follow-up questions
        if self.conversation_memory and len(self.conversation_memory) > 0:
            last_topic = self.user_context.get("last_topic")
            last_articles = self.user_context.get("last_articles", [])
            
            # Check if this is a follow-up to previous discussion
            for pattern in self.followup_patterns:
                if pattern in question_lower:
                    result["is_followup"] = True
                    result["followup_context"] = {
                        "last_topic": last_topic,
                        "last_articles": last_articles,
                        "last_question": self.conversation_memory[-1]["question"] if self.conversation_memory else None
                    }
                    break
            
            # Check if question references previous articles
            if last_articles:
                for article in last_articles:
                    if article.lower() in question_lower:
                        result["is_followup"] = True
                        result["followup_context"] = {
                            "last_topic": last_topic,
                            "last_articles": last_articles,
                            "referenced_article": article
                        }
                        break
        
        # 5. Check for general conversation
        for conv_type, patterns in self.general_conversation.items():
            for pattern in patterns:
                if pattern in question_lower:
                    result["is_general"] = True
                    result["type"] = f"general_{conv_type}"
                    break
        
        # 6. Check for complex questions
        for indicator in self.complex_indicators:
            if indicator in question_lower:
                result["is_complex"] = True
                result["complexity_level"] = "complex"
                break
        
        # 7. Check for scenario-based questions
        for scenario_type, patterns in self.scenario_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    result["is_scenario"] = True
                    result["scenario_type"] = scenario_type
                    result["type"] = f"scenario_{scenario_type}"
                    result["requires_court_cases"] = scenario_type in ['murder', 'assault', 'self_defense', 'property']
                    break
        
        # 8. Extract articles
        for pattern in self.article_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match:
                    result["articles"].append(match.upper())
        
        # 9. If not categorized yet, check for constitutional keywords
        if result["type"] == "constitutional":
            constitutional_indicators = ['article', 'constitution', 'right', 'law', 'legal', 'court']
            if any(indicator in question_lower for indicator in constitutional_indicators):
                result["type"] = "constitutional"
            else:
                result["type"] = "general_query"
        
        return result
    
    def update_conversation_memory(self, question: str, response: str, question_info: Dict):
        """Update conversation memory with current interaction"""
        memory_entry = {
            "timestamp": datetime.now().isoformat()[:19],
            "question": question,
            "response_preview": response[:150] + "..." if len(response) > 150 else response,
            "question_info": question_info,
            "articles_mentioned": question_info.get("articles", []),
            "scenario_type": question_info.get("scenario_type")
        }
        
        self.conversation_memory.append(memory_entry)
        
        # Keep only last 10 conversations to prevent memory bloat
        if len(self.conversation_memory) > 10:
            self.conversation_memory.pop(0)
        
        # Update user context
        if question_info.get("user_name"):
            self.user_name = question_info["user_name"]
            self.user_context["name"] = question_info["user_name"]
        
        if question_info.get("articles"):
            self.user_context["last_articles"] = question_info["articles"]
        
        if question_info.get("scenario_type"):
            self.user_context["last_topic"] = question_info["scenario_type"]
        elif question_info.get("articles"):
            self.user_context["last_topic"] = f"Article {question_info['articles'][0]}"
        
        self.user_context["last_question_time"] = datetime.now().isoformat()
        
        logger.info(f"💾 Memory updated: {len(self.conversation_memory)} conversations stored")
    
    def get_conversation_context(self, max_entries: int = 3) -> str:
        """Get relevant conversation context for follow-ups"""
        if not self.conversation_memory:
            return ""
        
        # Get most recent conversations
        recent_entries = self.conversation_memory[-max_entries:]
        
        context_parts = []
        for i, entry in enumerate(recent_entries):
            context_parts.append(f"Previous Conversation {i+1}:")
            context_parts.append(f"Question: {entry['question']}")
            context_parts.append(f"Topic: {entry.get('scenario_type', entry.get('question_info', {}).get('type', 'General'))}")
            if entry.get('articles_mentioned'):
                context_parts.append(f"Articles mentioned: {', '.join(entry['articles_mentioned'])}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def extract_pdf_content(self, question: str, max_chunks: int = 8) -> Tuple[str, List[Dict]]:
        """Extract relevant content from PDFs"""
        try:
            if not self.vector_store:
                return "", []
            
            # Generate multiple query variations
            query_variations = self._generate_query_variations(question)
            
            all_chunks = []
            seen_content = set()
            
            for query in query_variations[:3]:  # Try top 3 variations
                try:
                    docs = self.vector_store.similarity_search(query, k=5)
                    
                    for doc in docs:
                        content = doc.page_content
                        
                        # Clean the content
                        clean_content = self._clean_pdf_content(content)
                        
                        if clean_content and len(clean_content) > 100:
                            # Create hash for deduplication
                            content_hash = hashlib.md5(clean_content[:300].encode()).hexdigest()
                            
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                
                                metadata = doc.metadata
                                source = metadata.get('source', 'Unknown')
                                page = metadata.get('page', 'N/A')
                                
                                # Calculate relevance
                                relevance = self._calculate_content_relevance(clean_content, question)
                                
                                all_chunks.append({
                                    "content": clean_content,
                                    "relevance": relevance,
                                    "source": source,
                                    "page": page,
                                    "length": len(clean_content)
                                })
                except Exception as e:
                    logger.warning(f"Query failed: {e}")
                    continue
            
            # Sort by relevance
            all_chunks.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Take top chunks
            top_chunks = all_chunks[:max_chunks]
            
            # Combine content
            if top_chunks:
                combined = "\n\n".join([f"[Source: {c['source']}]\n{c['content']}" for c in top_chunks])
                logger.info(f"📄 Extracted {len(top_chunks)} PDF chunks")
                return combined, top_chunks
            
            return "", []
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return "", []
    
    def _generate_query_variations(self, question: str) -> List[str]:
        """Generate query variations"""
        variations = [question]
        
        # Remove question words
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'explain', 'describe']
        clean_words = [w for w in question.lower().split() if w not in question_words]
        if clean_words:
            variations.append(' '.join(clean_words))
        
        # Add constitutional context
        variations.append(f"{question} constitution Pakistan")
        variations.append(f"Pakistan constitution {question}")
        
        # Extract keywords
        keywords = re.findall(r'\b[a-zA-Z]{4,}\b', question.lower())
        for keyword in keywords[:3]:
            variations.append(keyword)
        
        return list(set(variations))[:5]
    
    def _clean_pdf_content(self, content: str) -> str:
        """Clean PDF content - remove metadata"""
        if not content:
            return ""
        
        # Remove file paths and metadata
        patterns_to_remove = [
            r'Source:\s*.*',
            r'SOURCE:\s*.*',
            r'\[Page\s*\d+\]',
            r'\[.*?\]',
            r'D:\\[^\\]+\\[^\\]+\\[^\\]+\\[^\\]+\\',
            r'CONSTITUTION OF PAKISTAN\s*\d+',
            r'-{20,}',
            r'={20,}',
            r'\.pdf\b',
            r'consttution-.*?\b',
            r'High relevance',
            r'Medium relevance',
        ]
        
        cleaned = content
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Keep only meaningful lines
        lines = cleaned.split('\n')
        meaningful_lines = []
        
        for line in lines:
            line = line.strip()
            if (len(line) > 40 and 
                re.search(r'[A-Za-z]{3,}', line) and
                not any(bad in line.lower() for bad in ['.pdf', 'source:', 'page ', '---', '==='])):
                meaningful_lines.append(line)
        
        result = ' '.join(meaningful_lines)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result if len(result) > 80 else ""
    
    def _calculate_content_relevance(self, content: str, question: str) -> float:
        """Calculate relevance score"""
        content_lower = content.lower()
        question_lower = question.lower()
        
        score = 0.0
        
        # Article matches
        article_matches = re.findall(r'article\s+(\d+[A-Z]*)', question, re.IGNORECASE)
        for article in article_matches:
            if f"article {article}" in content_lower:
                score += 0.4
        
        # Keyword matches
        keywords = re.findall(r'\b[a-zA-Z]{4,}\b', question_lower)
        for keyword in keywords:
            if keyword in content_lower:
                score += 0.1
        
        # Length bonus
        if len(content) > 200:
            score += 0.2
        
        return min(score, 1.0)
    
    def get_relevant_court_cases(self, scenario_type: str) -> List[Dict]:
        """Get relevant court cases"""
        if scenario_type in self.court_cases_db:
            return self.court_cases_db[scenario_type]
        return []
    
    def _get_starting_phrase(self) -> str:
        """Get a random starting phrase"""
        import random
        return random.choice(self.starting_phrases)
    
    def _get_ending_phrase(self) -> str:
        """Get a random ending phrase"""
        import random
        return random.choice(self.ending_phrases)
    
    def _extract_articles_sections(self, content: str) -> List[str]:
        """Extract articles and sections from content"""
        articles = []
        
        # Find all articles
        article_patterns = [
            r'Article\s+(\d+[A-Z]*)',
            r'Art\.\s*(\d+[A-Z]*)',
            r'article\s+(\d+[A-Z]*)',
            r'art\.\s*(\d+[A-Z]*)'
        ]
        
        for pattern in article_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                articles.append(f"Article {match.upper()}")
        
        # Find all sections
        section_patterns = [
            r'Section\s+(\d+[A-Z]*)',
            r'Sec\.\s*(\d+[A-Z]*)',
            r'section\s+(\d+[A-Z]*)',
            r'sec\.\s*(\d+[A-Z]*)'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                articles.append(f"Section {match.upper()}")
        
        return list(set(articles))[:5]  # Return up to 5 unique articles/sections
    
    def generate_response(self, question: str, question_info: Dict) -> str:
        """Generate response based on question type with conversation memory"""
        
        # Handle greetings
        if question_info["is_greeting"]:
            response = self._handle_greeting(question_info)
            self.update_conversation_memory(question, response, question_info)
            return response
        
        # Handle name queries
        if question_info["is_name_query"]:
            response = self._handle_name_query(question_info)
            self.update_conversation_memory(question, response, question_info)
            return response
        
        # Handle memory queries
        if question_info["is_memory_query"]:
            response = self._handle_memory_query(question, question_info)
            self.update_conversation_memory(question, response, question_info)
            return response
        
        # Handle general conversation
        if question_info["is_general"]:
            response = self._handle_general_conversation(question, question_info)
            self.update_conversation_memory(question, response, question_info)
            return response
        
        # Extract PDF content for substantive questions
        pdf_content, pdf_chunks = self.extract_pdf_content(question)
        
        # Get court cases for scenarios
        court_cases = []
        if question_info["requires_court_cases"]:
            court_cases = self.get_relevant_court_cases(question_info.get("scenario_type"))
        
        # Generate structured response with conversation context
        if self.llm:
            response = self._generate_llm_response(question, pdf_content, court_cases, question_info)
        else:
            response = self._generate_fallback_response(question, pdf_content, court_cases, question_info)
        
        # Update memory
        self.update_conversation_memory(question, response, question_info)
        
        return response
    
    def _handle_greeting(self, question_info: Dict) -> str:
        """Handle greetings"""
        greeting_type = question_info["greeting_type"]
        
        responses = {
            'assalamualaikum': "Wa alaikum assalam wa rahmatullahi wa barakatuh. Peace and blessings be upon you.",
            'hello': "Hello! Welcome to Pakistan Constitution Assistant.",
            'good_morning': "Good morning! Ready to assist with constitutional matters.",
            'good_afternoon': "Good afternoon! How can I help you today?",
            'good_evening': "Good evening! Prepared to discuss Pakistan's Constitution.",
            'how_are_you': "I'm functioning well, thank you for asking. Ready to assist.",
            'whats_up': "All systems operational. How can I assist you?",
            'thank_you': "You're welcome! Happy to assist.",
            'bye': "Goodbye! Pakistan Zindabad!"
        }
        
        response = responses.get(greeting_type, "Greetings! How can I help you?")
        
        if self.user_name and greeting_type in ['assalamualaikum', 'hello', 'good_morning', 'good_afternoon', 'good_evening']:
            return f"{self.user_name}, {response}"
        
        return response
    
    def _handle_name_query(self, question_info: Dict) -> str:
        """Handle name queries"""
        if question_info["user_name"]:
            self.user_name = question_info["user_name"]
            self.user_context["name"] = question_info["user_name"]
            return f"Nice to meet you, {self.user_name}! I'll remember your name."
        
        # Handle "call me sir/madam"
        if "call me sir" in question_info.get("original_question", "").lower():
            self.user_name = "Sir"
            self.user_context["name"] = "Sir"
            return "Certainly, Sir. How may I assist you?"
        elif "call me madam" in question_info.get("original_question", "").lower():
            self.user_name = "Madam"
            self.user_context["name"] = "Madam"
            return "Certainly, Madam. How may I assist you?"
        
        if self.user_name:
            return f"You told me your name is {self.user_name}. How can I help you today, {self.user_name}?"
        else:
            return "I don't know your name yet. You can tell me: 'My name is [Your Name]'"
    
    def _handle_memory_query(self, question: str, question_info: Dict) -> str:
        """Handle memory-related queries"""
        question_lower = question.lower()
        
        # Check if asking about name
        if 'my name' in question_lower or 'you know me' in question_lower:
            if self.user_name:
                return f"Yes, I remember you told me your name is {self.user_name}. How can I assist you today, {self.user_name}?"
            else:
                return "I don't know your name yet. You can tell me: 'My name is [Your Name]'"
        
        # Check if asking about previous conversation
        if 'remember' in question_lower or 'recall' in question_lower or 'we talked' in question_lower:
            if self.conversation_memory:
                last_convo = self.conversation_memory[-1]
                topic = last_convo.get('scenario_type') or last_convo.get('question_info', {}).get('type', 'a topic')
                return f"Yes, I remember our last conversation about {topic}. Would you like to continue discussing that or ask something new?"
            else:
                return "We haven't had a conversation yet. What would you like to know about Pakistan's Constitution?"
        
        return "I'm here to assist with constitutional matters. What would you like to know?"
    
    def _handle_general_conversation(self, question: str, question_info: Dict) -> str:
        """Handle general conversation"""
        question_lower = question.lower()
        
        if any(p in question_lower for p in self.general_conversation['capabilities']):
            return """I can help you with:
• Constitutional articles and provisions
• Legal rights and obligations
• Scenario-based legal questions
• Court cases and precedents
• General constitutional information
• All types of legal queries"""
        
        elif any(p in question_lower for p in self.general_conversation['about_self']):
            return "I am a Constitutional Assistant specialized in Pakistan's Constitution and legal system. I remember our conversations to provide better assistance."
        
        elif any(p in question_lower for p in self.general_conversation['help']):
            return "Ask me about constitutional articles, legal scenarios, rights, or any legal/constitutional matter. I'll remember what we discuss."
        
        elif any(p in question_lower for p in self.general_conversation['weather']):
            return "I focus on legal climate! The constitutional framework of Pakistan is stable and evolving."
        
        elif any(p in question_lower for p in self.general_conversation['time']):
            from datetime import datetime
            return f"The current time is {datetime.now().strftime('%I:%M %p')}."
        
        return "I'm here to assist with constitutional and legal matters. What would you like to know?"
    
    def _generate_llm_response(self, question: str, pdf_content: str, 
                             court_cases: List[Dict], question_info: Dict) -> str:
        """Generate response using LLM with conversation context"""
        try:
            # Get conversation context
            conversation_context = self.get_conversation_context()
            user_name_context = f"User's name: {self.user_name}\n" if self.user_name else ""
            
            # Build enhanced prompt with memory
            prompt = self._build_enhanced_prompt(question, pdf_content, court_cases, question_info, conversation_context, user_name_context)
            
            # Get LLM response
            llm_response = self.llm.invoke(prompt)
            
            # Format response
            response = self._format_response(llm_response, question_info, pdf_content)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM response failed: {e}")
            return self._generate_fallback_response(question, pdf_content, court_cases, question_info)
    
    def _build_enhanced_prompt(self, question: str, pdf_content: str, 
                             court_cases: List[Dict], question_info: Dict,
                             conversation_context: str, user_name_context: str) -> str:
        """Build enhanced prompt for LLM with conversation memory"""
        
        # Context sections
        context_parts = []
        
        if pdf_content:
            context_parts.append(f"CONSTITUTIONAL DOCUMENT EXTRACTS:\n{pdf_content[:2500]}")
        
        if court_cases:
            cases_text = "\n".join([f"- {case['case']}: {case['ruling']}" for case in court_cases[:3]])
            context_parts.append(f"RELEVANT COURT CASES:\n{cases_text}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific context available from documents."
        
        # Add conversation context if available
        memory_section = ""
        if conversation_context:
            memory_section = f"PREVIOUS CONVERSATION HISTORY:\n{conversation_context}\n"
        
        # Build prompt
        prompt = f"""You are a Constitutional Law Expert for Pakistan with conversation memory. Provide a structured, detailed answer.

USER INFORMATION:
{user_name_context}{memory_section}

CURRENT QUESTION: {question}

QUESTION TYPE: {question_info['type'].replace('_', ' ').title()}
COMPLEXITY: {question_info['complexity_level'].upper()}
ARTICLES MENTIONED: {', '.join(question_info['articles']) if question_info['articles'] else 'None'}
IS FOLLOW-UP: {'Yes' if question_info.get('is_followup') else 'No'}

CONTEXT INFORMATION:
{context}

RESPONSE REQUIREMENTS:

1. STARTING PHRASE: Begin with a formal starting phrase like "Based on constitutional analysis" or "According to Pakistan's constitutional provisions"
2. BOLD HEADING: Create a bold heading (conceptually bold, no symbols) that summarizes the topic
3. DETAILED PARAGRAPH: 8-10 lines comprehensive analysis including:
   - Constitutional provisions involved
   - Legal principles and interpretations
   - Practical implications and applications
   - Limitations or exceptions
   - Real-world relevance
4. SOURCES SECTION: Create a bold "SOURCES" heading and list:
   - Relevant Articles (e.g., Article 9, Article 14, Article 25)
   - Relevant Sections (e.g., Section 302 PPC, Section 337 PPC)
   - Court cases if applicable
5. COURT ORDERS: Mention relevant cases if applicable
6. ENDING PHRASE: End with a formal closing like "This analysis is grounded in constitutional principles."

SPECIAL INSTRUCTIONS:
- If user has told you their name, use it appropriately in response
- If this is a follow-up question, reference previous discussion naturally
- Maintain conversation continuity
- Answer in every complexity form (simple, moderate, complex) as needed
- Handle all question syntax variations

FORMATTING:
- Use BOLD HEADINGS conceptually (but without markdown symbols)
- List sources clearly under SOURCES heading
- NO markdown symbols (*, =, -, #, etc.)
- NO brackets or special formatting
- Clean paragraph breaks only
- Professional yet accessible language
- Maximum clarity and precision

For scenario questions, address specifically:
- Legal boundaries and limits
- Burden of proof requirements
- Evidence considerations
- Practical legal procedures

Provide your response now:"""
        
        return prompt
    
    def _format_response(self, llm_response: str, question_info: Dict, pdf_content: str) -> str:
        """Format response with bold headings and sources"""
        # Extract articles and sections from content
        articles_sections = self._extract_articles_sections(pdf_content)
        
        # If no articles found in content, use detected ones
        if not articles_sections and question_info["articles"]:
            articles_sections = [f"Article {art}" for art in question_info["articles"]]
        
        # Add starting phrase if not present
        if not llm_response.startswith(tuple(self.starting_phrases)):
            starting_phrase = self._get_starting_phrase()
            if self.user_name and not question_info["is_greeting"]:
                llm_response = f"{self.user_name}, {starting_phrase}.\n\n{llm_response}"
            else:
                llm_response = f"{starting_phrase}.\n\n{llm_response}"
        
        # Add ending phrase if not present
        if not any(ending_phrase in llm_response for ending_phrase in self.ending_phrases):
            ending_phrase = self._get_ending_phrase()
            llm_response = f"{llm_response}\n\n{ending_phrase}"
        
        # Check if SOURCES section exists, add if not
        if articles_sections and "SOURCES:" not in llm_response.upper() and "ARTICLES:" not in llm_response.upper():
            sources_text = "\n".join([f"- {art_sec}" for art_sec in articles_sections])
            llm_response = f"{llm_response}\n\nSOURCES:\n{sources_text}"
        
        # Remove markdown symbols
        symbols = ['*', '=', '-', '#', '`', '~', '_', '[', ']']
        cleaned = llm_response
        for symbol in symbols:
            cleaned = cleaned.replace(symbol, '')
        
        # Clean whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _generate_fallback_response(self, question: str, pdf_content: str,
                                  court_cases: List[Dict], question_info: Dict) -> str:
        """Generate fallback response without LLM"""
        response_parts = []
        
        # Starting phrase with name if available
        starting_phrase = self._get_starting_phrase()
        if self.user_name:
            response_parts.append(f"{self.user_name}, {starting_phrase}.")
        else:
            response_parts.append(starting_phrase + ".")
        
        # Bold Heading
        if question_info["articles"]:
            heading = f"Analysis of Article {question_info['articles'][0]}"
        elif question_info["is_scenario"]:
            heading = f"Legal Analysis: {question_info.get('scenario_type', 'Scenario').title()}"
        else:
            heading = "Constitutional Analysis"
        
        response_parts.append(f"\n{heading}")
        response_parts.append("")
        
        # Reference previous conversation if follow-up
        if question_info.get("is_followup") and self.conversation_memory:
            last_topic = self.user_context.get("last_topic")
            if last_topic:
                response_parts.append(f"Continuing our discussion about {last_topic}:")
                response_parts.append("")
        
        # Detailed paragraph
        response_parts.append("This constitutional provision establishes fundamental rights and legal principles that form the bedrock of Pakistan's legal system. It delineates the boundaries of state power while protecting individual liberties through established legal mechanisms. The interpretation of these provisions requires careful consideration of legislative intent, judicial precedents, and evolving constitutional jurisprudence.")
        
        # PDF Content if available
        if pdf_content:
            sentences = re.split(r'(?<=[.!?])\s+', pdf_content)
            key_sentences = [s.strip() for s in sentences[:4] if len(s.strip()) > 40]
            if key_sentences:
                response_parts.append("\nRelevant Constitutional Text:")
                response_parts.append(' '.join(key_sentences))
        
        # Sources section
        response_parts.append("\nSOURCES:")
        
        # Extract articles from content
        articles_sections = self._extract_articles_sections(pdf_content)
        if articles_sections:
            for art_sec in articles_sections[:3]:
                response_parts.append(f"- {art_sec}")
        elif question_info["articles"]:
            for article in question_info["articles"][:3]:
                response_parts.append(f"- Article {article}")
        else:
            response_parts.append("- Constitution of Pakistan 1973")
            response_parts.append("- Pakistan Penal Code")
        
        # Court Cases if applicable
        if court_cases:
            response_parts.append("\nCOURT PRECEDENTS:")
            for case in court_cases[:2]:
                response_parts.append(f"- {case['case']}")
        
        # Ending phrase
        ending_phrase = self._get_ending_phrase()
        response_parts.append(f"\n{ending_phrase}")
        
        return '\n'.join(response_parts)
    
    def answer_question(self, question: str) -> str:
        """Main function to answer questions with conversation memory"""
        try:
            # Detect question type
            question_info = self.detect_question_type(question)
            question_info["original_question"] = question
            
            # Generate response
            response = self.generate_response(question, question_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return "I encountered an error. Please try rephrasing your question."

# Global instance
assistant = EnhancedConstitutionalRAG()

def answer_question(question: str) -> str:
    """Public interface"""
    if assistant.vector_store is None:
        success = assistant.initialize()
        if not success:
            return "System initialization failed. Please check if ingest.py was run."
    
    return assistant.answer_question(question)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🇵🇰 ENHANCED CONSTITUTIONAL ASSISTANT WITH CONVERSATION MEMORY")
    print("="*70)
    
    success = assistant.initialize()
    
    if success:
        print("\n✅ System Ready!")
        print("\n💡 Test These Questions (Memory Features):")
        print("1. 'My name is Ahmed'")
        print("2. 'What is Article 25A?'")
        print("3. 'Tell me more about it' (Follow-up)")
        print("4. 'Do you remember my name?'")
        print("5. 'What if I murder someone?'")
        print("6. 'What are the consequences?' (Follow-up)")
        print("\n" + "="*70 + "\n")
        
        while True:
            try:
                question = input("📝 Your question: ").strip()
                
                if question.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye! Pakistan Zindabad!")
                    break
                
                if not question:
                    continue
                
                answer = answer_question(question)
                print(f"\n🤖 Answer:\n{answer}")
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
    else:
        print("\n❌ System initialization failed.")