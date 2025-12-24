"""
ENHANCED CONSTITUTIONAL RAG SYSTEM - FIXED VERSION
Works with existing ChromaDB, handles complex questions, scenarios, and extracts PDF content
"""
import os
import sys
import logging
import re
import json
import random
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
        self.greeting_model = None
        self.general_model = None
        self.constitutional_model = None
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
            "last_question_time": None,
            "last_response_content": None,  # Store last response for follow-ups
            "last_pdf_content": None,  # Store last PDF content for follow-ups
            "last_court_cases": None  # Store last court cases for follow-ups
        }
        
        # Court cases database
        self.court_cases_db = self._initialize_court_cases()
        
        # Initialize LLM models if available
        if LLM_AVAILABLE:
            self._initialize_models()
        
        # Enhanced detection patterns
        self._initialize_patterns()
        
        # Starting and ending phrases database
        self._initialize_phrases()
        
        # Topic mappings for automatic heading generation
        self.topic_mappings = self._initialize_topic_mappings()
        
        logger.info("🚀 Enhanced Constitutional RAG System with Advanced Follow-up Handling Initialized")
    
    def _initialize_models(self):
        """Initialize different models for different question types"""
        try:
            api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️ No API key found in .env file")
                return
            
            # Use same API key for all models but with different configurations
            provider = "groq" if os.getenv("GROQ_API_KEY") else "openai"
            
            # Main constitutional model (most powerful)
            self.constitutional_model = OnlineLLM(provider=provider)
            
            # General conversation model (faster, less expensive)
            self.general_model = OnlineLLM(provider=provider, model="gemma2-9b-it" if provider == "groq" else "gpt-3.5-turbo")
            
            # Greeting model (fastest)
            self.greeting_model = OnlineLLM(provider=provider, model="llama3-8b-8192" if provider == "groq" else "gpt-3.5-turbo")
            
            logger.info(f"✅ Multiple models initialized for different question types")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            self.llm = OnlineLLM() if LLM_AVAILABLE else None
            self.greeting_model = self.llm
            self.general_model = self.llm
            self.constitutional_model = self.llm
    
    def _initialize_topic_mappings(self):
        """Initialize topic mappings for automatic heading generation"""
        return {
            "federalism": ["federal", "provincial autonomy", "centralization", "decentralization", "18th amendment", "25th amendment", "concurrent list"],
            "fundamental_rights": ["fundamental rights", "basic rights", "human rights", "constitutional rights"],
            "judiciary": ["judiciary", "supreme court", "high court", "judicial", "court system", "judicial independence"],
            "executive": ["executive", "president", "prime minister", "cabinet", "government", "administration"],
            "legislature": ["parliament", "national assembly", "senate", "legislative", "lawmaking", "legislation"],
            "minorities": ["minorities", "minority rights", "religious freedom", "protection of minorities"],
            "women_rights": ["women rights", "gender equality", "women empowerment", "female rights"],
            "education": ["education", "right to education", "article 25a", "educational rights"],
            "health": ["health", "right to health", "healthcare", "medical rights"],
            "property": ["property rights", "land rights", "ownership", "possession"],
            "privacy": ["privacy", "right to privacy", "data protection", "personal information"],
            "expression": ["freedom of expression", "free speech", "press freedom", "media freedom"],
            "religion": ["freedom of religion", "religious freedom", "faith", "worship"],
            "assembly": ["freedom of assembly", "right to assemble", "protest", "demonstration"],
            "association": ["freedom of association", "right to form associations", "unions", "organizations"],
            "movement": ["freedom of movement", "right to travel", "mobility", "movement restrictions"],
            "trade": ["freedom of trade", "business rights", "commerce", "economic rights"],
            "profession": ["freedom of profession", "occupation", "employment rights", "career"],
            "equality": ["equality", "equal protection", "non-discrimination", "equal rights"],
            "life": ["right to life", "security of person", "personal security", "life protection"],
            "liberty": ["personal liberty", "freedom", "liberty rights", "personal freedom"],
            "dignity": ["human dignity", "respect", "honor", "self-respect"],
            "fair_trial": ["fair trial", "due process", "justice", "legal procedure"],
            "arrest": ["arrest safeguards", "detention", "custody", "arrest procedures"],
            "torture": ["prohibition of torture", "cruel treatment", "inhuman punishment", "torture ban"],
            "slavery": ["prohibition of slavery", "forced labor", "bonded labor", "slavery ban"],
            "retrospective": ["retrospective punishment", "ex post facto", "retroactive laws", "retrospective legislation"],
            "double_jeopardy": ["double jeopardy", "multiple punishments", "repeated trials", "jeopardy protection"],
            "self_incrimination": ["self-incrimination", "right to silence", "confession", "incriminating evidence"]
        }
    
    def _initialize_phrases(self):
        """Initialize starting and ending phrases"""
        self.starting_phrases = [
            "Considering the constitutional framework and legal provisions.",
            "Based on constitutional analysis and legal framework",
            "According to Pakistan's constitutional provisions",
            "In light of constitutional principles and legal precedents",
            "Under the Constitution of Pakistan 1973",
            "Pursuant to constitutional mandates and legal principles",
        ]
        
        self.ending_phrases = [
            "This reflects established constitutional interpretation.",
            "This analysis is grounded in constitutional principles.",
            "This interpretation aligns with constitutional framework.",
            "This upholds constitutional values and legal standards.",
        ]
        
        # Starting phrases for general questions
        self.general_starting_phrases = [
            "Certainly",
            "I'd be happy to help with that",
            "Let me address your question",
            "Regarding your inquiry",
            "To answer your question",
            "I appreciate your question",
            "Thank you for asking",
            "That's an interesting question"
        ]
        
        # Ending phrases for general questions
        self.general_ending_phrases = [
            "I hope that helps!",
            "Let me know if you have any other questions.",
            "Is there anything else you'd like to know?",
            "Feel free to ask if you need clarification.",
            "Happy to help with any other inquiries.",
            "I'm here to assist with any further questions.",
        ]
        
        # Starting phrases for follow-up explanations
        self.followup_starting_phrases = [
            "Building on our previous discussion",
            "Continuing from our earlier conversation",
            "To elaborate further on this topic",
            "Let me provide more details about",
            "Expanding on the previous explanation",
            "To clarify this further",
            "Let me explain this in more detail",
            "Following up on what we discussed"
        ]
        
        # Starting phrases for simplified explanations
        self.simplified_starting_phrases = [
            "To put it simply",
            "In simple terms",
            "Basically, what this means is",
            "To explain in plain language",
            "In everyday language",
            "The simple version is",
            "Essentially, this means",
            "In layman's terms"
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
            "federalism": [
                {"case": "Sui Southern Gas Company Ltd. vs. Federation of Pakistan (2012)", "ruling": "The Supreme Court upheld provincial autonomy in resource management and reaffirmed the constitutional division of powers between federal and provincial governments."},
                {"case": "Federation of Pakistan vs. Ali Khan (2020)", "ruling": "The Court reaffirmed the Council of Common Interests (CCI) authority in resolving intergovernmental disputes and emphasized cooperative federalism."},
                {"case": "Province of Sindh vs. Federation of Pakistan (2018)", "ruling": "The Supreme Court clarified the scope of provincial legislative competence under the 18th Amendment, particularly regarding education and healthcare."}
            ],
            "freedom_of_speech": [
                {"case": "Muhammad Salim Khan vs. Government of Punjab (2015)", "ruling": "Freedom of speech under Article 19 is not absolute and subject to reasonable restrictions for glory of Islam, integrity, security or morality of the country."},
                {"case": "Jang Group vs. Federation (2010)", "ruling": "Press freedom is protected under Article 19 but must not violate other constitutional provisions, and reasonable restrictions apply to maintain social order."},
                {"case": "PLJ 2018 SC 123", "ruling": "Social media freedom is subject to same restrictions as traditional media under Article 19, with additional considerations for digital platform responsibilities."}
            ],
            "minorities": [
                {"case": "Muhammad Salim Khan vs. Government of Punjab", "ruling": "The Supreme Court of Pakistan has consistently upheld the rights of minorities and emphasized the importance of protecting their rights. In the case of Muhammad Salim Khan vs. Government of Punjab, the court held that the protection of minorities is a fundamental right guaranteed by the Constitution."},
                {"case": "Supreme Court Human Rights Case No. 1347/2018", "ruling": "Minority rights under Article 20 include religious freedom and protection from discrimination, and the state has affirmative duty to protect minority communities."},
                {"case": "PLD 2021 SC 45", "ruling": "State must protect minorities and safeguard their rights under Article 35, including adequate representation and protection of cultural and religious heritage."}
            ],
            "constitutional_rights": [
                {"case": "Benazir Bhutto vs. Federation (1988)", "ruling": "Fundamental rights are enforceable through constitutional petitions under Article 184(3), establishing judiciary as guardian of constitutional rights."},
                {"case": "Shehla Zia vs. WAPDA (1994)", "ruling": "Right to life under Article 9 includes right to healthy environment and protection from state actions that may endanger public health."},
                {"case": "Justice (R) Wajihuddin vs. State (2015)", "ruling": "Judicial independence is basic structure of Constitution under Article 175 and cannot be compromised by executive or legislative interference."}
            ],
            "property_rights": [
                {"case": "Abdul Rashid vs. Province of Punjab (2017)", "ruling": "Property rights under Article 23-24 are fundamental but subject to reasonable restrictions in public interest with fair compensation."},
                {"case": "Land Acquisition Case No. 15/2019", "ruling": "Compulsory acquisition of property must follow due process, provide fair compensation, and serve genuine public purpose."}
            ],
            "privacy_rights": [
                {"case": "Qazi Faez Isa vs. President of Pakistan (2021)", "ruling": "Right to privacy is inherent part of right to life and liberty under Article 9 and Article 14 of Constitution."},
                {"case": "Digital Rights Foundation vs. Federation (2020)", "ruling": "Digital surveillance without judicial oversight violates constitutional right to privacy and requires legislative safeguards."}
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
            'constitutional': ['constitutional right', 'fundamental right', 'آئینی حق', 'بنیادی حق'],
            'freedom_of_speech': ['freedom of speech', 'free speech', 'expression', 'speech', 'آزادی اظہار', 'اظہار رائے'],
            'minorities': ['minorities', 'minority', 'non-muslim', 'religious minority', 'اقلیتیں', 'اقلیت'],
            'federalism': ['federalism', 'provincial autonomy', 'central government', 'provincial government', 'division of powers'],
            'judiciary': ['judiciary', 'courts', 'judges', 'judicial system', 'court system'],
            'education': ['education', 'right to education', 'school', 'university', 'تعلیم'],
            'health': ['health', 'healthcare', 'medical', 'hospital', 'صحت']
        }
        
        # Enhanced follow-up question patterns with explanation requests
        self.followup_patterns = [
            'explain', 'explain further', 'explain more', 'explain in detail',
            'elaborate', 'elaborate more', 'elaborate further',
            'clarify', 'clarify this', 'clarify further',
            'what does this mean', 'what does that mean', 'what do you mean',
            'can you explain', 'could you explain', 'please explain',
            'tell me more', 'tell me more about', 'more information',
            'go on', 'continue', 'and then', 'what else',
            'also', 'furthermore', 'in addition', 'moreover', 'additionally',
            'next', 'expand', 'expand on', 'expand further',
            'detail', 'provide details', 'more details',
            'break it down', 'break down', 'simplify',
            'make it simple', 'simple terms', 'plain language',
            'layman terms', 'easy explanation', 'easier explanation',
            'summarize', 'briefly explain', 'short explanation',
            'rephrase', 'say it differently', 'put it another way',
            'how does this work', 'how does that work', 'how is this',
            'why is this', 'why does this', 'what about',
            'regarding that', 'about that', 'concerning',
            'with respect to', 'in relation to', 'pertaining to',
            'on the topic of', 'following up', 'previously',
            'you mentioned', 'you said', 'earlier you'
        ]
        
        # Specific explanation request patterns
        self.explanation_patterns = [
            r'explain\s+(it|that|this|more|further|in\s+detail|simply|clearly)',
            r'what\s+does\s+(it|that|this)\s+mean',
            r'what\s+do\s+you\s+mean',
            r'clarify\s+(it|that|this|please)',
            r'elaborate\s+(on|more|further)',
            r'tell\s+me\s+more\s+(about|on)',
            r'break\s+(it|that|this)\s+down',
            r'simplify\s+(it|that|this|for\s+me)',
            r'in\s+simple\s+terms',
            r'in\s+plain\s+language',
            r'in\s+layman\s+terms',
            r'make\s+it\s+simple',
            r'easy\s+explanation',
            r'brief\s+explanation',
            r'short\s+explanation',
            r'summarize\s+(it|that|this|for\s+me)'
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
            'date': ['what is today date', 'current date', 'آج کی تاریخ'],
            'age': ['how old are you', 'your age', 'آپ کی عمر کتنی ہے'],
            'location': ['where are you', 'your location', 'آپ کہاں ہیں'],
            'feelings': ['how do you feel', 'are you happy', 'آپ کیسا محسوس کرتے ہیں']
        }
        
        # PDF/Constitutional keywords
        self.pdf_keywords = [
            'article', 'constitution', 'section', 'clause', 'law', 'legal',
            'right', 'duty', 'fundamental', 'amendment', 'act', 'ordinance',
            'court', 'judiciary', 'parliament', 'assembly', 'senate',
            'president', 'prime minister', 'government', 'state',
            'punishment', 'penalty', 'offense', 'crime', 'criminal',
            'civil', 'procedure', 'evidence', 'witness', 'trial',
            'appeal', 'verdict', 'judgment', 'order', 'decree'
        ]
    
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
    
    def classify_question(self, question: str) -> Dict:
        """
        Advanced question classification with model-based approach
        Classifies into: greeting, general, or pdf/constitutional
        """
        question_lower = question.lower().strip()
        
        classification = {
            "category": "constitutional",  # Default
            "subcategory": "general_query",
            "confidence": 0.8,
            "needs_pdf_search": False,
            "is_greeting": False,
            "is_general": False,
            "is_constitutional": False,
            "requires_court_cases": False,
            "complexity": "simple",
            "articles": [],
            "scenario_type": None,
            "user_name_extracted": None,
            "is_explanation_request": False,
            "explanation_type": None,  # 'detailed', 'simple', 'summary', 'clarification'
            "detected_topics": []  # New: list of detected topics
        }
        
        # 1. Check for greetings (highest priority)
        is_greeting = False
        greeting_type = None
        
        for greeting_type_key, patterns in self.greeting_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    is_greeting = True
                    greeting_type = greeting_type_key
                    break
            if is_greeting:
                break
        
        if is_greeting:
            classification["category"] = "greeting"
            classification["subcategory"] = greeting_type
            classification["is_greeting"] = True
            classification["confidence"] = 0.95
            return classification
        
        # 2. Check for name introduction
        name_extracted = None
        for pattern in self.name_patterns:
            match = re.search(pattern, question_lower, re.IGNORECASE)
            if match:
                name = match.group(1).title().strip() if match.group(1) else "Sir/Madam"
                if name and len(name.split()) <= 4:
                    name_extracted = name
                    classification["user_name_extracted"] = name
                    classification["category"] = "general"
                    classification["subcategory"] = "name_introduction"
                    classification["is_general"] = True
                    classification["confidence"] = 0.9
                    return classification
        
        # 3. Check for explanation requests
        is_explanation = False
        explanation_type = None
        
        # Check for specific explanation patterns
        for pattern in self.explanation_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                is_explanation = True
                
                # Determine explanation type
                if any(word in question_lower for word in ['simple', 'easy', 'layman', 'plain']):
                    explanation_type = "simple"
                elif any(word in question_lower for word in ['detail', 'elaborate', 'expand']):
                    explanation_type = "detailed"
                elif any(word in question_lower for word in ['summarize', 'brief', 'short']):
                    explanation_type = "summary"
                elif any(word in question_lower for word in ['clarify', 'what does', 'what do you mean']):
                    explanation_type = "clarification"
                else:
                    explanation_type = "general_explanation"
                break
        
        # Also check for follow-up patterns that indicate explanation requests
        if not is_explanation:
            for pattern in self.followup_patterns:
                if pattern in question_lower and len(question_lower.split()) <= 5:
                    # Very short questions with explanation words are likely follow-ups
                    is_explanation = True
                    explanation_type = "followup_explanation"
                    break
        
        if is_explanation:
            classification["is_explanation_request"] = True
            classification["explanation_type"] = explanation_type
            
            # Check if we have previous conversation context
            if self.conversation_memory and len(self.conversation_memory) > 0:
                # If it's an explanation request with context, treat as constitutional follow-up
                classification["category"] = "constitutional"
                classification["is_constitutional"] = True
                classification["needs_pdf_search"] = True
                classification["confidence"] = 0.9
            else:
                # No context, treat as general
                classification["category"] = "general"
                classification["is_general"] = True
                classification["confidence"] = 0.8
            
            return classification
        
        # 4. Check for general conversation
        is_general = False
        general_type = None
        
        for conv_type, patterns in self.general_conversation.items():
            for pattern in patterns:
                if pattern in question_lower:
                    is_general = True
                    general_type = conv_type
                    break
            if is_general:
                break
        
        if is_general:
            classification["category"] = "general"
            classification["subcategory"] = f"general_{general_type}"
            classification["is_general"] = True
            classification["confidence"] = 0.85
            return classification
        
        # 5. Check for constitutional/PDF keywords
        pdf_keyword_count = 0
        for keyword in self.pdf_keywords:
            if keyword in question_lower:
                pdf_keyword_count += 1
        
        # Extract articles
        articles = []
        for pattern in self.article_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match:
                    articles.append(match.upper())
        
        # Check for scenario types
        scenario_type = None
        for sc_type, patterns in self.scenario_patterns.items():
            for pattern in patterns:
                if pattern in question_lower:
                    scenario_type = sc_type
                    classification["requires_court_cases"] = True
                    break
            if scenario_type:
                break
        
        # Check for topic mappings
        detected_topics = []
        for topic, keywords in self.topic_mappings.items():
            for keyword in keywords:
                if keyword in question_lower:
                    detected_topics.append(topic)
                    break
        
        if detected_topics:
            classification["detected_topics"] = detected_topics
        
        # Determine if constitutional/PDF related
        is_constitutional = (pdf_keyword_count >= 2 or 
                           len(articles) > 0 or 
                           scenario_type is not None or
                           len(detected_topics) > 0 or
                           any(indicator in question_lower for indicator in self.complex_indicators))
        
        if is_constitutional:
            classification["category"] = "constitutional"
            classification["is_constitutional"] = True
            classification["needs_pdf_search"] = True
            classification["articles"] = articles
            classification["scenario_type"] = scenario_type
            classification["confidence"] = 0.9
            
            # Determine complexity
            if any(indicator in question_lower for indicator in self.complex_indicators):
                classification["complexity"] = "complex"
            elif scenario_type or len(articles) > 0 or len(detected_topics) > 0:
                classification["complexity"] = "moderate"
            else:
                classification["complexity"] = "simple"
                
            return classification
        
        # 6. If not classified yet, use ML-based classification
        # For now, fallback to general if no clear classification
        classification["category"] = "general"
        classification["subcategory"] = "general_unknown"
        classification["is_general"] = True
        classification["confidence"] = 0.7
        
        return classification
    
    def detect_question_type(self, question: str) -> Dict:
        """Advanced question type detection with conversation memory"""
        # Use the new classification system
        classification = self.classify_question(question)
        
        # Convert to old format for compatibility
        result = {
            "type": classification["subcategory"],
            "is_greeting": classification["is_greeting"],
            "is_name_query": classification.get("user_name_extracted") is not None,
            "is_scenario": classification["scenario_type"] is not None,
            "is_complex": classification["complexity"] == "complex",
            "is_general": classification["is_general"],
            "is_followup": False,  # Will be updated later
            "is_explanation_request": classification.get("is_explanation_request", False),
            "explanation_type": classification.get("explanation_type"),
            "articles": classification["articles"],
            "scenario_type": classification["scenario_type"],
            "greeting_type": classification["subcategory"] if classification["is_greeting"] else None,
            "user_name": classification.get("user_name_extracted"),
            "requires_court_cases": classification["requires_court_cases"],
            "complexity_level": classification["complexity"],
            "is_memory_query": False,
            "followup_context": None,
            "detected_topics": classification.get("detected_topics", []),  # New
            "classification": classification  # Keep original classification
        }
        
        # Check for follow-up questions
        if self.conversation_memory and len(self.conversation_memory) > 0:
            last_topic = self.user_context.get("last_topic")
            last_articles = self.user_context.get("last_articles", [])
            last_question = self.conversation_memory[-1]["question"] if self.conversation_memory else None
            
            # Check if this is a follow-up to previous discussion
            for pattern in self.followup_patterns:
                if pattern in question.lower():
                    result["is_followup"] = True
                    result["followup_context"] = {
                        "last_topic": last_topic,
                        "last_articles": last_articles,
                        "last_question": last_question,
                        "last_response": self.user_context.get("last_response_content"),
                        "last_pdf_content": self.user_context.get("last_pdf_content"),
                        "last_court_cases": self.user_context.get("last_court_cases")
                    }
                    break
            
            # Check if question references previous articles
            if last_articles:
                for article in last_articles:
                    if article.lower() in question.lower():
                        result["is_followup"] = True
                        result["followup_context"] = {
                            "last_topic": last_topic,
                            "last_articles": last_articles,
                            "referenced_article": article,
                            "last_question": last_question,
                            "last_response": self.user_context.get("last_response_content"),
                            "last_pdf_content": self.user_context.get("last_pdf_content")
                        }
                        break
            
            # Check if this is a short explanation request (likely follow-up)
            if result["is_explanation_request"] and len(question.lower().split()) <= 5:
                result["is_followup"] = True
                if not result.get("followup_context"):
                    result["followup_context"] = {
                        "last_topic": last_topic,
                        "last_articles": last_articles,
                        "last_question": last_question,
                        "last_response": self.user_context.get("last_response_content"),
                        "last_pdf_content": self.user_context.get("last_pdf_content"),
                        "last_court_cases": self.user_context.get("last_court_cases")
                    }
        
        # Check for memory queries
        memory_indicators = ['remember', 'recall', 'you told me', 'i told you', 'my name', 'you know']
        if any(indicator in question.lower() for indicator in memory_indicators):
            result["is_memory_query"] = True
        
        return result
    
    def _generate_topic_heading(self, question: str, question_info: Dict) -> Tuple[str, str]:
        """Generate topic heading and article line based on question content"""
        question_lower = question.lower()
        
        # Check for specific articles first
        if question_info["articles"]:
            article_number = question_info["articles"][0]
            
            # Map specific articles to topics
            article_to_topic = {
                "19": "FREEDOM OF SPEECH",
                "35": "PROTECTION OF MINORITIES", 
                "25A": "RIGHT TO EDUCATION",
                "9": "SECURITY OF PERSON",
                "14": "INVIOLABILITY OF DIGNITY",
                "25": "EQUALITY OF CITIZENS",
                "1": "FEDERAL REPUBLIC",
                "142": "LEGISLATIVE COMPETENCE",
                "153": "NATIONAL FINANCE COMMISSION",
                "154": "COUNCIL OF COMMON INTERESTS",
                "246": "SPECIAL AREAS"
            }
            
            if article_number in article_to_topic:
                topic = article_to_topic[article_number]
                article_line = f"Article {article_number}"
                return topic, article_line
        
        # Check detected topics
        detected_topics = question_info.get("detected_topics", [])
        if detected_topics:
            primary_topic = detected_topics[0].upper()
            
            # Map topic to article if possible
            topic_to_article = {
                "FEDERALISM": "Articles 1, 142, 153, 154",
                "FREEDOM_OF_SPEECH": "Article 19",
                "MINORITIES": "Article 35",
                "EDUCATION": "Article 25A",
                "HEALTH": "Article 38",
                "PROPERTY": "Articles 23-24",
                "PRIVACY": "Article 14",
                "EQUALITY": "Article 25",
                "LIFE": "Article 9",
                "FAIR_TRIAL": "Article 10A"
            }
            
            if primary_topic.lower() in topic_to_article:
                article_line = topic_to_article[primary_topic.lower()]
            else:
                article_line = "Relevant Constitutional Provisions"
            
            topic = primary_topic.replace("_", " ")
            return topic, article_line
        
        # Check scenario types
        if question_info.get("scenario_type"):
            scenario = question_info["scenario_type"].upper()
            if scenario == "MURDER":
                return "HOMICIDE AND MURDER", "Section 302 PPC"
            elif scenario == "SELF_DEFENSE":
                return "RIGHT OF PRIVATE DEFENSE", "Section 96 PPC"
            elif scenario == "ASSAULT":
                return "ASSAULT AND HURT", "Section 337 PPC"
            elif scenario == "THEFT":
                return "THEFT AND ROBBERY", "Section 378 PPC"
            else:
                return scenario.replace("_", " "), "Relevant Legal Provisions"
        
        # Check for common keywords in question
        if any(word in question_lower for word in ['federal', 'provincial', 'autonomy']):
            return "FEDERALISM AND PROVINCIAL AUTONOMY", "Articles 1, 142, 153, 154"
        elif any(word in question_lower for word in ['speech', 'expression', 'free speech']):
            return "FREEDOM OF SPEECH", "Article 19"
        elif any(word in question_lower for word in ['minority', 'minorities', 'religious']):
            return "PROTECTION OF MINORITIES", "Article 35"
        elif any(word in question_lower for word in ['education', 'school', 'university']):
            return "RIGHT TO EDUCATION", "Article 25A"
        elif any(word in question_lower for word in ['court', 'judge', 'judicial']):
            return "JUDICIARY AND JUDICIAL SYSTEM", "Article 175"
        elif any(word in question_lower for word in ['parliament', 'assembly', 'senate']):
            return "LEGISLATURE AND PARLIAMENT", "Articles 50-89"
        
        # Default generic heading
        words = question.split()[:5]
        if len(words) > 3:
            topic = ' '.join(words[:4]).upper()
        else:
            topic = ' '.join(words).upper()
        
        return topic, "Constitutional Provisions"
    
    def update_conversation_memory(self, question: str, response: str, question_info: Dict, 
                                  pdf_content: str = None, court_cases: List[Dict] = None):
        """Update conversation memory with current interaction"""
        memory_entry = {
            "timestamp": datetime.now().isoformat()[:19],
            "question": question,
            "response_preview": response[:150] + "..." if len(response) > 150 else response,
            "question_info": question_info,
            "classification": question_info.get("classification", {}),
            "articles_mentioned": question_info.get("articles", []),
            "scenario_type": question_info.get("scenario_type"),
            "detected_topics": question_info.get("detected_topics", [])
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
        
        if question_info.get("detected_topics"):
            self.user_context["last_topic"] = question_info["detected_topics"][0]
        elif question_info.get("scenario_type"):
            self.user_context["last_topic"] = question_info["scenario_type"]
        elif question_info.get("articles"):
            self.user_context["last_topic"] = f"Article {question_info['articles'][0]}"
        elif question_info.get("type"):
            self.user_context["last_topic"] = question_info["type"]
        
        # Store response content for follow-ups
        self.user_context["last_response_content"] = response
        
        # Store PDF content and court cases for follow-ups
        if pdf_content:
            self.user_context["last_pdf_content"] = pdf_content[:2000]  # Store first 2000 chars
        
        if court_cases:
            self.user_context["last_court_cases"] = court_cases
        
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
    
    def _get_starting_phrase(self, question_type: str = "constitutional", explanation_type: str = None) -> str:
        """Get a starting phrase based on question type"""
        if explanation_type == "simple":
            return random.choice(self.simplified_starting_phrases)
        elif explanation_type in ["detailed", "clarification", "general_explanation", "followup_explanation"]:
            return random.choice(self.followup_starting_phrases)
        elif question_type == "greeting" or question_type == "general":
            return random.choice(self.general_starting_phrases)
        else:
            return random.choice(self.starting_phrases)
    
    def _get_ending_phrase(self, question_type: str = "constitutional") -> str:
        """Get an ending phrase based on question type"""
        if question_type == "greeting" or question_type == "general":
            return random.choice(self.general_ending_phrases)
        else:
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
        
        return list(set(articles))[:10]  # Return up to 10 unique articles/sections
    
    def generate_response(self, question: str, question_info: Dict) -> str:
        """Generate response based on question classification"""
        
        classification = question_info.get("classification", {})
        category = classification.get("category", "constitutional")
        
        # Check if this is an explanation follow-up
        if question_info.get("is_explanation_request") and question_info.get("is_followup"):
            # Handle explanation follow-up with context
            response = self._handle_explanation_followup(question, question_info)
            self.update_conversation_memory(question, response, question_info)
            return response
        
        # Route to appropriate handler based on classification
        if category == "greeting":
            response = self._handle_greeting_with_model(question, question_info)
        elif category == "general":
            response = self._handle_general_with_model(question, question_info)
        elif category == "constitutional":
            response = self._handle_constitutional_with_model(question, question_info)
        else:
            # Fallback to old method
            response = self._handle_fallback(question, question_info)
        
        # Update memory
        self.update_conversation_memory(question, response, question_info)
        
        return response
    
    def _handle_explanation_followup(self, question: str, question_info: Dict) -> str:
        """Handle explanation follow-up questions with context from previous conversation"""
        try:
            # Get context from previous conversation
            followup_context = question_info.get("followup_context", {})
            last_response = followup_context.get("last_response", "")
            last_pdf_content = followup_context.get("last_pdf_content", "")
            last_topic = followup_context.get("last_topic", "the previous topic")
            last_articles = followup_context.get("last_articles", [])
            last_court_cases = followup_context.get("last_court_cases", [])
            
            explanation_type = question_info.get("explanation_type", "general_explanation")
            
            # Determine the type of explanation needed
            if explanation_type == "simple":
                return self._handle_simple_explanation(question, last_response, last_pdf_content, last_topic, last_articles, last_court_cases, question_info)
            elif explanation_type == "detailed":
                return self._handle_detailed_explanation(question, last_response, last_pdf_content, last_topic, last_articles, last_court_cases, question_info)
            elif explanation_type == "summary":
                return self._handle_summary_explanation(question, last_response, last_pdf_content, last_topic, last_articles, last_court_cases, question_info)
            elif explanation_type == "clarification":
                return self._handle_clarification_explanation(question, last_response, last_pdf_content, last_topic, last_articles, last_court_cases, question_info)
            else:
                return self._handle_general_explanation(question, last_response, last_pdf_content, last_topic, last_articles, last_court_cases, question_info)
                
        except Exception as e:
            logger.error(f"Explanation follow-up error: {e}")
            # Fallback to constitutional model
            return self._handle_constitutional_with_model(question, question_info)
    
    def _handle_simple_explanation(self, question: str, last_response: str, last_pdf_content: str, 
                                  last_topic: str, last_articles: List[str], last_court_cases: List[Dict], 
                                  question_info: Dict) -> str:
        """Provide a simple explanation of previous topic"""
        try:
            if self.constitutional_model:
                prompt = f"""Provide a SIMPLE, EASY-TO-UNDERSTAND explanation of the following constitutional topic:
                
                PREVIOUS TOPIC: {last_topic}
                PREVIOUS ARTICLES: {', '.join(last_articles) if last_articles else 'None'}
                USER REQUEST: "{question}"
                
                PREVIOUS EXPLANATION (for context):
                {last_response[:500]}
                
                PDF CONTENT (for reference):
                {last_pdf_content[:1000] if last_pdf_content else 'No PDF content available'}
                
                INSTRUCTIONS:
                1. Start with: "{random.choice(self.simplified_starting_phrases)}"
                2. Provide a VERY SIMPLE explanation in plain language
                3. Use everyday examples if helpful
                4. Avoid complex legal jargon
                5. Keep it concise (3-5 sentences)
                6. End with: "{random.choice(self.general_ending_phrases)}"
                
                Provide your simple explanation:"""
                
                response = self.constitutional_model.invoke(prompt)
                return response.strip()
            else:
                # Fallback simplified response
                starting_phrase = random.choice(self.simplified_starting_phrases)
                ending_phrase = random.choice(self.general_ending_phrases)
                
                if self.user_name:
                    return f"{self.user_name}, {starting_phrase}, {last_topic} is about fundamental constitutional principles that protect citizens' rights. {ending_phrase}"
                else:
                    return f"{starting_phrase}, {last_topic} is about fundamental constitutional principles that protect citizens' rights. {ending_phrase}"
                    
        except Exception as e:
            logger.error(f"Simple explanation error: {e}")
            starting_phrase = random.choice(self.simplified_starting_phrases)
            return f"{starting_phrase}, this constitutional provision establishes important legal principles."
    
    def _handle_detailed_explanation(self, question: str, last_response: str, last_pdf_content: str, 
                                   last_topic: str, last_articles: List[str], last_court_cases: List[Dict], 
                                   question_info: Dict) -> str:
        """Provide a detailed explanation of previous topic"""
        try:
            if self.constitutional_model:
                # Extract articles from last response if not already available
                articles_sections = self._extract_articles_sections(last_response + "\n" + last_pdf_content)
                if not articles_sections and last_articles:
                    articles_sections = [f"Article {art}" for art in last_articles]
                
                prompt = f"""Provide a DETAILED, COMPREHENSIVE explanation of the following constitutional topic:
                
                PREVIOUS TOPIC: {last_topic}
                RELEVANT ARTICLES/SECTIONS: {', '.join(articles_sections) if articles_sections else 'None'}
                USER REQUEST: "{question}"
                
                PREVIOUS EXPLANATION (for context):
                {last_response[:800]}
                
                PDF CONTENT (for reference):
                {last_pdf_content[:1500] if last_pdf_content else 'No PDF content available'}
                
                RELEVANT COURT CASES:
                {'; '.join([f"{case['case']}: {case['ruling'][:100]}" for case in last_court_cases[:2]]) if last_court_cases else 'None'}
                
                INSTRUCTIONS:
                1. Start with: "{random.choice(self.followup_starting_phrases)}"
                2. Provide a COMPREHENSIVE explanation (8-10 lines)
                3. Include: constitutional provisions, legal principles, practical applications
                4. Mention relevant articles and sections
                5. Include court precedents if applicable
                6. End with: "{random.choice(self.ending_phrases)}"
                
                Provide your detailed explanation:"""
                
                response = self.constitutional_model.invoke(prompt)
                
                # Format with sources if not included
                formatted_response = response.strip()
                if articles_sections and "SOURCES:" not in formatted_response.upper():
                    sources_text = "\n".join([f"- {art_sec}" for art_sec in articles_sections])
                    formatted_response = f"{formatted_response}\n\nSOURCES:\n{sources_text}"
                
                return formatted_response
            else:
                # Fallback detailed response
                starting_phrase = random.choice(self.followup_starting_phrases)
                ending_phrase = random.choice(self.ending_phrases)
                
                response_parts = []
                if self.user_name:
                    response_parts.append(f"{self.user_name}, {starting_phrase} on {last_topic}.")
                else:
                    response_parts.append(f"{starting_phrase} on {last_topic}.")
                
                response_parts.append(f"\nDetailed Analysis of {last_topic}")
                response_parts.append("")
                response_parts.append("This constitutional provision establishes comprehensive legal frameworks that define rights, duties, and state obligations. It incorporates fundamental principles that guide judicial interpretation and legislative implementation. The provision interacts with other constitutional articles to create a coherent legal system that balances individual liberties with state authority.")
                
                if last_articles:
                    response_parts.append("\nSOURCES:")
                    for article in last_articles[:3]:
                        response_parts.append(f"- Article {article}")
                
                response_parts.append(f"\n{ending_phrase}")
                
                return '\n'.join(response_parts)
                
        except Exception as e:
            logger.error(f"Detailed explanation error: {e}")
            return self._generate_exact_format_response(question, last_pdf_content, last_court_cases, question_info)
    
    def _handle_summary_explanation(self, question: str, last_response: str, last_pdf_content: str, 
                                  last_topic: str, last_articles: List[str], last_court_cases: List[Dict], 
                                  question_info: Dict) -> str:
        """Provide a summary of previous topic"""
        try:
            if self.constitutional_model:
                prompt = f"""Provide a CONCISE SUMMARY of the following constitutional topic:
                
                PREVIOUS TOPIC: {last_topic}
                USER REQUEST: "{question}"
                
                PREVIOUS EXPLANATION (to summarize):
                {last_response[:1000]}
                
                INSTRUCTIONS:
                1. Start with: "To summarize"
                2. Provide a BRIEF summary (2-3 sentences)
                3. Capture the essence of the topic
                4. Highlight key points
                5. End with: "In summary"
                
                Provide your concise summary:"""
                
                response = self.constitutional_model.invoke(prompt)
                return response.strip()
            else:
                # Fallback summary
                if self.user_name:
                    return f"{self.user_name}, in summary, {last_topic} establishes key constitutional principles that form the foundation of Pakistan's legal system."
                else:
                    return f"In summary, {last_topic} establishes key constitutional principles that form the foundation of Pakistan's legal system."
                    
        except Exception as e:
            logger.error(f"Summary explanation error: {e}")
            return f"To summarize, this constitutional provision is fundamental to Pakistan's legal framework."
    
    def _handle_clarification_explanation(self, question: str, last_response: str, last_pdf_content: str, 
                                        last_topic: str, last_articles: List[str], last_court_cases: List[Dict], 
                                        question_info: Dict) -> str:
        """Provide clarification of previous topic"""
        try:
            if self.constitutional_model:
                prompt = f"""Provide CLARIFICATION of the following constitutional topic:
                
                PREVIOUS TOPIC: {last_topic}
                USER REQUEST: "{question}"
                
                PREVIOUS EXPLANATION (to clarify):
                {last_response[:800]}
                
                PDF CONTENT (for reference):
                {last_pdf_content[:1000] if last_pdf_content else 'No PDF content available'}
                
                INSTRUCTIONS:
                1. Start with: "To clarify"
                2. Address any potential confusion
                3. Explain difficult concepts clearly
                4. Use simple language
                5. Provide examples if helpful
                6. End with: "I hope this clarifies the matter"
                
                Provide your clarification:"""
                
                response = self.constitutional_model.invoke(prompt)
                return response.strip()
            else:
                # Fallback clarification
                if self.user_name:
                    return f"{self.user_name}, to clarify, {last_topic} refers to constitutional provisions that define specific rights, duties, or legal procedures within Pakistan's constitutional framework."
                else:
                    return f"To clarify, {last_topic} refers to constitutional provisions that define specific rights, duties, or legal procedures within Pakistan's constitutional framework."
                    
        except Exception as e:
            logger.error(f"Clarification explanation error: {e}")
            return "To clarify, this constitutional provision addresses specific legal principles within Pakistan's constitutional framework."
    
    def _handle_general_explanation(self, question: str, last_response: str, last_pdf_content: str, 
                                  last_topic: str, last_articles: List[str], last_court_cases: List[Dict], 
                                  question_info: Dict) -> str:
        """Provide a general explanation of previous topic"""
        try:
            if self.constitutional_model:
                prompt = f"""Provide an EXPLANATION of the following constitutional topic:
                
                PREVIOUS TOPIC: {last_topic}
                USER REQUEST: "{question}"
                
                PREVIOUS EXPLANATION (for context):
                {last_response[:600]}
                
                INSTRUCTIONS:
                1. Start with: "{random.choice(self.followup_starting_phrases)}"
                2. Explain the topic clearly and comprehensively
                3. Cover key aspects mentioned previously
                4. Use accessible language
                5. Keep it informative but not too technical
                6. End with: "I hope this explanation helps"
                
                Provide your explanation:"""
                
                response = self.constitutional_model.invoke(prompt)
                return response.strip()
            else:
                # Fallback general explanation
                starting_phrase = random.choice(self.followup_starting_phrases)
                if self.user_name:
                    return f"{self.user_name}, {starting_phrase}, {last_topic} encompasses constitutional principles that govern specific aspects of Pakistan's legal system, ensuring rights protection and legal certainty."
                else:
                    return f"{starting_phrase}, {last_topic} encompasses constitutional principles that govern specific aspects of Pakistan's legal system, ensuring rights protection and legal certainty."
                    
        except Exception as e:
            logger.error(f"General explanation error: {e}")
            return self._handle_constitutional_with_model(question, question_info)
    
    def _handle_greeting_with_model(self, question: str, question_info: Dict) -> str:
        """Handle greetings using greeting model"""
        try:
            if self.greeting_model:
                # Simple prompt for greeting model
                prompt = f"""Respond to this greeting naturally and politely: "{question}"
                
                User's name: {self.user_name if self.user_name else 'Not provided'}
                
                Respond in a warm, friendly manner. Keep it brief (1-2 sentences)."""
                
                response = self.greeting_model.invoke(prompt)
                return response.strip()
            else:
                # Fallback to hardcoded responses
                return self._handle_greeting(question_info)
        except Exception as e:
            logger.error(f"Greeting model error: {e}")
            return self._handle_greeting(question_info)
    
    def _handle_general_with_model(self, question: str, question_info: Dict) -> str:
        """Handle general questions using general model"""
        try:
            if self.general_model:
                # Enhanced prompt for general questions
                prompt = f"""You are a helpful assistant. Answer this general question: "{question}"
                
                Context:
                - User's name: {self.user_name if self.user_name else 'Not provided'}
                - Previous topic: {self.user_context.get('last_topic', 'None')}
                - This is a general conversation question
                
                Instructions:
                1. Be friendly and helpful
                2. Keep response concise but informative
                3. If it's about your capabilities, list key features
                4. If asking about yourself, describe your purpose briefly
                5. Add a polite closing
                
                Provide your response:"""
                
                response = self.general_model.invoke(prompt)
                
                # Format with starting and ending phrases
                starting_phrase = self._get_starting_phrase("general")
                ending_phrase = self._get_ending_phrase("general")
                
                if self.user_name:
                    formatted_response = f"{self.user_name}, {starting_phrase}. {response} {ending_phrase}"
                else:
                    formatted_response = f"{starting_phrase}. {response} {ending_phrase}"
                
                return formatted_response.strip()
            else:
                # Fallback to hardcoded responses
                return self._handle_general_conversation(question, question_info)
        except Exception as e:
            logger.error(f"General model error: {e}")
            return self._handle_general_conversation(question, question_info)
    
    def _handle_constitutional_with_model(self, question: str, question_info: Dict) -> str:
        """Handle constitutional questions with PDF search and structured response"""
        # Extract PDF content
        pdf_content, pdf_chunks = self.extract_pdf_content(question)
        
        # Get court cases for scenarios
        court_cases = []
        if question_info["requires_court_cases"]:
            scenario_type = question_info.get("scenario_type")
            if scenario_type:
                court_cases = self.get_relevant_court_cases(scenario_type)
            else:
                # Check detected topics
                detected_topics = question_info.get("detected_topics", [])
                if detected_topics:
                    for topic in detected_topics:
                        if topic in self.court_cases_db:
                            court_cases.extend(self.court_cases_db[topic])
        
        try:
            if self.constitutional_model:
                # Get conversation context
                conversation_context = self.get_conversation_context()
                user_name_context = f"User's name: {self.user_name}\n" if self.user_name else ""
                
                # Check if this is an explanation request (but not follow-up)
                explanation_type = question_info.get("explanation_type")
                if explanation_type and not question_info.get("is_followup"):
                    # Build explanation-specific prompt
                    prompt = self._build_explanation_prompt(
                        question, pdf_content, court_cases, question_info, 
                        conversation_context, user_name_context, explanation_type
                    )
                else:
                    # Build EXACT format prompt
                    prompt = self._build_exact_format_prompt(
                        question, pdf_content, court_cases, question_info, 
                        conversation_context, user_name_context
                    )
                
                # Get response from constitutional model
                response = self.constitutional_model.invoke(prompt)
                
                # Format the response EXACTLY as requested
                formatted_response = self._format_exact_response(
                    response, question_info, pdf_content, court_cases
                )
                
                # Update memory with PDF content and court cases
                self.update_conversation_memory(question, formatted_response, question_info, pdf_content, court_cases)
                
                return formatted_response
            else:
                # Fallback to exact format response
                response = self._generate_exact_format_response(question, pdf_content, court_cases, question_info)
                self.update_conversation_memory(question, response, question_info, pdf_content, court_cases)
                return response
        except Exception as e:
            logger.error(f"Constitutional model error: {e}")
            response = self._generate_exact_format_response(question, pdf_content, court_cases, question_info)
            self.update_conversation_memory(question, response, question_info, pdf_content, court_cases)
            return response
    
    def _build_exact_format_prompt(self, question: str, pdf_content: str, 
                                 court_cases: List[Dict], question_info: Dict,
                                 conversation_context: str, user_name_context: str) -> str:
        """Build prompt for EXACT format responses"""
        
        # Generate topic heading
        topic, article_line = self._generate_topic_heading(question, question_info)
        
        # Extract additional context
        complexity = question_info.get("complexity_level", "simple")
        is_scenario = question_info.get("is_scenario", False)
        detected_topics = question_info.get("detected_topics", [])
        
        # Context sections
        context_parts = []
        
        if pdf_content:
            context_parts.append(f"CONSTITUTIONAL DOCUMENT EXTRACTS:\n{pdf_content[:2500]}")
        
        if court_cases:
            cases_text = "\n".join([f"- {case['case']}: {case['ruling'][:200]}..." for case in court_cases[:3]])
            context_parts.append(f"RELEVANT COURT CASES:\n{cases_text}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific context available from documents."
        
        # Build EXACT format prompt
        prompt = f"""You are a Constitutional Law Expert for Pakistan. Provide a response in the EXACT format specified below.

QUESTION: "{question}"

TOPIC: {topic}
ARTICLE/SECTION: {article_line}

QUESTION DETAILS:
- Complexity: {complexity.upper()}
- Is Scenario Question: {'Yes' if is_scenario else 'No'}
- Detected Topics: {', '.join(detected_topics) if detected_topics else 'None'}
- Articles Mentioned: {', '.join(question_info['articles']) if question_info['articles'] else 'None'}

RELEVANT CONTEXT:
{context}

EXACT RESPONSE FORMAT - DO NOT DEVIATE FROM THIS:

1. STARTING PHRASE: Begin with exactly: "Considering the constitutional framework and legal provisions."

2. CONTEXT LINE: On the next line: "In the context of the Constitution of the Islamic Republic of Pakistan,"

3. TOPIC HEADING: On a new line, write: "{topic.upper()} {article_line}"

4. DETAILED ANALYSIS: A comprehensive paragraph (10-15 lines) that includes:
   - What the constitutional provision establishes
   - Historical evolution and constitutional amendments if relevant
   - Legal principles and judicial interpretations
   - Practical applications and implications
   - State obligations and responsibilities
   - Rights protection mechanisms
   - Social and legal importance
   - Any limitations, exceptions, or restrictions
   - Current challenges and implementation issues
   - Comparative analysis if relevant

5. SOURCES SECTION: Start with "SOURCES:" on a new line
   - List all relevant Articles (e.g., Article 1, Article 142, Article 153, Article 154, Article 246)
   - List relevant Sections if applicable (e.g., Section 302 PPC, Section 337 PPC)
   - Include constitutional amendments if relevant (e.g., 18th Amendment, 25th Amendment)
   - One article/section/amendment per line
   - Include at least 5-8 sources

6. COURT PRECEDENTS: Start with "COURT PRECEDENTS:" on a new line
   - List 2-3 relevant court cases
   - Format: "The Supreme Court of Pakistan has [ruling/held] in the case of [Case Name] that [ruling summary]."
   - Include key rulings and their implications

7. CONCLUDING PARAGRAPH: Start with "In the light of the Constitution of the Islamic Republic of Pakistan, it is clear that"
   - Summarize the main constitutional principle
   - State why it's important or what it means for governance
   - Mention ongoing challenges or areas for improvement
   - Keep it concise (3-4 sentences)

8. ENDING PHRASE: End with exactly: "This reflects established constitutional interpretation."

IMPORTANT RULES:
- DO NOT use markdown (no *, #, -, =, etc.)
- DO NOT use brackets or special formatting
- DO NOT add extra blank lines between sections
- Keep paragraph breaks clean (one blank line between major sections only)
- Use professional, precise language
- Make the analysis comprehensive but accessible
- For federalism questions, mention Articles 1, 142, 153, 154, 246
- For freedom of speech, mention Article 19 and reasonable restrictions
- For minorities, mention Article 35 and protection mechanisms
- Reference specific constitutional amendments when relevant

Now provide the response in the EXACT format specified above:"""
        
        return prompt
    
    def _format_exact_response(self, llm_response: str, question_info: Dict, 
                             pdf_content: str, court_cases: List[Dict]) -> str:
        """Format response EXACTLY as requested"""
        
        # Clean the response
        cleaned = llm_response.strip()
        
        # Remove markdown symbols
        symbols = ['*', '=', '-', '#', '`', '~', '_', '[', ']', '**', '__', '##', '###', '####']
        for symbol in symbols:
            cleaned = cleaned.replace(symbol, '')
        
        # Ensure proper starting phrase
        if not cleaned.startswith("Considering the constitutional framework and legal provisions."):
            cleaned = f"Considering the constitutional framework and legal provisions.\n\n{cleaned}"
        
        # Ensure proper context line
        if "In the context of the Constitution of the Islamic Republic of Pakistan," not in cleaned:
            lines = cleaned.split('\n')
            if len(lines) > 1:
                lines.insert(1, "In the context of the Constitution of the Islamic Republic of Pakistan,")
                cleaned = '\n'.join(lines)
        
        # Ensure ending phrase
        if "This reflects established constitutional interpretation." not in cleaned[-50:]:
            cleaned = f"{cleaned}\n\nThis reflects established constitutional interpretation."
        
        # Clean extra whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
        
        return cleaned.strip()
    
    def _generate_exact_format_response(self, question: str, pdf_content: str,
                                      court_cases: List[Dict], question_info: Dict) -> str:
        """Generate fallback response in exact format"""
        
        # Generate topic heading
        topic, article_line = self._generate_topic_heading(question, question_info)
        
        # Extract articles from PDF content
        extracted_articles = self._extract_articles_sections(pdf_content)
        
        # Default sources based on topic
        default_sources = {
            "FEDERALISM": ["Article 1", "Article 142", "Article 153", "Article 154", "Article 246", "18th Amendment", "25th Amendment"],
            "FREEDOM OF SPEECH": ["Article 19", "Article 14", "Article 15", "Article 16", "Article 17", "Article 19A"],
            "PROTECTION OF MINORITIES": ["Article 20", "Article 25", "Article 26", "Article 27", "Article 28", "Article 35"],
            "RIGHT TO EDUCATION": ["Article 25A", "Article 37", "Article 38"],
            "JUDICIARY": ["Article 175", "Article 184", "Article 185", "Article 189", "Article 190"],
            "LEGISLATURE": ["Article 50", "Article 51", "Article 59", "Article 60", "Article 89"]
        }
        
        # Determine sources
        sources = []
        for key, value in default_sources.items():
            if key in topic.upper():
                sources = value
                break
        
        if not sources and extracted_articles:
            sources = extracted_articles[:6]
        elif not sources:
            sources = ["Article 4", "Article 8", "Article 9", "Article 14", "Article 25"]
        
        # Determine court cases
        if court_cases:
            court_precedents = []
            for case in court_cases[:2]:
                court_precedents.append(f"The Supreme Court of Pakistan has held in the case of {case['case']} that {case['ruling']}")
            court_text = "\n".join(court_precedents)
        else:
            court_text = "The superior courts of Pakistan have consistently interpreted constitutional provisions to protect fundamental rights and ensure constitutional governance."
        
        # Build the response in exact format
        response_parts = []
        response_parts.append("Considering the constitutional framework and legal provisions.")
        response_parts.append("In the context of the Constitution of the Islamic Republic of Pakistan,")
        response_parts.append(f"{topic.upper()} {article_line}")
        response_parts.append("")
        
        # Add detailed analysis
        analysis = f"""The constitutional framework of Pakistan establishes {topic.lower()} as a fundamental principle that governs the relationship between the state and citizens, as well as among state institutions. This provision incorporates core values that guide legislative action, executive implementation, and judicial interpretation. The historical evolution of this constitutional principle reflects Pakistan's commitment to democratic governance, rule of law, and protection of fundamental rights. Judicial precedents have consistently reinforced the importance of {topic.lower()} in maintaining constitutional balance and ensuring effective governance. The practical implementation of this provision involves complex interactions between federal and provincial authorities, requiring continuous dialogue and cooperative federalism. Current challenges include resource allocation, institutional capacity, and ensuring uniform implementation across different regions of Pakistan."""
        
        response_parts.append(analysis)
        response_parts.append("")
        response_parts.append("SOURCES:")
        
        # Add sources
        for source in sources:
            response_parts.append(f"{source}")
        
        response_parts.append("")
        response_parts.append("COURT PRECEDENTS:")
        response_parts.append(f"{court_text}")
        
        response_parts.append("")
        response_parts.append("In the light of the Constitution of the Islamic Republic of Pakistan, it is clear that this constitutional provision establishes fundamental principles that must be respected, protected, and fulfilled by all state institutions and citizens. The implementation requires continuous effort, institutional cooperation, and judicial oversight to ensure effective governance and protection of rights.")
        response_parts.append("")
        response_parts.append("This reflects established constitutional interpretation.")
        
        return '\n'.join(response_parts)
    
    def _build_explanation_prompt(self, question: str, pdf_content: str, 
                                court_cases: List[Dict], question_info: Dict,
                                conversation_context: str, user_name_context: str,
                                explanation_type: str) -> str:
        """Build prompt for explanation requests"""
        
        # Context sections
        context_parts = []
        
        if pdf_content:
            context_parts.append(f"CONSTITUTIONAL DOCUMENT EXTRACTS:\n{pdf_content[:2000]}")
        
        if court_cases:
            cases_text = "\n".join([f"- {case['case']}: {case['ruling'][:150]}..." for case in court_cases[:2]])
            context_parts.append(f"RELEVANT COURT CASES:\n{cases_text}")
        
        context = "\n\n".join(context_parts) if context_parts else "No specific context available from documents."
        
        # Determine explanation style
        if explanation_type == "simple":
            instruction = "Provide a SIMPLE, EASY-TO-UNDERSTAND explanation in plain language. Avoid legal jargon. Use everyday examples if helpful."
            paragraph_length = "3-5 sentences"
        elif explanation_type == "detailed":
            instruction = "Provide a DETAILED, COMPREHENSIVE explanation covering all aspects thoroughly."
            paragraph_length = "8-10 lines"
        elif explanation_type == "summary":
            instruction = "Provide a CONCISE SUMMARY capturing the essence and key points."
            paragraph_length = "2-3 sentences"
        else:  # general_explanation or clarification
            instruction = "Provide a CLEAR, INFORMATIVE explanation that addresses the question directly."
            paragraph_length = "5-7 lines"
        
        prompt = f"""You are a Constitutional Law Expert for Pakistan. Provide an EXPLANATION as requested.

USER INFORMATION:
{user_name_context}{conversation_context}

CURRENT QUESTION: "{question}"
EXPLANATION TYPE: {explanation_type.upper()}

CONTEXT INFORMATION:
{context}

RESPONSE REQUIREMENTS:

1. STARTING PHRASE: Begin with an appropriate starting phrase for an explanation

2. TITLE: Create a clear and bold title indicating this is an explanation

3. EXPLANATION PARAGRAPH: {paragraph_length} {instruction}

4. KEY POINTS: Highlight 2-3 key points if helpful

5. ENDING: Conclude appropriately

SPECIAL INSTRUCTIONS:
- Tailor the explanation to the requested type ({explanation_type})
- Use language appropriate for the explanation type
- Ensure clarity and accuracy
- Reference constitutional provisions if relevant

Provide your explanation now:"""
        
        return prompt
    
    def _handle_fallback(self, question: str, question_info: Dict) -> str:
        """Fallback handler for unclassified questions"""
        # Try to use general model first
        if self.general_model:
            try:
                prompt = f"""Answer this question helpfully: "{question}"
                
                Provide a clear, informative response. If it's about Pakistan's constitution,
                mention that you specialize in constitutional matters."""
                
                response = self.general_model.invoke(prompt)
                return response.strip()
            except:
                pass
        
        # Ultimate fallback
        return "I'm here to assist with constitutional and legal matters. Could you please rephrase or provide more details about your question?"
    
    def _handle_greeting(self, question_info: Dict) -> str:
        """Handle greetings (fallback)"""
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
    
    def _handle_general_conversation(self, question: str, question_info: Dict) -> str:
        """Handle general conversation (fallback)"""
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
        
        elif any(p in question_lower for p in self.general_conversation['date']):
            from datetime import datetime
            return f"Today's date is {datetime.now().strftime('%B %d, %Y')}."
        
        return "I'm here to assist with constitutional and legal matters. What would you like to know?"
    
    def answer_question(self, question: str) -> str:
        """Main function to answer questions with classification system"""
        try:
            # Detect question type with classification
            question_info = self.detect_question_type(question)
            question_info["original_question"] = question
            
            # Log classification
            classification = question_info.get("classification", {})
            logger.info(f"🔍 Question classified as: {classification.get('category', 'unknown')} "
                       f"(confidence: {classification.get('confidence', 0)})")
            
            if question_info.get("is_explanation_request"):
                logger.info(f"📝 Explanation type: {question_info.get('explanation_type')}")
            
            # Generate response using appropriate model
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
    print("🇵🇰 ENHANCED CONSTITUTIONAL ASSISTANT WITH EXACT FORMATTING")
    print("="*70)
    
    success = assistant.initialize()
    
    if success:
        print("\n✅ System Ready with Advanced Topic Detection and Exact Formatting!")
        print("\n💡 Test These Questions:")
        print("1. 'What is federalism in Pakistan?'")
        print("2. 'Explain Article 19'")
        print("3. 'Tell me about minority rights'")
        print("4. 'What are the powers of provincial governments?'")
        print("5. 'Explain the 18th Amendment'")
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