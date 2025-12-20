"""
MAIN FASTAPI APPLICATION FOR PAKISTAN CONSTITUTION ASSISTANT
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import logging
import os
import time

# Import our RAG system
from rag_chain import answer_question, assistant

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    user_name: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    success: bool = True
    error: Optional[str] = None
    processing_time: float = None

class HealthResponse(BaseModel):
    status: str
    ready: bool
    database_loaded: bool = False
    document_count: int = 0
    llm_available: bool = False
    error: Optional[str] = None

class SystemInfo(BaseModel):
    service: str
    version: str = "2.0.0"
    status: str
    endpoints: List[dict]
    features: List[str]

# Initialize FastAPI
app = FastAPI(
    title="Pakistan Constitution AI Assistant",
    description="API for asking questions about Pakistan's Constitution with structured responses",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization flag
system_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    global system_initialized
    try:
        logger.info("🚀 Starting Pakistan Constitution Assistant...")
        
        # Initialize the assistant
        logger.info("📚 Initializing RAG system...")
        assistant.initialize()
        
        # Test with a simple question
        logger.info("🧪 Testing system with sample question...")
        test_response = answer_question("hello")
        
        if test_response:
            system_initialized = True
            logger.info("✅ System initialized successfully!")
            logger.info(f"📝 Test response: {test_response[:100]}...")
        else:
            logger.error("❌ System test failed - no response received")
            system_initialized = False
            
    except FileNotFoundError as e:
        logger.error(f"❌ Database not found: {e}")
        logger.info("💡 Please run: python backend/ingest.py")
        system_initialized = False
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {str(e)}")
        system_initialized = False

@app.get("/", response_model=SystemInfo)
async def root():
    """Root endpoint with system information"""
    return SystemInfo(
        service="Pakistan Constitution AI Assistant",
        status="online" if system_initialized else "initializing",
        endpoints=[
            {"method": "GET", "path": "/", "description": "System information"},
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "POST", "path": "/ask", "description": "Ask constitutional questions"},
            {"method": "GET", "path": "/examples", "description": "Example questions"}
        ],
        features=[
            "Structured constitutional answers",
            "Greeting detection (Assalamualaikum, Hello, etc.)",
            "Name recognition and personalization",
            "General conversation handling",
            "Clean formatting without symbols",
            "PDF-based constitutional content",
            "Conversation memory"
        ]
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        if system_initialized:
            # Get database stats if possible
            document_count = 0
            database_loaded = False
            
            try:
                if assistant.vector_store:
                    # Try different methods to get count
                    try:
                        document_count = assistant.vector_store._collection.count()
                    except:
                        all_docs = assistant.vector_store.get()
                        document_count = len(all_docs['ids']) if 'ids' in all_docs else 0
                    database_loaded = document_count > 0
            except:
                pass
            
            llm_available = assistant.llm is not None
            
            return HealthResponse(
                status="healthy",
                ready=True,
                database_loaded=database_loaded,
                document_count=document_count,
                llm_available=llm_available
            )
        else:
            return HealthResponse(
                status="initializing",
                ready=False,
                database_loaded=False,
                llm_available=False,
                error="System not fully initialized. Check logs for details."
            )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="error",
            ready=False,
            error=str(e)
        )

@app.get("/examples")
async def get_examples():
    """Get example questions"""
    examples = {
        "greetings": [
            "Assalamualaikum",
            "Hello",
            "Good morning",
            "How are you?",
            "Salam"
        ],
        "name_handling": [
            "My name is Ahmed",
            "Call me sir",
            "Whats my name?",
            "You can call me Ali"
        ],
        "general_conversation": [
            "How's it going?",
            "Tell me about yourself",
            "What can you do?",
            "Who created you?"
        ],
        "constitutional_questions": [
            "What is Article 25A?",
            "Explain Article 19",
            "Tell me about freedom of speech",
            "What does the Constitution say about education?",
            "Explain right to equality",
            "Article 14 explanation"
        ],
        "follow_up_questions": [
            "Can you explain further?",
            "What else about this?",
            "Tell me more details",
            "How is this implemented?"
        ]
    }
    return {
        "message": "Try these example questions:",
        "examples": examples,
        "tips": [
            "Use clear, simple English",
            "Include article numbers for specific queries",
            "You can ask follow-up questions",
            "System remembers your name if you provide it"
        ]
    }

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Main endpoint for asking constitutional questions"""
    start_time = time.time()
    
    try:
        # Validate input
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
        
        # Check if system is ready
        if not system_initialized:
            raise HTTPException(
                status_code=503,
                detail="System is initializing. Please try again in a moment."
            )
        
        logger.info(f"📥 Question: {request.question[:100]}...")
        
        # Get answer from RAG system
        answer = answer_question(request.question)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Answered in {processing_time:.2f} seconds")
        
        return ChatResponse(
            question=request.question,
            answer=answer,
            success=True,
            processing_time=processing_time
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Error processing question: {str(e)}")
        
        # Provide user-friendly error response
        error_message = f"I encountered an error while processing your question: '{request.question}'. "
        error_message += "Please try rephrasing your question or ask about a different constitutional topic."
        
        return ChatResponse(
            question=request.question,
            answer=error_message,
            success=False,
            error=str(e),
            processing_time=processing_time
        )

@app.post("/chat", response_model=ChatResponse)
async def chat_legacy(request: ChatRequest):
    """Legacy endpoint for backward compatibility"""
    return await ask_question(request)

@app.get("/clear_memory")
async def clear_memory():
    """Clear conversation memory"""
    try:
        if hasattr(assistant, 'user_name'):
            old_name = assistant.user_name
            assistant.user_name = None
            assistant.conversation_history = []
            
            logger.info(f"🧹 Memory cleared for user: {old_name}")
            
            return {
                "success": True,
                "message": f"Conversation memory cleared. Goodbye {old_name}!" if old_name else "Conversation memory cleared."
            }
        else:
            return {
                "success": True,
                "message": "Memory system not initialized yet."
            }
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/memory_status")
async def memory_status():
    """Get current memory status"""
    try:
        status = {
            "user_name": assistant.user_name,
            "conversation_count": len(assistant.conversation_history),
            "system_initialized": system_initialized
        }
        
        if assistant.conversation_history:
            recent_questions = [
                {
                    "question": item.get("question", "")[:50],
                    "time": item.get("timestamp", "")
                }
                for item in assistant.conversation_history[-5:]
            ]
            status["recent_conversations"] = recent_questions
        
        return status
    except Exception as e:
        return {
            "error": str(e),
            "user_name": None,
            "conversation_count": 0
        }

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify system is working"""
    test_questions = [
        "Assalamualaikum",
        "My name is TestUser",
        "What is Article 25A?",
        "Hello"
    ]
    
    results = []
    
    for question in test_questions:
        try:
            start_time = time.time()
            answer = answer_question(question)
            processing_time = time.time() - start_time
            
            results.append({
                "question": question,
                "answer_preview": answer[:100] + ("..." if len(answer) > 100 else ""),
                "length": len(answer),
                "processing_time": round(processing_time, 2),
                "success": True
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e),
                "success": False
            })
    
    return {
        "system_status": "online" if system_initialized else "offline",
        "test_results": results,
        "total_tests": len(test_questions),
        "successful_tests": sum(1 for r in results if r.get("success", False))
    }

if __name__ == "__main__":
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print("\n" + "="*70)
    print("🇵🇰 PAKISTAN CONSTITUTION AI ASSISTANT API")
    print("="*70)
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Health Check: http://{host}:{port}/health")
    print(f"💬 Main Endpoint: POST http://{host}:{port}/ask")
    print("\n📋 Example curl command:")
    print(f'curl -X POST http://{host}:{port}/ask \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"question": "What is Article 25A?"}\'')
    print("\n" + "="*70)
    
    # Start the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,  # Auto-reload during development
        log_level="info"
    )