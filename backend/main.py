"""
MAIN FASTAPI APPLICATION FOR PAKISTAN CONSTITUTION ASSISTANT
Railway Production Version
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import time

# Import RAG system
from rag_chain import answer_question, assistant

# --------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    user_name: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    success: bool = True
    error: Optional[str] = None
    processing_time: float | None = None


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

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="Pakistan Constitution AI Assistant",
    description="Ask questions about the Constitution of Pakistan using AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --------------------------------------------------
# CORS (Allow Frontend Access)
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------
system_initialized = False

# --------------------------------------------------
# STARTUP EVENT
# --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global system_initialized
    try:
        logger.info("🚀 Starting Pakistan Constitution AI Assistant...")

        logger.info("📚 Initializing RAG system...")
        assistant.initialize()

        logger.info("🧪 Running test query...")
        test = answer_question("Hello")

        if test:
            system_initialized = True
            logger.info("✅ System initialized successfully")
        else:
            system_initialized = False
            logger.error("❌ Initialization test failed")

    except FileNotFoundError as e:
        system_initialized = False
        logger.error(f"❌ Database not found: {e}")
        logger.info("💡 Run ingest.py before deployment")

    except Exception as e:
        system_initialized = False
        logger.error(f"❌ Startup failed: {str(e)}")

# --------------------------------------------------
# ROOT
# --------------------------------------------------
@app.get("/", response_model=SystemInfo)
async def root():
    return SystemInfo(
        service="Pakistan Constitution AI Assistant",
        status="online" if system_initialized else "initializing",
        endpoints=[
            {"method": "GET", "path": "/", "description": "System info"},
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "POST", "path": "/ask", "description": "Ask questions"},
            {"method": "GET", "path": "/examples", "description": "Example queries"}
        ],
        features=[
            "Pakistan Constitution Q&A",
            "PDF-based RAG system",
            "Greeting detection",
            "Name memory",
            "Conversation memory",
            "Clean AI responses"
        ]
    )

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    if not system_initialized:
        return HealthResponse(
            status="initializing",
            ready=False,
            error="System is still starting"
        )

    document_count = 0
    database_loaded = False

    try:
        if assistant.vector_store:
            try:
                document_count = assistant.vector_store._collection.count()
            except Exception:
                docs = assistant.vector_store.get()
                document_count = len(docs.get("ids", []))
            database_loaded = document_count > 0
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        ready=True,
        database_loaded=database_loaded,
        document_count=document_count,
        llm_available=assistant.llm is not None
    )

# --------------------------------------------------
# EXAMPLES
# --------------------------------------------------
@app.get("/examples")
async def examples():
    return {
        "constitutional": [
            "What is Article 25A?",
            "Explain Article 19",
            "Right to equality in Pakistan",
            "Freedom of speech"
        ],
        "general": [
            "Assalamualaikum",
            "Hello",
            "What can you do?",
            "Who created you?"
        ],
        "tips": [
            "Mention article numbers",
            "Ask follow-up questions",
            "Simple English works best"
        ]
    }

# --------------------------------------------------
# ASK QUESTION (MAIN ENDPOINT)
# --------------------------------------------------
@app.post("/ask", response_model=ChatResponse)
async def ask(request: ChatRequest):
    start_time = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not system_initialized:
        raise HTTPException(status_code=503, detail="System initializing")

    try:
        logger.info(f"📥 Question: {request.question[:100]}")

        answer = answer_question(request.question)

        return ChatResponse(
            question=request.question,
            answer=answer,
            processing_time=round(time.time() - start_time, 2)
        )

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return ChatResponse(
            question=request.question,
            answer="An error occurred while processing your question.",
            success=False,
            error=str(e),
            processing_time=round(time.time() - start_time, 2)
        )

# --------------------------------------------------
# LEGACY CHAT
# --------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_legacy(request: ChatRequest):
    return await ask(request)

# --------------------------------------------------
# MEMORY STATUS
# --------------------------------------------------
@app.get("/memory_status")
async def memory_status():
    return {
        "user_name": getattr(assistant, "user_name", None),
        "conversation_count": len(getattr(assistant, "conversation_history", [])),
        "system_initialized": system_initialized
    }

# --------------------------------------------------
# CLEAR MEMORY
# --------------------------------------------------
@app.get("/clear_memory")
async def clear_memory():
    try:
        assistant.user_name = None
        assistant.conversation_history = []
        return {"success": True, "message": "Memory cleared"}
    except Exception as e:
        return {"success": False, "error": str(e)}
