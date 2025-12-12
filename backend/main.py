# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import uvicorn
import os
import re

# Import our RAG system
from rag_chain import answer_question, assistant

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    success: bool = True
    error: str = None
    processing_time: float = None

class HealthResponse(BaseModel):
    status: str
    ready: bool
    pdf_count: int = 0
    error: str = None

class SystemInfo(BaseModel):
    status: str
    version: str = "1.0.0"
    service: str = "Pakistan Constitution Assistant"
    endpoints: list

# Initialize FastAPI
app = FastAPI(
    title="Pakistan Constitution Assistant API",
    description="API for asking questions about the Constitution of Pakistan",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json"  # ✅ Added for API prefix compatibility
)

# CORS middleware - update with your frontend URLs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # React default
        "http://127.0.0.1:3000",
        "http://localhost:5500",    # Python HTTP server
        "http://127.0.0.1:5500",
        "http://localhost:8080",    # Your backend port
        "http://127.0.0.1:8080",
        "http://localhost:8000",    # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global initialization flag
rag_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_initialized
    try:
        logger.info("🚀 Starting Pakistan Constitution Assistant...")
        
        # Check if ChromaDB exists
        if not os.path.exists("./chroma_db"):
            logger.error("❌ ChromaDB not found! Please run ingest.py first.")
            rag_initialized = False
            return
        
        # Initialize the assistant
        assistant.initialize()
        
        # Test the system
        test_response = answer_question("hello")
        logger.info(f"✅ System test: {test_response[:50]}...")
        
        rag_initialized = True
        logger.info("✅ RAG system initialized and ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        rag_initialized = False
        raise

# ✅ API PREFIX ROUTES
@app.get("/api/", response_model=SystemInfo)
async def api_root():
    """Root endpoint with system information (with /api/ prefix)"""
    return SystemInfo(
        status="online" if rag_initialized else "initializing",
        endpoints=[
            {"method": "GET", "path": "/api/", "description": "System information"},
            {"method": "GET", "path": "/api/health", "description": "Health check"},
            {"method": "POST", "path": "/api/chat", "description": "Ask questions about Constitution"},
            {"method": "GET", "path": "/api/stats", "description": "Get PDF statistics"}
        ]
    )

@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    """Health check endpoint (with /api/ prefix)"""
    try:
        if rag_initialized:
            # Try to get some stats
            import chromadb
            try:
                client = chromadb.PersistentClient(path="./chroma_db")
                collection = client.get_collection("pakistan_constitution")
                pdf_count = collection.count()
            except:
                pdf_count = 0
            
            return HealthResponse(
                status="healthy",
                ready=True,
                pdf_count=pdf_count
            )
        else:
            return HealthResponse(
                status="initializing",
                ready=False,
                error="RAG system not initialized. Run ingest.py first."
            )
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return HealthResponse(
            status="error",
            ready=False,
            error=str(e)
        )

@app.get("/api/stats")
async def api_get_stats():
    """Get PDF statistics (with /api/ prefix)"""
    try:
        if not rag_initialized:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("constitutional_docs")
        
        # Get all documents
        all_docs = collection.get()
        
        # Count sources
        sources = {}
        if all_docs['metadatas']:
            for metadata in all_docs['metadatas']:
                if metadata and 'source' in metadata:
                    source = metadata['source']
                    sources[source] = sources.get(source, 0) + 1
        
        # Find unique articles
        articles_found = set()
        for doc in all_docs['documents'][:1000]:  # Check first 1000 docs
            article_matches = re.findall(r'Article\s+(\d+[A-Z\-]*)', doc, re.IGNORECASE)
            for match in article_matches:
                articles_found.add(match.upper())
        
        return {
            "status": "success",
            "total_chunks": collection.count(),
            "pdf_files": len(sources),
            "sources": [{"name": name, "chunks": count} for name, count in sources.items()][:10],
            "articles_found": sorted(list(articles_found))[:20],
            "system_ready": rag_initialized
        }
        
    except Exception as e:
        logger.error(f"❌ Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """Main chat endpoint - ask questions about Constitution (with /api/ prefix)"""
    import time
    
    start_time = time.time()
    
    try:
        # Check if system is ready
        if not rag_initialized:
            raise HTTPException(
                status_code=503,
                detail="System not initialized. Please check if ingest.py was run."
            )
        
        logger.info(f"📨 Question from {request.session_id}: {request.question}")
        
        # Get answer from our RAG system
        answer = answer_question(request.question)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Answered in {processing_time:.2f}s: {answer[:100]}...")
        
        return ChatResponse(
            answer=answer,
            success=True,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Chat error: {str(e)}")
        
        return ChatResponse(
            answer=f"Sorry, I encountered an error: {str(e)[:200]}",
            success=False,
            error=str(e),
            processing_time=processing_time
        )

@app.post("/api/batch")
async def api_batch_chat(requests: list[ChatRequest]):
    """Batch processing endpoint for multiple questions (with /api/ prefix)"""
    responses = []
    
    for request in requests:
        try:
            response = await api_chat(request)
            responses.append({
                "question": request.question,
                "answer": response.answer,
                "success": response.success,
                "processing_time": response.processing_time
            })
        except Exception as e:
            responses.append({
                "question": request.question,
                "answer": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "total_questions": len(requests),
        "successful": sum(1 for r in responses if r["success"]),
        "responses": responses
    }

# ✅ KEEP OLD ENDPOINTS FOR BACKWARD COMPATIBILITY
@app.get("/", response_model=SystemInfo)
async def root():
    """Legacy root endpoint"""
    return await api_root()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Legacy health endpoint"""
    return await api_health_check()

@app.get("/stats")
async def get_stats():
    """Legacy stats endpoint"""
    return await api_get_stats()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Legacy chat endpoint"""
    return await api_chat(request)

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8080))  # ✅ Changed to 8080 to match your frontend
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    logger.info(f"📚 Pakistan Constitution Assistant API")
    logger.info(f"📄 Docs available at: http://{host}:{port}/docs")
    logger.info(f"📡 API endpoints prefixed with: /api/")
    logger.info(f"💬 Chat endpoint: POST http://{host}:{port}/api/chat")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,  # Auto-reload during development
        log_level="info"
    )
    