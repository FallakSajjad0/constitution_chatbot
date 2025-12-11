# models.py
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    max_length: Optional[int] = 1000

class ChatResponse(BaseModel):
    answer: str
    success: bool = True
    error: Optional[str] = None
    processing_time: Optional[float] = None
    sources: List[str] = []

class HealthResponse(BaseModel):
    status: str
    ready: bool
    pdf_count: Optional[int] = 0
    error: Optional[str] = None

class SystemInfo(BaseModel):
    status: str
    version: str = "1.0.0"
    service: str = "PDF Document Assistant"
    endpoints: List[dict]

class PDFSource(BaseModel):
    name: str
    chunks: int

class StatsResponse(BaseModel):
    status: str
    total_chunks: int
    pdf_files: int
    sources: List[PDFSource]
    articles_found: List[str]
    system_ready: bool

class BatchRequestItem(BaseModel):
    question: str
    session_id: str = "default"

class BatchResponseItem(BaseModel):
    question: str
    answer: str
    success: bool
    processing_time: Optional[float] = None
    error: Optional[str] = None

class BatchResponse(BaseModel):
    status: str
    total_questions: int
    successful: int
    responses: List[BatchResponseItem]