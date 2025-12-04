from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = Field(None, description="Optional context for the query")

    class Config:
        schema_median = {
            "example": {
                "query": "What is the capital of France?",
                "context": "Geography"
            }
        }

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The answer to the query")
    sources: List[str] = Field(..., description="List of sources for the answer")

    class Config:
        schema_median = {
            "example": {
                "answer": "The capital of France is Paris.",
                "sources": ["https://example.com/france", "https://example.com/paris"]
            }
        }

