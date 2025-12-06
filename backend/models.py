from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question about the Constitution of Pakistan")
    context: Optional[str] = Field(
        None,
        description="Optional additional context (rarely used in RAG)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What does Article 19 say about freedom of speech?",
                    "context": None
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The AI's answer based only on constitutional documents")
    sources: List[str] = Field(
        default=[],
        description="List of document sources (e.g., article filenames) used in the answer"
    )
    relevant_docs: Optional[int] = Field(
        None,
        description="Number of retrieved constitutional articles used"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Article 19 of the Constitution of Pakistan guarantees every citizen the right to freedom of speech and expression...",
                    "sources": ["article_19.pdf", "part_ii.pdf"],
                    "relevant_docs": 3
                }
            ]
        }
    }