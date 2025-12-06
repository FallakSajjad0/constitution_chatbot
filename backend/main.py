# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from rag_chain import answer_question

app = FastAPI(title="Pakistan Constitution AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

class Response(BaseModel):
    answer: str
    sources: List[str] = []
    relevant_docs: int = 0

@app.get("/")
def home():
    return {"message": "Pakistan Constitution AI Assistant – Running!"}

@app.post("/chat", response_model=Response)
async def chat(req: Query):
    result = answer_question(req.query)
    return Response(**result)