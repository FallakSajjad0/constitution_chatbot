# Pakistan Constitution RAG Chatbot

This app answers questions using Retrieval-Augmented Generation over the Constitution of Pakistan PDF, with LangChain + ChromaDB, a FastAPI backend, and a simple HTML frontend. It chunks the PDF, vectorizes it, and retrieves relevant passages to ground the LLM’s answers.

## Requirements
- Python 3.14
- An OpenAI API key in `.env` (see below)

## Setup
1. Create `.env`:
