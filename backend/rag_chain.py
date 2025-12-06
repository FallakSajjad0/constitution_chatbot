# backend/rag_chain.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

load_dotenv()  # loads .env from project root

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Put HF_TOKEN in constitution_chatbot/.env")

# DB path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Best free model Dec 2025
client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

SYSTEM_PROMPT = """You are a Supreme Court Justice of Pakistan specializing only in the 1973 Constitution.
Answer using ONLY the context below. Quote exact Articles. Never add external knowledge.

If the answer is not in context → reply exactly:
"This specific information is not present in the retrieved constitutional provisions."

Answer in 4–7 clear sentences. Always start with the relevant Article number."""

def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {os.path.basename(doc.metadata.get('source',''))}\n{doc.page_content.strip()}"
        for i, doc in enumerate(docs)
    )

def answer_question(query: str) -> dict:
    try:
        docs = retriever.invoke(query)
        if not docs:
            return {"answer": "No relevant articles found.", "sources": [], "relevant_docs": 0}

        context = format_docs(docs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + f"\n\nContext:\n{context}"},
            {"role": "user", "content": query}
        ]

        resp = client.chat_completion(
            messages=messages,
            max_tokens=600,
            temperature=0.05,
            top_p=0.95,
            stream=False
        )

        sources = list({os.path.basename(doc.metadata.get("source","Unknown")) for doc in docs})

        return {
            "answer": resp.choices[0].message.content.strip(),
            "sources": sources,
            "relevant_docs": len(docs)
        }
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "relevant_docs": 0}