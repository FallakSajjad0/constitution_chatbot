import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------
# 1️⃣  LOAD EMBEDDINGS
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# ---------------------------------------------------
# 2️⃣  LOAD CHROMA DB (from project root)
# ---------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fixed __file__
DB_DIR = os.path.join(PROJECT_ROOT, "db")

print(f"📁 Loading vector database from: {DB_DIR}")

if not os.path.exists(DB_DIR):
    raise FileNotFoundError(f"Vector database not found at {DB_DIR}. Run ingestion first.")

vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------
# 3️⃣  SIMPLE LLM FUNCTION
# ---------------------------------------------------
def simple_llm(inputs: dict) -> str:
    context = inputs.get("context", "")
    question = inputs.get("question", "")
    
    if context:
        passages = context.split("\n\n")
        if passages:
            return f"Based on the constitution: {passages[0][:300]}..."
    return "I don't have enough information to answer that question."

# ---------------------------------------------------
# 4️⃣  PROMPT TEMPLATE
# ---------------------------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a constitutional law assistant. Answer based only on this context:
    
    {context}
    
    Question: {question}
    
    Answer concisely. If the answer isn't in the context, say "I cannot find this information in the constitution documents."
    """
)

# ---------------------------------------------------
# 5️⃣  HELPER FUNCTIONS
# ---------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---------------------------------------------------
# 6️⃣  FINAL ANSWER FUNCTION
# ---------------------------------------------------
def answer_question(query: str):
    try:
        docs = retriever.invoke(query)
        
        if not docs:
            return {
                "answer": "I cannot find relevant information in the constitution documents.",
                "sources": []
            }
        
        context = format_docs(docs)
        formatted_prompt = prompt_template.format(context=context, question=query)
        result = simple_llm({"prompt": formatted_prompt, "context": context, "question": query})
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
        
        return {
            "answer": result,
            "sources": sources,
            "relevant_docs": len(docs)
        }
        
    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": [],
            "error": str(e)
        }
