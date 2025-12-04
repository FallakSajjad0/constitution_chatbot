import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------
# 1️⃣  LOAD EMBEDDINGS (sentence-transformers/all-mpnet-base-v2)
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# ---------------------------------------------------
# 2️⃣  LOAD CHROMA DB
# ---------------------------------------------------
DB_DIR = os.path.join(os.path.dirname(__file__), "db")
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------
# 3️⃣  LOAD LOCAL CAUSAL MODEL (for text generation)
# ---------------------------------------------------
CAUSAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "causal_model")
print("🔄 Loading causal model from:", CAUSAL_MODEL_PATH)

# Check if the model directory exists
if not os.path.exists(CAUSAL_MODEL_PATH):
    raise FileNotFoundError(f"Model directory not found at {CAUSAL_MODEL_PATH}. Please ensure the model is downloaded.")

# Load tokenizer and model from the local path
tokenizer = AutoTokenizer.from_pretrained(CAUSAL_MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    CAUSAL_MODEL_PATH,
    device_map="auto",
    torch_dtype="auto",
    low_cpu_mem_usage=True
)

# Create a pipeline for text generation
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.2,
    num_return_sequences=1,
)

# LangChain wrapper for local model
def local_llm(inputs: dict) -> str:
    formatted_prompt = inputs["prompt"]
    result = llm_pipeline(formatted_prompt)
    # Extract only the generated text after the prompt
    response = result[0]['generated_text'].strip()
    # Remove the prompt from the response if it's included
    if formatted_prompt in response:
        response = response.replace(formatted_prompt, "").strip()
    return response

# ---------------------------------------------------
# 4️⃣  PROMPT TEMPLATE
# ---------------------------------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Context: {context}

    Question: {question}

    Answer the question based solely on the context provided above. Do not repeat the context or the question in your answer. Provide a concise and accurate response.
    """
)

# ---------------------------------------------------
# 5️⃣  BUILD CHAIN
# ---------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | local_llm
)

# ---------------------------------------------------
# 6️⃣  FINAL ANSWER FUNCTION
# ---------------------------------------------------
def answer_question(query: str):
    try:
        docs = retriever.invoke(query)
        if not docs:
            return {"answer": "No relevant information found.", "sources": []}
        context = format_docs(docs)
        formatted_prompt = prompt_template.format(context=context, question=query)
        result = local_llm({"prompt": formatted_prompt})

        # Post-process the result to ensure it is concise and relevant
        sentences = [sentence.strip() for sentence in result.split('.') if sentence.strip()]
        concise_answer = sentences[0] if sentences else result
        return {"answer": concise_answer, "sources": [doc.metadata.get("source", "unknown") for doc in docs]}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}
