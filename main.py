
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from pathlib import Path

from dotenv import load_dotenv
from langgraph_flow import graph
import google.generativeai as genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------
# Environment
# -----------------------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)

model = genai.GenerativeModel("models/gemini-3.6-flash")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Zepto AI Support Assistant")

# -----------------------------
# Load FAISS Vector Store
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTOR_PATH = "vectorstore"

vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Request / Response Schema
# -----------------------------
class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Zepto AI Support Assistant Running"}


# -----------------------------
# Ask Endpoint (LangGraph Version)
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):

    # Retrieve relevant chunks from FAISS
    docs = vectorstore.similarity_search(request.question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    # Run LangGraph
    result = graph.invoke(
        {
            "question": request.question,
            "context": context,
        }
    )

    return AnswerResponse(
        answer=result["answer"],
        sources=[doc.metadata.get("source", "PDF") for doc in docs],
        confidence=0.95,
    )