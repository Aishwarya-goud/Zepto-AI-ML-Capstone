# Zepto AI Support Assistant (RAG Chatbot)

## Project Overview

This project is an AI-powered customer support assistant for Zepto built using Retrieval-Augmented Generation (RAG). It answers customer queries by retrieving relevant information from PDF documents and generating responses based on the retrieved context.

The project uses FastAPI for the API backend, Streamlit for the chatbot interface, FAISS for vector search, HuggingFace sentence embeddings for semantic retrieval, and LangGraph for workflow routing.

---

## Architecture

The RAG pipeline consists of four stages:

### 1. Ingestion

**File:** `app.py`

* Upload PDF documents.
* Read PDF using PyPDFLoader.
* Split documents into chunks using RecursiveCharacterTextSplitter.

### 2. Embedding

**File:** `app.py`

* Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
* Store embeddings inside FAISS vector database.

### 3. Retrieval

**File:** `main.py`

* Receive a customer question.
* Perform similarity search on FAISS.
* Retrieve top matching document chunks.

### 4. Generation

**Files:** `main.py`, `langgraph_flow.py`

* LangGraph classifies intent.
* Policy questions are routed to retrieval.
* General questions are routed to direct response.
* In MOCK mode, canned responses are returned.
* In real mode, Gemini generates answers using retrieved context.

---

## LangGraph Flow

Nodes:

1. `classify_intent`
2. `retrieve_and_answer`
3. `direct_answer`

Routing:

* `policy_question` → retrieve_and_answer
* `general_question` → direct_answer

---

## Prompt Template

System Role:
You are a Zepto Customer Support AI assistant.

Task:
Answer only using the retrieved PDF context.

Context:
{retrieved_context}

Question:
{user_question}

Negative Constraint:
Do not use outside knowledge. If the answer is unavailable in the PDF, say so.

Few-shot Example:

Question:
What is the refund policy?

Answer:
The refund policy states that refunds are processed according to the policy described in the uploaded document.

---

## MOCK Mode

Default:
`MOCK_LLM=1`

Behavior:

* No Gemini API calls.
* Retrieval still runs.
* Policy questions return:
  `Based on the retrieved context: ...`
* General questions return:
  `This is a canned response for a general question.`

---

## Example API Calls

### Policy Question

POST `/ask`

```json
{
  "question": "What is the return policy?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["return_policy.pdf"],
  "confidence": 0.95
}
```

### General Question

POST `/ask`

```json
{
  "question": "Tell me a joke."
}
```

Response:

```json
{
  "answer": "This is a canned response for a general question.",
  "sources": [],
  "confidence": 0.95
}
```

---

## Running the Project

### Streamlit

```bash
streamlit run app.py
```

### FastAPI

```bash
uvicorn main:app --reload
```

API Documentation:

`http://127.0.0.1:8000/docs`

---

## Docker

Build:

```bash
docker build -t zepto-support-assistant .
```

Run:

```bash
docker run -p 8000:8000 zepto-support-assistant
```

Open:

`http://localhost:8000/docs`

---

## Technologies Used

* FastAPI
* Streamlit
* LangGraph
* FAISS
* HuggingFace Sentence Transformers
* Gemini API
* Python 3.13
