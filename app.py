import streamlit as st
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI PDF Chatbot", page_icon="📄", layout="wide")

# ---------------- GEMINI ----------------
load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ API Key Missing")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-3.6-flash")

# ---------------- SESSION ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📚 PDF Chatbot")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []

# ---------------- PROCESS PDF ----------------
if uploaded_file and st.session_state.vectorstore is None:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            pdf_path = temp_pdf.name

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

    st.success("✅ PDF Ready!")

# ---------------- CHAT ----------------
st.title("🤖 AI PDF Chatbot")
st.write("Ask anything from your uploaded PDF.")

# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
question = st.chat_input("Ask your question...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.vectorstore is None:
            st.error("Upload a PDF first.")

        else:
            with st.spinner("Thinking..."):

                docs = st.session_state.vectorstore.similarity_search(
                    question,
                    k=3
                )

                context = "\n\n".join(
                    doc.page_content for doc in docs
                )

                prompt = f"""
You are an AI assistant.

Answer ONLY using the PDF context below.

PDF Context:
{context}

Question:
{question}

If the answer is not found in the PDF, reply:
'This information is not available in the uploaded PDF.'
"""

                response = model.generate_content(prompt)
                answer = response.text

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )