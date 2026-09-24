from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# -----------------------------
# PDF Path
# -----------------------------
pdf_path = "data/pdfs/sample.pdf"

if os.path.exists(pdf_path):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("PDF Loaded Successfully!")
    print("Total Pages:", len(documents))

    # -----------------------------
    # Split into Chunks
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    print("Chunking Completed!")
    print("Total Chunks:", len(chunks))

    # -----------------------------
    # Create Embeddings
    # -----------------------------
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding Model Loaded!")

    # -----------------------------
    # Create FAISS Vector Store
    # -----------------------------
    vector_db = FAISS.from_documents(chunks, embedding_model)

    # Save Database
    vector_db.save_local("vectorstore")

    print("FAISS Vector Store Created Successfully!")
    print("Saved inside chatbot/vectorstore/")

else:
    print("PDF not found!")
    print("Please place sample.pdf inside chatbot/data/pdfs/")