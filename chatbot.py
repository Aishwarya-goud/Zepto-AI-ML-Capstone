from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Load FAISS Vector Store
# -----------------------------
vector_db = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("Vector Store Loaded Successfully!")

# -----------------------------
# Chat Function
# -----------------------------
def ask_question(question):

    # Search similar chunks
    docs = vector_db.similarity_search(question, k=2)

    print("\nAnswer:\n")

    for doc in docs:
        print(doc.page_content)
        print("-" * 50)

# -----------------------------
# Test Chatbot
# -----------------------------
user_question = input("Ask a question: ")
ask_question(user_question)