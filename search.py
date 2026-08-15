import os

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq


# ==========================================
# Environment variables
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


# ==========================================
# Validate environment variables
# ==========================================

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set."
    )

if not CHROMA_API_KEY:
    raise ValueError(
        "CHROMA_API_KEY environment variable is not set."
    )

if not CHROMA_TENANT:
    raise ValueError(
        "CHROMA_TENANT environment variable is not set."
    )

if not CHROMA_DATABASE:
    raise ValueError(
        "CHROMA_DATABASE environment variable is not set."
    )


# ==========================================
# Groq client
# ==========================================

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# ==========================================
# Chroma Cloud client
# ==========================================

chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)


# ==========================================
# Chroma collection
# ==========================================

collection = chroma_client.get_or_create_collection(
    name="pdf_document"
)


# ==========================================
# Embedding model
# ==========================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# Answer function
# ==========================================

def answer(question):

    if not question or not question.strip():
        return "Please enter a question."

    question = question.strip()

    # --------------------------------------
    # Convert question to embedding
    # --------------------------------------

    question_embedding = model.encode(question)

    # --------------------------------------
    # Search Chroma Cloud
    # --------------------------------------

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=3
    )

    # --------------------------------------
    # Check whether anything was found
    # --------------------------------------

    documents = results.get("documents", [])

    if not documents or not documents[0]:
        return "I don't know based on the provided document."

    # --------------------------------------
    # Get retrieved chunks
    # --------------------------------------

    chunks = documents[0]

    context = "\n\n".join(chunks)

    # --------------------------------------
    # Create RAG prompt
    # --------------------------------------

    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the context provided below.

Context:
{context}

Question:
{question}

Rules:
- Use only information from the context.
- Do not make up information.
- If the answer is not present in the context, say:
"I don't know based on the provided document."
"""

    # --------------------------------------
    # Groq request
    # --------------------------------------

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    # --------------------------------------
    # Return answer
    # --------------------------------------

    return response.choices[0].message.content
