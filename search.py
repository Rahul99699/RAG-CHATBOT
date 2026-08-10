import os

import chromadb
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY")

if not GROQ_KEY:
    raise ValueError("GROQ_KEY not found in .env")


# ==========================================
# 2. Connect to Groq
# ==========================================

groq_client = Groq(
    api_key=GROQ_KEY
)


# ==========================================
# 3. Connect to ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="pdf_document"
)


# ==========================================
# 4. Load embedding model ONCE
# ==========================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# 5. Answer function
# ==========================================

def answer(question):

    # 1. Convert question → embedding
    question_embedding = model.encode(question)

    # 2. Search ChromaDB
    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=3
    )

    # 3. Get retrieved chunks
    chunks = results["documents"][0]

    # 4. Combine chunks
    context = "\n\n".join(chunks)

    # 5. Create prompt
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided document."
"""

    # 6. Send context + question to LLM
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 7. Return answer to Gradio
    return response.choices[0].message.content