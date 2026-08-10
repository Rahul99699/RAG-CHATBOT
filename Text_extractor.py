import uuid

import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Load embedding model ONCE
# ==========================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# Connect to ChromaDB ONCE
# ==========================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_document"
)


# ==========================================
# Process PDF
# ==========================================

def process_pdf(pdf_path):

    # 1. Open PDF
    with pymupdf.open(pdf_path) as data:

        # 2. Extract text
        text = ""

        for page in data:
            text += page.get_text()

    # 3. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=550,
        chunk_overlap=10,
        length_function=len
    )

    data_split = splitter.split_text(text)

    # 4. Create embeddings
    embeddings = model.encode(data_split)

    # 5. Create unique IDs
    ids = [
        f"{uuid.uuid4()}_{i}"
        for i in range(len(data_split))
    ]

    # 6. Store in ChromaDB
    collection.add(
        documents=data_split,
        embeddings=embeddings.tolist(),
        ids=ids
    )

    return f"PDF processed successfully. {len(data_split)} chunks stored."