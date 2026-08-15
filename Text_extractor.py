import os
import uuid

import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Environment variables
# ==========================================

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")


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
# Embedding model
# ==========================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# Chroma Cloud
# ==========================================

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)


collection = client.get_or_create_collection(
    name="pdf_document"
)


# ==========================================
# Process PDF
# ==========================================

def process_pdf(pdf_path):

    if not pdf_path:
        return "Please upload a PDF first."

    try:

        # ----------------------------------
        # Extract PDF text
        # ----------------------------------

        with pymupdf.open(pdf_path) as data:

            text = ""

            for page in data:
                text += page.get_text() + "\n"

        if not text.strip():
            return "Could not extract any text from this PDF."

        # ----------------------------------
        # Split text into chunks
        # ----------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=550,
            chunk_overlap=50,
            length_function=len
        )

        data_split = splitter.split_text(text)

        if not data_split:
            return "No text chunks were created."

        # ----------------------------------
        # Create embeddings
        # ----------------------------------

        embeddings = model.encode(
            data_split,
            show_progress_bar=False
        )

        # ----------------------------------
        # Unique document ID
        # ----------------------------------

        document_id = str(uuid.uuid4())

        ids = [
            f"{document_id}_{i}"
            for i in range(len(data_split))
        ]

        # ----------------------------------
        # Metadata
        # ----------------------------------

        metadatas = [
            {
                "document_id": document_id,
                "source": os.path.basename(pdf_path)
            }
            for _ in data_split
        ]

        # ----------------------------------
        # Store in Chroma Cloud
        # ----------------------------------

        collection.add(
            documents=data_split,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas
        )

        return (
            f"PDF processed successfully.\n"
            f"{len(data_split)} chunks stored in Chroma Cloud."
        )

    except Exception as e:

        return f"Error processing PDF: {str(e)}"
