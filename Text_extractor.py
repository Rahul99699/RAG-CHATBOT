import os
import uuid
import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# Chroma Cloud
# ==========================================

client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE")
)

collection = client.get_or_create_collection(
    name="pdf_document"
)


# ==========================================
# Embedding model
# ==========================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# Process PDF
# ==========================================

def process_pdf(pdf_path):

    if pdf_path is None:
        return "Please upload a PDF first."

    try:

        # ==========================================
        # 1. Extract text
        # ==========================================

        with pymupdf.open(pdf_path) as data:

            text = ""

            for page in data:
                text += page.get_text()

        if not text.strip():
            return "Could not extract any text from the PDF."


        # ==========================================
        # 2. Split text
        # ==========================================

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=550,
            chunk_overlap=10,
            length_function=len
        )

        data_split = splitter.split_text(text)

        if not data_split:
            return "No text chunks were created."


        # ==========================================
        # 3. Create embeddings
        # ==========================================

        embeddings = model.encode(data_split)


        # ==========================================
        # 4. DELETE PREVIOUS PDF
        # ==========================================

        old_data = collection.get()

        old_ids = old_data.get("ids", [])

        if old_ids:
            collection.delete(
                ids=old_ids
            )


        # ==========================================
        # 5. Create IDs for new PDF
        # ==========================================

        ids = [
            f"{uuid.uuid4()}_{i}"
            for i in range(len(data_split))
        ]


        # ==========================================
        # 6. Store NEW PDF
        # ==========================================

        collection.add(
            documents=data_split,
            embeddings=embeddings.tolist(),
            ids=ids
        )


        return (
            f"Previous document deleted.\n"
            f"New PDF processed successfully.\n"
            f"{len(data_split)} chunks stored."
        )


    except Exception as e:

        return f"Error processing PDF: {str(e)}"
