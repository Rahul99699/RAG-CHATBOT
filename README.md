# DocuRAG — Intelligent PDF Question Answering System | [LINK](https://rag-chatbot-production-4782.up.railway.app/)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions using natural language. The system retrieves relevant document chunks using semantic similarity and uses an LLM to generate context-grounded answers.

## Architecture

```text
                 ┌───────────────┐
                 │   Gradio UI   │
                 └───────┬───────┘
                         │
                    Upload PDF
                         ↓
                 ┌───────────────┐
                 │    PyMuPDF    │
                 │ PDF → Text    │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │   Chunking    │
                 │ 550 chars     │
                 │ overlap = 10  │
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │ Sentence      │
                 │ Transformer   │
                 │ 384D Embedding│
                 └───────┬───────┘
                         ↓
                 ┌───────────────┐
                 │   ChromaDB    │
                 │ Chunks +      │
                 │ Embeddings    │
                 └───────┬───────┘
                         │
                         │ Similarity Search
                         ↑
                    Question
                         ↓
                 Question Embedding
                         ↓
                   Top-K Chunks
                         ↓
              Context + Question
                         ↓
                 ┌───────────────┐
                 │   Groq LLM    │
                 └───────┬───────┘
                         ↓
                      Answer
```

## Features

* PDF text extraction and recursive chunking
* 384-dimensional Sentence Transformer embeddings
* Semantic similarity search using ChromaDB
* Context-grounded question answering with Groq LLM
* Interactive Gradio interface
* Secure API key management using environment variables

## Tech Stack

**Python · PyMuPDF · Sentence Transformers · ChromaDB · Groq API · LangChain · Gradio**

## Run Locally

```bash
git clone <your-repository-url>
cd RAG-CHATBOT

pip install -r requirements.txt

python app.py
```

Create a `.env` file:

```env
GROQ_KEY=your_api_key
```

> Never commit `.env`, API keys, `venv/`, or the local `chroma_db/` directory to GitHub.
