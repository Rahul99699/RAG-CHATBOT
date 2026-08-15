# 📚 DocuRAG — Intelligent PDF Question Answering System

*DocuRAG* is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions using natural language. It extracts and chunks document content, converts the chunks into semantic embeddings, retrieves the most relevant information using vector similarity search, and uses an LLM to generate context-grounded answers.

🔗 *Live Demo:* https://rag-chatbot-production-4782.up.railway.app/

---

## ✨ Features

* 📄 Upload and process PDF documents
* 🔍 Extract text using *PyMuPDF*
* ✂️ Recursive text chunking for efficient retrieval
* 🧠 Generate *384-dimensional semantic embeddings*
* 🔎 Perform semantic similarity search using *ChromaDB*
* 🤖 Generate context-aware answers using *Groq LLM*
* 💬 Interactive question-answering interface with *Gradio*
* 🔐 Secure API key management using environment variables
* 🚀 Deployed application accessible through a web interface

---

## 🏗️ Architecture

text
                         ┌─────────────────┐
                         │    Gradio UI    │
                         └────────┬────────┘
                                  │
                             Upload PDF
                                  ↓
                         ┌─────────────────┐
                         │    PyMuPDF      │
                         │   PDF → Text    │
                         └────────┬────────┘
                                  │
                                  ↓
                         ┌─────────────────┐
                         │    Chunking     │
                         │ 550 chars       │
                         │ overlap = 10    │
                         └────────┬────────┘
                                  │
                                  ↓
                    ┌─────────────────────────┐
                    │  Sentence Transformer  │
                    │   384D Embeddings       │
                    └────────────┬────────────┘
                                 │
                                 ↓
                         ┌─────────────────┐
                         │    ChromaDB     │
                         │ Chunks + Vectors│
                         └────────┬────────┘
                                  │
                                  │ Similarity Search
                                  ↑
                              User Query
                                  │
                                  ↓
                         ┌─────────────────┐
                         │ Question        │
                         │ Embedding       │
                         └────────┬────────┘
                                  │
                                  ↓
                         ┌─────────────────┐
                         │   Top-K Chunks  │
                         └────────┬────────┘
                                  │
                                  ↓
                         ┌─────────────────┐
                         │ Context + Query │
                         └────────┬────────┘
                                  │
                                  ↓
                         ┌─────────────────┐
                         │    Groq LLM     │
                         └────────┬────────┘
                                  │
                                  ↓
                              Answer


---

## 🔄 How It Works

### 1. PDF Upload

The user uploads a PDF through the Gradio interface.

### 2. Text Extraction

*PyMuPDF* extracts readable text from the uploaded PDF.

### 3. Text Chunking

The extracted text is divided into smaller chunks of approximately *550 characters* with an overlap of *10 characters*. This makes individual sections easier to retrieve during semantic search.

### 4. Embedding Generation

Each text chunk is converted into a *384-dimensional vector embedding* using a Sentence Transformer model.

### 5. Vector Storage

The generated embeddings and their corresponding text chunks are stored in *ChromaDB*, which acts as the vector database.

### 6. Semantic Retrieval

When the user asks a question, the question is converted into an embedding. ChromaDB compares this embedding against stored document embeddings and retrieves the most relevant *Top-K chunks*.

### 7. Context-Grounded Generation

The retrieved chunks are combined with the user's question and sent to the *Groq LLM*.

### 8. Final Answer

The LLM generates an answer based on the retrieved document context, reducing the need for the model to rely solely on its pretrained knowledge.

---

## 🧰 Tech Stack

| Technology                | Purpose                               |
| ------------------------- | ------------------------------------- |
| *Python*                | Core application logic                |
| *PyMuPDF*               | PDF text extraction                   |
| *Sentence Transformers* | Semantic text embeddings              |
| *ChromaDB*              | Vector database and similarity search |
| *Groq API*              | LLM-powered answer generation         |
| *LangChain*             | RAG and LLM workflow components       |
| *Gradio*                | Interactive web interface             |
| *Railway*               | Application deployment                |

---

## 📁 Project Structure

text
RAG-CHATBOT/
│
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not committed)
├── .gitignore             # Ignored files and directories
├── chroma_db/             # Local vector database (not committed)
└── README.md              # Project documentation


---

## 🚀 Run Locally

### 1. Clone the Repository

bash
git clone <your-repository-url>
cd RAG-CHATBOT


### 2. Create a Virtual Environment

bash
python -m venv venv


Activate it on Windows:

bash
venv\Scripts\activate


On macOS/Linux:

bash
source venv/bin/activate


### 3. Install Dependencies

bash
pip install -r requirements.txt


### 4. Configure Environment Variables

Create a .env file in the project root:

env
GROQ_KEY=your_api_key


### 5. Run the Application

bash
python app.py


The Gradio interface will then be available locally.

---

## 🔐 Security

Never commit sensitive credentials or generated local data to GitHub.

Add the following to your .gitignore:

text
.env
venv/
__pycache__/
chroma_db/


*Important:* Never expose your Groq API key directly inside source code.

---

## 🧠 RAG Pipeline

The core pipeline can be summarized as:

text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
Vector Database
 ↓
Question Embedding
 ↓
Similarity Search
 ↓
Top-K Relevant Chunks
 ↓
Context + Question
 ↓
LLM
 ↓
Grounded Answer


---

## 🎯 Why RAG?

Traditional LLM applications may struggle when answering questions about private or previously unseen documents.

DocuRAG addresses this by retrieving relevant information from the uploaded document *before generating the answer*.

This approach helps:

* Ground responses in the provided document
* Reduce irrelevant responses
* Work with user-provided knowledge
* Avoid sending the entire document to the LLM for every question
* Build scalable document-question-answering applications

---

## 🌐 Live Demo

Try DocuRAG here:

**https://rag-chatbot-production-4782.up.railway.app/**

Upload a PDF, ask a question, and DocuRAG will retrieve the relevant document context and generate an answer.

---

## 🔮 Future Improvements

* [ ] Support multiple PDF documents simultaneously
* [ ] Add document-level metadata and filtering
* [ ] Improve chunking using token-based strategies
* [ ] Add source/page citations to generated answers
* [ ] Add conversation memory for follow-up questions
* [ ] Implement hybrid keyword + semantic retrieval
* [ ] Add reranking for retrieved chunks
* [ ] Support additional document formats such as DOCX and TXT
* [ ] Add authentication and user-specific document collections
* [ ] Add evaluation metrics for retrieval and answer quality

---

## 👨‍💻 Author

*Rahul*

Built as an exploration of *Retrieval-Augmented Generation, vector databases, semantic search, and LLM-based applications*.

--

## ⭐ If You Like This Project

Give the repository a ⭐ and feel free to explore, fork, and improve the project!
