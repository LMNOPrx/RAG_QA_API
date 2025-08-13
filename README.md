# RAG QA API

A **Dockerized FastAPI application** implementing an advanced **Retrieval-Augmented Generation (RAG)** pipeline for document-based Question Answering.  
Supports PDFs, DOCX, and Email formats, combining **FiD architecture**, **hybrid retrieval (FAISS + BM25)**, **Cohere Rerank**, and **multi-query expansion** using **LLaMA-3-70B (Groq)** for accurate, context-rich answers.

---

## 🚀 Features

- **FiD-based RAG pipeline** for improved answer generation from multiple context chunks.
- **Hybrid Retrieval**: Combines FAISS dense vector search with BM25 keyword search.
- **Contextual Compression** with **Cohere Rerank** for high-relevance results.
- **Multi-Query Expansion** using LLaMA-3-70B to broaden recall.
- **Document Ingestion from URL** (PDF, DOCX, EML, MSG) with metadata extraction and text chunking.
- **Persistent FAISS Indexing** for faster repeat queries.
- **Token-based Authentication** for secure API access.
- **Dockerized** for consistent deployment.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python
- **Retrieval**: FAISS, BM25, Cohere Rerank
- **LLMs**: LLaMA-3-70B (Groq API)
- **Embeddings**: Cohere `embed-english-v3.0`
- **Document Loaders**: LangChain Community (PDF, DOCX, Email)
- **Containerization**: Docker

---

## 📂 Project Structure

```
├── app/
│   ├── rag_pipeline.py     # Core RAG logic
│   ├── config.py           # API key & settings management
│   ├── main.py.            # FastAPI server and API routes
│   └── __init__.py
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker build instructions
├── .gitignore
├── .dockerignore
└── README.md               # Project documentation
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/LMNOPrx/RAG_QA_API.git
cd RAG_QA_API
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Environment Variables
Create a .env file in the root directory.
```env
COHERE_API_KEY_1=your_cohere_key_for_embeddings
COHERE_API_KEY_2=your_cohere_key_for_rerank
GROQ_API_KEY_1=your_groq_key_for_query_expansion
GROQ_API_KEY_2=your_groq_key_for_answer_generation
```

### 4️⃣ Run Locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Access API docs at: http://localhost:8000/docs

---

## 🐳 Docker Deployment

### Build the image
```bash
docker build -t RAG_QA_API .
```

### Run the container
```bash
docker run -d -p 8000:8000 --env-file .env RAG_QA_API
```

---

## 📡 API Usage

#### Endpoint: POST /hackrx/run
#### Auth: Bearer Token (set in EXPECTED_TOKEN in main.py)

### Request
```json
{
  "documents": "https://example.com/sample.pdf",
  "questions": [
    "What is the claim process?",
    "What is the grace period for premium payment?"
  ]
}
```

### Response
```json
{
  "answers": [
    "The claim process involves submitting required documents...",
    "The grace period is 30 days from the due date..."
  ]
}
```

---

## 📌 Notes

- Uses persistent FAISS indexes to avoid reprocessing documents.
- Multi-query expansion boosts retrieval recall for complex questions.
- Designed for quick adaptation to different LLM providers.























