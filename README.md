# 📄 Smart PDF Contextual Reader

**Smart PDF Contextual Reader** is a lightweight Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF, ask natural language questions, and receive **context-aware answers grounded in the PDF content and enriched with real-time web search results**.

The project demonstrates a **hybrid RAG pipeline** combining vector search over document embeddings with live web search, wrapped in a simple and intuitive Streamlit interface.

🔗 **Live Demo:** https://smartypdf.streamlit.app/

---

## 🚀 Features

- 📤 **PDF Upload & Processing**
  - Upload any PDF document
  - Automatic text extraction and chunking

- 🧠 **Contextual Question Answering**
  - Ask questions in natural language
  - Answers are grounded in the uploaded PDF

- 🔍 **Hybrid Retrieval (PDF + Web)**
  - Semantic search over PDF using vector embeddings
  - Parallel live web search for additional context

- ⚡ **Fast & Efficient**
  - ChromaDB-based vector similarity search
  - Parallel execution for low-latency responses

- 🖥️ **Simple UI**
  - Clean and interactive Streamlit interface
  - No setup required for end users

---

## 🛠️ Tech Stack

### Frontend
- **Streamlit** – interactive web application UI

### Backend / AI
- **Google Gemini (ChatGoogleGenerativeAI)** – LLM for answer generation  
- **Google Generative AI Embeddings** – semantic vector embeddings  
- **LangChain** – orchestration of LLMs, retrieval, and prompts  

### Retrieval & Search
- **ChromaDB** – vector database for similarity search  
- **DuckDuckGo Search** – live web search integration  

### Utilities
- **PyPDF2** – PDF text extraction  
- **Python** – core application logic  
- **dotenv** – environment variable management  

---

## 🧠 Concepts Used

- **Retrieval-Augmented Generation (RAG)**  
  Combines retrieval (PDF + web) with LLM generation to reduce hallucinations.

- **Document Chunking & Overlap**  
  Large PDFs are split into overlapping chunks for better semantic recall.

- **Vector Embeddings & Similarity Search**  
  Text chunks are embedded and searched using ChromaDB.

- **Hybrid Context Fusion**  
  PDF context is prioritized, with web search used for enrichment.

- **Parallel Retrieval**  
  PDF retrieval and web search run concurrently to reduce latency.

---

## 📂 Project Structure

```text
smart-pdf-contextual-reader/
│
├── ai-agent.py            # Streamlit app entry point & core logic
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md
