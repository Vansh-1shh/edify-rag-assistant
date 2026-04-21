# Edify — RAG-Powered Academic Assistant

A Third-year CSE Mini-Project demonstrating Retrieval-Augmented Generation (RAG) for academic study.

## Features
- RAG Chatbot with source citations
- Question Generator (MCQ / Short / Long Answer)
- Summary Generator (Short / Detailed / Bullet-Point Notes)
- Multi-user login with SQLite
- PDF upload + Web URL ingestion

## Architecture

```
Input (PDF / URL)
       │
       ▼
Text Extraction
  ├── PDF: PyPDFLoader (LangChain)
  └── URL: requests + BeautifulSoup
       │
       ▼
Chunking (RecursiveCharacterTextSplitter, 500 tok / 50 overlap)
       │
       ▼
Embeddings: sentence-transformers/all-MiniLM-L6-v2  (HuggingFace, runs locally)
       │
       ▼
FAISS Vector Store  (persisted to disk per document)
       │
  ┌────┴────┐
  │  Query  │
  │(Cosine  │
  │ Search) │
  └────┬────┘
       │  top-k chunks (score < 1.5)
       ▼
Prompt Construction  (context + query)
       │
       ▼
Gemini 3 Flash  (Google GenAI API)
       │
       ▼
Response + Source Metadata → Streamlit UI
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your Gemini API key as an env variable
export GEMINI_API_KEY="your_key_here"

# 3. Run
streamlit run app.py
```

## Project Structure

```
edify/
├── app.py                     # Main Streamlit application
├── requirements.txt
├── backend/
│   ├── pdf_processor.py       # PDF loading & chunking
│   ├── web_processor.py       # URL fetching & chunking  ← NEW
│   ├── vector_store.py        # FAISS create/load/merge
│   ├── rag_pipeline.py        # RAG QA with source citations
│   ├── question_generator.py  # Question + key-topic extraction
│   └── summary_generator.py   # Summary generation
└── faiss_indexes/             # Auto-created; persisted vector stores
```

## Key Technical Decisions

- **Local embeddings** (no API cost) via `sentence-transformers/all-MiniLM-L6-v2`
- **FAISS** for fast approximate nearest-neighbour search
- **Score-filtered retrieval**: only chunks with cosine distance < 1.5 are used, reducing hallucination
- **Source metadata** propagated through the pipeline to show the user which chunk each answer came from
- **Multi-source merging**: `add_texts()` on an existing FAISS store allows combining PDFs and URLs without rebuilding

## Tech Stack
Python · Streamlit · LangChain · FAISS · HuggingFace · Gemini 3 Flash · SQLite
