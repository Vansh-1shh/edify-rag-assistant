# Edify — RAG-Powered Academic Assistant

A fourth-year CSE project demonstrating Retrieval-Augmented Generation (RAG) for academic study.

## Features

| Feature | Description |
|---|---|
| **PDF Upload** | Upload lecture notes / textbook chapters as PDF |
| **URL Ingestion** | Add any web page (Wikipedia, docs, articles) as a knowledge source |
| **Multi-source** | Mix PDFs and URLs in a single session |
| **RAG Chatbot** | Ask questions — answers grounded strictly in your content, with **source citations** |
| **Question Generator** | MCQs, Short Answer, Long Answer — Easy / Medium / Hard — download as `.txt` |
| **Summary Generator** | Short, Detailed, or Bullet-Point Notes — download as `.txt` |
| **Key Topics Extractor** | Auto-detect the 6 main topics in a document |
| **Persistent Indexes** | FAISS indexes saved per document — reload instantly |
| **Dashboard** | Live metrics: docs, chunks, questions generated, chat turns |

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
Gemini 2.0 Flash  (Google GenAI API)
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
