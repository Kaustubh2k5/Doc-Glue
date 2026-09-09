<div style="width: 500px; height: 100px; overflow: hidden; position: relative;">
  <img src="Assets/Frame 2608495 (1).png" alt="Profile Banner" style="width: 100%; position: absolute; top: 50%; transform: translateY(-50%);">
</div>
<h1> Don't Let Large Context files slow down your work. </h1>
<p> Doc-Glue Acts like a Middle layer between your context and your Agent.</p>

## Pipeline Architecture

  <img src="Assets/arch.png" alt="Pipeline arch">

### Processing Pipeline
1. Document Ingestion and Parsing: Parses incoming PDFs using Docling (with PyMuPDF fallback).
2. Quality Filtering: Removes low-information chunks and noise before extraction.
3. Grounded Fact Extraction: Extracts structured atomic facts with page and snippet lineage.
4. Vector Indexing and Deduplication: Embeds facts and stores them in PostgreSQL with pgvector using SHA-256 content deduplication.
5. Cluster-First Reconciliation: Groups related facts vectorially and evaluates cross-document agreement and contradictions.
6. Fact Ledger: Exposes queryable reference cards via FastAPI endpoints and Streamlit UI.

## Tech Stack

* Backend and API: Python 3.11, FastAPI, Uvicorn, Pydantic v2
* PDF Parsing: Docling (Primary), PyMuPDF (Fallback)
* LLM and Embeddings: OpenRouter / OpenAI API, Gemini, Ollama
* Database and Vector Search: PostgreSQL 16 with pgvector, Psycopg 3
* Frontend: Streamlit
* Containerization: Docker, Docker Compose

## Local Setup

All you need is Docker.

1. Get a free API key from OpenRouter (or OpenAI).
2. Create a `.env` file from `.env.example` and set your API key:
   ```bash
   cp .env.example .env
   ```
3. Run Docker Compose:
   ```bash
   docker compose up --build
   ```

Access the Streamlit UI at `http://localhost:8501` and FastAPI docs at `http://localhost:8000/docs`.
