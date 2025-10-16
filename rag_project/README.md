# Gilead Agentic QA (AWS) - Enhanced RAG System Documentation

## Overview

This codebase implements a **Retrieval-Augmented Generation (RAG) system** built with Streamlit for an interactive web UI. The system enables natural language querying over uploaded PDF documents and CSV datasets, leveraging AWS services for embeddings, LLM inference, and storage. It supports document ingestion (PDFs), vector-based retrieval, and generation using Bedrock models, alongside a SQL interface for CSV data analysis.

Key features:
- **PDF Q&A Tab**: Upload and index PDFs, then query them conversationally.
- **CSV SQL Tab**: Upload CSVs, load into a local DuckDB database, and convert natural language questions to SQL queries.
- **Analytics Tab**: View cache statistics, feedback metrics, and query history.
- **Cache Management Tab**: Manage query cache, clear entries, and handle feedback.
- **Caching**: Session-based caching for repeated queries (case-insensitive, top-k dependent).
- **Feedback System**: Thumb up/down for responses, with storage for analytics.

The system is designed for local development and deployment (e.g., via Streamlit), with persistence in ChromaDB for vectors and DuckDB for CSV data. It assumes AWS credentials are configured via `.env`.

## System Architecture

### High-Level Flow
1. **Ingestion (PDFs)**:
   - Upload PDFs (single or batch).
   - Extract text using PyPDF.
   - Chunk text (overlapping, 500 chars default).
   - Generate embeddings via Bedrock.
   - Store in ChromaDB collection (`pdf_docs`).

2. **Querying (PDFs)**:
   - Embed query.
   - Retrieve top-k chunks from ChromaDB.
   - Build context and prompt Claude for response.
   - Cache results.

3. **Ingestion (CSVs)**:
   - Upload CSVs.
   - Load into DuckDB tables (named from filenames).
   - Validate and sample data.

4. **Querying (CSVs)**:
   - Use NL to generate SQL via Claude.
   - Validate SQL (syntax/safety).
   - Execute in DuckDB and display results as DataFrame.

5. **Shared Features**:
   - Sidebar: Index stats (chunks, docs).
   - Caching: In-memory (Streamlit session state).
   - Logging: Console prints for debugging.
   - Feedback: Stored in session for analytics.

### Data Flow Diagram (Conceptual)
```
User Upload (PDF/CSV) → Extraction/Chunking → Embed (Bedrock) → Store (ChromaDB/DuckDB)
↓
User Query (NL) → Embed Query → Retrieve → Prompt LLM (Bedrock) → Response + Cache
↓
Analytics/Cache Mgmt → Metrics & Cleanup
```

## Services Used

- **AWS Bedrock**: Core inference service for embeddings and LLM.
  - Embeddings: Generates vectors for text chunks/queries.
  - LLM: Generates responses and SQL queries.
- **AWS S3**: File storage for uploaded PDFs (via `boto3` client).
  - Used in upload utils for single/batch uploads.
- **ChromaDB**: Persistent vector database (local `./chroma_db`).
  - Stores document embeddings, metadata (file, chunk_id).
- **DuckDB**: In-memory/persistent SQL engine for CSV data.
  - Loads CSVs as tables; executes safe SQL queries.
- **No external APIs beyond AWS**: All processing is local/AWS-integrated.

## Models Used

- **Embedding Model**: `amazon.titan-embed-text-v2:0`
  - Dimension: 1536 (used for zero-vector fallback).
  - Purpose: Text-to-vector for RAG retrieval.
- **LLM Model**: `anthropic.claude-3-sonnet-20240229-v1:0`
  - Purpose: Response generation for PDF queries; NL-to-SQL conversion.
  - Prompting: Structured for context-aware answers (PDF) or SQL-only output (CSV).
  - Parameters: `max_tokens=2000` (PDF), `500` (SQL); `temperature=0.1` (SQL for determinism).

Models are configured via `.env` (e.g., `EMBEDDING_MODEL_ID`, `BEDROCK_MODEL_ID`).

## Frameworks and Libraries Used

| Category | Library | Version/Purpose |
|----------|---------|-----------------|
| **UI/Web** | Streamlit | Interactive dashboard with tabs, uploaders, metrics, spinners. |
| **PDF Processing** | pypdf | Text extraction from PDFs (page-by-page). |
| **Vector DB** | chromadb | Persistent collection for embeddings; similarity search. |
| **AWS Integration** | boto3 | Bedrock runtime, S3 clients. |
| **SQL/CSV** | duckdb | Lightweight OLAP DB for CSV loading/execution. |
| **SQL Parsing** | sqlparse | Syntax validation for generated SQL. |
| **Env/Config** | python-dotenv | Loads `.env` vars (AWS keys, models). |
| **Data Handling** | pandas | DataFrames for query results, samples. |
| **Utils** | json, os, re | Standard lib for parsing, paths, sanitization. |
| **Requirements** | See `requirements.txt` (e.g., streamlit, chromadb, pypdf, boto3, duckdb). |

No additional installs needed beyond `pip install -r requirements.txt`.

## File Structure and Key Files

```
rag_project/
├── .env                  # AWS keys, model IDs, S3 bucket.
├── .gitignore            # Ignores __pycache__, chroma_db, etc.
├── README.md             # This doc (update as needed).
├── requirements.txt      # Dependencies.
├── data/                 # Sample PDFs (lim.pdf, quant.pdf, scaling.pdf).
├── chroma_db/            # Persistent vector DB (auto-created).
├── csv_workspace/        # DuckDB files and uploaded CSVs (auto-created).
├── embed.py              # Embedding func (Bedrock Titan); index builder (PDF chunk/embed/store).
├── extract_text.py       # PDF text extraction (pypdf; page-formatted).
├── main.py               # Streamlit app: Tabs, UI logic, caching, feedback.
├── query.py              # RAG query: Embed → Retrieve → Prompt Claude → Response.
├── upload.py             # S3 upload utils (single/batch PDFs).
├── utils.py              # S3 helpers (upload/list/download), PDF extraction alt (PyPDF2).
└── watcher.py            # (Unused?) File watcher for auto-reindexing.
```

- **Core Entry**: Run `streamlit run main.py`.
- **PDF Samples**: `lim.pdf`, `quant.pdf`, `scaling.pdf` in `data/` for testing.

## Usage Guide

### Setup
1. Clone/copy codebase.
2. `pip install -r requirements.txt`.
3. Configure `.env`:
   ```
   AWS_REGION=us-west-2
   S3_BUCKET=your-bucket
   EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
   BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
   ```
4. Set AWS credentials (`aws configure` or env vars).

### Running the App
- `streamlit run main.py`.
- **PDF Tab**: Upload PDFs → Ingest → Query (e.g., "Summarize lim.pdf").
- **CSV Tab**: Upload CSVs → Load → NL Query (e.g., "Top sales by product") → Generate/Execute SQL.
- **Analytics**: View metrics post-queries.
- **Cache**: Repeats use cache; clear via tab.

### Testing
- Upload sample PDFs from `data/`.
- Query: "What is quantization?" (hits quant.pdf).
- CSV: Upload a sample CSV, query: "Average salary".

### Limitations/Notes
- **Local Only**: No production scaling; Chroma/DuckDB are file-based.
- **Safety**: SQL validation blocks DML/DDL.
- **Caching**: Session-only (resets on rerun).
- **Errors**: Check console for Bedrock/S3 issues.
- **Extensions**: Add Kendra for citations (code stubs exist).

## Deployment Considerations
- **Cloud**: Deploy Streamlit on EC2/ECS; use RDS for DuckDB if needed.
- **Scaling**: Shard Chroma; use Bedrock queues.
- **Security**: IAM roles for Bedrock/S3; validate uploads.

For issues, check console logs or extend `query.py`/`embed.py`. Contribute via PRs!