# Financial Q&A Assistant

A CLI-based question answering system over earnings call transcripts, built with RAG (Retrieval-Augmented Generation), ChromaDB, and OpenAI.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running Tests](#running-tests)

---

## Overview

This system allows users to ask natural language questions about earnings call transcripts via a CLI. It supports:

- Querying across multiple companies and quarters
- Context-aware multi-turn conversations (remembers previous questions)
- Guardrails to block off-topic questions and prompt injection attempts
- Filtering responses by company, quarter, or fiscal year

**Tech Stack**

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (local, persistent) |
| CLI | Typer + Rich |
| PDF Parsing | pdfplumber |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│         Guardrails          │  ← Sanitize, classify intent, rewrite query
└─────────────┬───────────────┘
              │ allowed
    ▼
┌─────────────────────────────┐
│          Embedder           │  ← text-embedding-3-small
└─────────────┬───────────────┘
              │ query vector
    ▼
┌─────────────────────────────┐
│         VectorStore         │  ← ChromaDB similarity search + metadata filter
└─────────────┬───────────────┘
              │ top-k chunks
    ▼
┌─────────────────────────────┐
│         RAG Engine          │  ← Build context prompt + call LLM
└─────────────┬───────────────┘
              │
    ▼
   Answer + Sources
```

**Ingestion Pipeline**

```
PDF File
    │
    ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Parser  │ →  │ Chunker  │ →  │ Embedder │ →  │  Store   │
│          │    │          │    │          │    │          │
│ Extract  │    │ Q+A pair │    │ Batch    │    │ ChromaDB │
│ metadata │    │ chunking │    │ embed    │    │ upsert   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

---

## Project Structure

```
financial-qa/
│
├── main.py                     # Entry point
├── requirements.txt
├── .env.example
│
├── core/                       # Shared business logic
│   ├── config.py               # Settings from environment variables
│   ├── llm.py                  # OpenAI LLM wrapper (streaming + retry)
│   ├── embedder.py             # text-embedding-3-small wrapper
│   ├── vectorstore.py          # ChromaDB abstraction layer
│   ├── rag.py                  # RAG engine (retrieve + generate)
│   └── guardrails.py           # Input sanitization + intent classifier
│
├── ingestion/                  # Data pipeline
│   ├── parser.py               # PDF → text + metadata extraction
│   ├── chunker.py              # Speaker-aware Q+A pair chunking
│   └── pipeline.py             # Orchestrates parse → chunk → embed → store
│
├── cli/
│   └── main.py                 # Typer CLI (chat, ingest, list commands)
│
├── data/
│   └── transcripts/            # Place PDF files here
│       └── <scrip_code>/       # e.g. 532400 (Birlasoft)
│           ├── Q1/
│           ├── Q2/
│           ├── Q3/
│           └── Q4/
│
└── tests/
    ├── test_llm.py
    ├── test_ingestion.py
    └── test_guardrails.py
```

---

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd financial-qa

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Usage

### 1. Ingest Transcripts

Place your PDF transcript files in the data folder following this structure:

```
data/transcripts/
└── 532400/          ← BSE scrip code
    ├── Q1/
    │   └── transcript.pdf
    ├── Q2/
    │   └── transcript.pdf
    ├── Q3/
    │   └── transcript.pdf
    └── Q4/
        └── transcript.pdf
```

Then run:

```bash
# Ingest entire directory (all companies, all quarters)
python main.py ingest ./data/transcripts/

# Ingest a single company
python main.py ingest ./data/transcripts/532400/

# Ingest a single quarter
python main.py ingest ./data/transcripts/532400/Q1/

# Ingest a single PDF directly
python main.py ingest ./data/transcripts/532400/Q1/transcript.pdf
```

### 2. List Indexed Transcripts

```bash
python main.py list
```

Example output:

```
┌────────────┬─────────┬─────────────┬────────────┐
│ Company    │ Quarter │ Fiscal Year │ Date       │
├────────────┼─────────┼─────────────┼────────────┤
│ Birlasoft  │ Q1      │ FY25        │ 2024-07-31 │
│ Birlasoft  │ Q2      │ FY25        │ 2024-10-23 │
│ Birlasoft  │ Q3      │ FY25        │ 2025-01-30 │
│ Birlasoft  │ Q4      │ FY25        │ 2025-06-05 │
└────────────┴─────────┴─────────────┴────────────┘
Total chunks in store: 124
```

### 3. Chat

```bash
# Chat across all indexed transcripts
python main.py chat

# Filter to a specific company
python main.py chat --company Birlasoft

# Filter to a specific quarter
python main.py chat --company Birlasoft --quarter Q1

# Filter to a fiscal year
python main.py chat --fy FY25

# Use a different model
python main.py chat --model gpt-4o
```

**In-chat commands:**

| Command | Description |
|---|---|
| `/sources` | Show source chunks used in the last answer |
| `/clear` | Reset conversation history |
| `/help` | Show available commands |
| `/quit` | Exit |

**Example session:**

```
You: What was Birlasoft's revenue in Q1 FY25?

A: Birlasoft's revenue in Q1 FY25 was INR 12,274 million (~$159.1M),
   reflecting a 2.6% sequential decline but 5.1% year-on-year growth,
   as stated by Kamini Shah on July 31, 2024.

   📎 2 source(s) · Birlasoft · Q1 FY25 · /sources for details

You: How does that compare to Q2?

A: In Q2 FY25, revenue grew to $163.3 million, a 2.6% sequential
   improvement driven by delayed ERP projects resuming...
```

---

## Configuration

All settings are controlled via the `.env` file:

```bash
# Required
OPENAI_API_KEY=your_key_here

# LLM settings
LLM_MODEL=gpt-4o-mini          # Model to use
MAX_TOKENS=2048                 # Max response tokens
TEMPERATURE=0.2                 # Lower = more factual

# RAG settings
TOP_K=5                         # Number of chunks to retrieve
SIMILARITY_THRESHOLD=0.55       # Cosine distance cutoff (lower = stricter)

# ChromaDB
CHROMA_DIR=./data/chroma        # Where vectors are stored
CHROMA_COLLECTION=earnings_transcripts
```

**Tuning tips:**

- If answers are missing context → increase `SIMILARITY_THRESHOLD` (e.g. 0.6)
- If answers contain irrelevant info → decrease `SIMILARITY_THRESHOLD` (e.g. 0.4)
- If answers are too short → increase `MAX_TOKENS`
- For faster/cheaper responses → keep `LLM_MODEL=gpt-4o-mini`
- For more accurate responses → use `LLM_MODEL=gpt-4o`

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_llm.py -v
python -m pytest tests/test_ingestion.py -v
python -m pytest tests/test_guardrails.py -v
```

**No API key required for tests** — all LLM calls are mocked.

Expected output:

```
tests/test_guardrails.py    23 passed
tests/test_ingestion.py     14 passed
tests/test_llm.py            7 passed
─────────────────────────────────────
Total                       44 passed
```