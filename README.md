# AI Support Ticket + Conversational RAG API

A production-style FastAPI backend for AI-powered support ticket extraction and conversational document Q&A — featuring session-scoped document management, PostgreSQL + pgvector storage, parent-child chunking, hybrid BM25 + dense retrieval, multi-query expansion with RRF reranking, LLM thinking traces, and a multi-theme React frontend.

---

## Table of Contents

- [Features](#features)
- [Stack](#stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Caching Architecture](#caching-architecture)
- [Conversational Memory](#conversational-memory)
- [Memory Summarization](#memory-summarization)
- [Session Management](#session-management)
- [Semantic Session Search](#semantic-session-search)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Async Indexing](#async-indexing)
- [Background Tasks](#background-tasks)
- [Why PostgreSQL + pgvector](#why-postgresql--pgvector)
- [Why SentenceTransformers](#why-sentencetransformers)
- [Why a Centralized LLM Service](#why-a-centralized-llm-service)
- [Scalability](#scalability)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Features

### Support Ticket Extraction

Extracts structured data from unstructured support text — tickets, emails, bug reports.

**Output fields:**

| Field | Description |
|---|---|
| `intent` | Detected intent of the request |
| `entities` | Key entities mentioned |
| `priority_score` | Numeric urgency score (0–10) |
| `suggested_action` | Recommended next step |

Powered by Groq LLM with function/tool calling, Redis exact cache, and semantic cache.

---

### Conversational PDF RAG

Upload PDFs and query them conversationally with full session continuity.

- Session-scoped document ownership via `Session → SessionDocument → Document → DocumentChunk`
- SHA-256 checksum deduplication — re-uploading an identical file reuses existing chunks and embeddings
- Async document indexing via background tasks
- Parent-child chunking for context-aware retrieval
- Hybrid BM25 + pgvector dense retrieval per query
- Multi-query expansion with LLM-generated variants; results merged via Reciprocal Rank Fusion (RRF)
- LLM thinking traces streamed as SSE events alongside answer tokens
- Session-based long-term memory with rolling summarization

---

### Centralized LLM Service

All LLM invocations route through a single `llm_service.py`.

- Single Groq client initialization at startup
- Model name loaded once from environment
- Exposes `generate_response()` and `generate_tool_response()`
- Decouples LLM provider from all business logic
- Simplifies retries, logging, and provider swapping

---

### Long-Term Vector Memory

Every session maintains a persistent FAISS memory store on disk.

- Each conversation turn is embedded and added to the session index
- Semantic retrieval injects only relevant memories into prompts
- Memory grows with the conversation, bounded by summarization

---

### Memory Summarization

When a session's memory exceeds the configured threshold, older entries are compressed by the LLM into a rolling summary, keeping prompt sizes bounded as conversations grow.

---

### Multi-Theme React Frontend

A full chat UI served alongside the API.

- 5 CSS variable themes: Void, Terminal, Ember, Arctic, Noir — persisted in localStorage
- ThinkingDrawer: collapsible panel showing live LLM reasoning steps, auto-opens on stream
- Session sidebar with custom session naming and drag-and-drop PDF upload
- Document selection scoped to the active session via `session_id + document_id`

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| LLM Abstraction | `app/services/llm_service.py` |
| Embeddings | Sentence Transformers — `BAAI/bge-base-en-v1.5` |
| Document + Memory Vectors | PostgreSQL + pgvector |
| Full-Text Search | PostgreSQL full-text search (BM25) |
| Caching | Redis (exact) + pgvector embeddings (semantic) |
| Rate Limiting | SlowAPI (5 req/min) |
| Async Processing | `asyncio.gather()`, `asyncio.to_thread()` |
| Background Tasks | FastAPI `BackgroundTasks` |
| Containerization | Docker, Docker Compose |

---

## Project Structure

```
.
├── .env
├── .gitignore
├── requirements.txt
├── requirements-linux.txt
│
└── app/
    ├── main.py
    │
    ├── clients/
    │   └── groq_client.py              # Groq SDK client initialization
    │
    ├── config/
    │   └── settings.py                 # Environment config, chunk sizes, thresholds
    │
    ├── controllers/
    │   ├── rag_controller.py           # RAG + session HTTP handlers
    │   └── extract_controller.py       # Ticket extraction HTTP handlers
    │
    ├── models/                         # SQLAlchemy ORM models
    │   ├── session.py                  # Session (UUID PK, name, created_at)
    │   ├── session_document.py         # SessionDocument join table
    │   ├── document.py                 # Document (UUID PK, checksum, status)
    │   ├── document_chunk.py           # DocumentChunk (parent_id, child_id, pgvector embedding)
    │   ├── conversation_memory.py      # ConversationMemory (per-turn vector + metadata)
    │   ├── conversation_summary.py     # ConversationSummary (rolling LLM summary)
    │   ├── extract_model.py            # Pydantic extraction request/response models
    │   ├── rag_model.py                # Pydantic RAG request/response models
    │   └── session_model.py            # Pydantic session models
    │
    ├── schemas/
    │   └── tools.py                    # Groq tool/function call schemas
    │
    ├── services/
    │   ├── database.py                 # PostgreSQL connection setup and session factory
    │   ├── llm_service.py              # Centralized Groq client + generation methods
    │   ├── ai_service.py               # Embedding model initialization and inference
    │   ├── rag_service.py              # Core RAG orchestration (chunking, ingestion, retrieval)
    │   ├── langchain_rag_service.py    # LangChain-based retrieval pipeline variant
    │   ├── vector_search_service.py    # pgvector similarity search + BM25 full-text search
    │   ├── memory_service.py           # Long-term FAISS memory + rolling summarization
    │   ├── pdf_service.py              # PDF text extraction via PyMuPDF
    │   ├── semantic_cache_service.py   # Embedding-based semantic cache layer
    │   ├── cache_service.py            # Redis exact cache logic
    │   └── rate_limit.py               # SlowAPI rate limiter setup
    │
    ├── utils/
    │   ├── file_utils.py               # File I/O helpers
    │   └── helpers.py                  # Shared utility functions
    │
    ├── vector_store/
    │   ├── indexes/                    # Legacy FAISS IndexHNSWFlat files (pre-migration)
    │   ├── metadata/                   # Legacy chunk metadata JSON (pre-migration)
    │   └── bm25/                       # Legacy BM25 pickle indexes (pre-migration)
    │
    └── memory_store/                   # Per-session FAISS index + metadata + summary
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- Redis
- [Groq API key](https://console.groq.com)

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key
MODEL=llama-3.3-70b-versatile

REDIS_HOST=localhost
REDIS_PORT=6379

DATABASE_URL=postgresql://user:password@localhost:5432/ragdb
```

### Run with Docker Compose

```bash
docker compose up
```

This starts PostgreSQL (with pgvector), Redis, and the FastAPI server together.

### Run Manually

```bash
# Start dependencies
docker run -p 6379:6379 redis
docker run -e POSTGRES_PASSWORD=password -p 5432:5432 ankane/pgvector

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## API Reference

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |

```json
{ "message": "FastAPI working!" }
```

---

### Support Ticket Extraction

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/extract` | Extract structured data from one or more support texts |

**Request:**

```json
{
  "texts": [
    "Payment dashboard crashing in production",
    "Need refund for duplicate billing"
  ]
}
```

**Response:**

```json
{
  "cached": false,
  "data": [
    {
      "intent": "Production crash",
      "entities": ["payment dashboard"],
      "priority_score": 10,
      "suggested_action": "Immediate production investigation"
    }
  ]
}
```

---

### Session Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rag/sessions` | Create a new session with an optional custom name |
| `GET` | `/rag/sessions` | List all sessions |
| `GET` | `/rag/session/{session_id}` | Paginated conversation history for a session |
| `DELETE` | `/rag/session/{session_id}` | Delete session and all associated state |
| `POST` | `/rag/session/search` | Semantic search over a session's memory |

**Create session request:**

```json
{ "name": "Q4 Financial Review" }
```

**Create session response:**

```json
{ "session_id": "a1b2c3d4-...", "name": "Q4 Financial Review" }
```

---

### Document Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rag/upload` | Upload a PDF; associates it with a session |
| `GET` | `/rag/sessions/{session_id}/documents` | List documents linked to a session |
| `DELETE` | `/rag/sessions/{session_id}/documents/{document_id}` | Unlink a document from a session |
| `POST` | `/rag/ask` | Ask a question against one or more session documents |

**Upload request:** `multipart/form-data` with `file` + `session_id`

**Upload response:**

```json
{
  "document_id": "f9e8d7c6-...",
  "status": "processing",
  "deduplicated": false
}
```

If an identical file (same SHA-256) was previously uploaded, `deduplicated: true` is returned and existing chunks are reused — no re-embedding occurs.

**Ask request:**

```json
{
  "session_id": "a1b2c3d4-...",
  "document_ids": ["f9e8d7c6-..."],
  "question": "What backend technologies does he know?",
  "stream": true
}
```

**Ask response (streaming SSE):**

```
event: thinking
data: {"step": "Expanding query into 3 variants..."}

event: thinking
data: {"step": "Running hybrid retrieval across expanded queries..."}

event: token
data: {"token": "The document mentions FastAPI"}

event: token
data: {"token": ", Redis, and pgvector..."}

event: done
data: {}
```

---

### Legacy Document Endpoints (filename-based, deprecated)

The original `filename`-based `/rag/ask` and `/rag/documents` endpoints remain available for backward compatibility but are superseded by the session-scoped API above.

---

## Caching Architecture

Two cache layers sit in front of every LLM call.

### Layer 1 — Redis Exact Cache

```
key = md5(document_ids + question)
    ├─ HIT  → return immediately (~0ms)
    └─ MISS → continue to semantic cache
```

### Layer 2 — Semantic Cache

```
Embed question
    ↓
pgvector cosine similarity search over cached questions
    ├─ similarity ≥ threshold → return stored answer (~5–15ms)
    └─ below threshold → LLM call (~800–2000ms)
```

Both cache writes happen asynchronously via `BackgroundTasks` after the response is returned.

**Example semantic hits:**

| Incoming | Cached Hit |
|---|---|
| `"What databases does he know?"` | `"Which DB technologies are mentioned?"` |
| `"Summarize his work experience"` | `"What jobs has he had?"` |

---

## Conversational Memory

Each session has a dedicated FAISS index that persists across server restarts.

### Memory Files (per session)

```
memory_store/
    <session_id>.index          # FAISS IndexFlatIP (exact cosine similarity)
    <session_id>.json           # Metadata: question, answer, timestamp per turn
    <session_id>.txt            # LLM-generated rolling summary of older turns
```

### Memory Write Flow

```
User question + LLM answer
    ↓
Embed (question + answer) with SentenceTransformers
    ↓
Add vector to session IndexFlatIP
    ↓
Append metadata entry to session JSON
    ↓
Persist both to disk (background task)
```

### Memory Read Flow

```
Incoming question
    ↓
Embed question
    ↓
Cosine similarity search over session FAISS index
    ↓
Retrieve top-K relevant prior turns
    ↓
Inject into prompt alongside document chunks and rolling summary
```

`IndexFlatIP` with L2-normalized vectors gives exact cosine similarity — appropriate for per-session memory stores where index size remains manageable and precision is preferred over approximation.

---

## Memory Summarization

Long sessions are compressed to prevent unbounded context window growth.

### Trigger

```python
if len(memories) > SUMMARY_TRIGGER:
    summarize_older_memories(session_id)
```

### Summarization Flow

```
All memories in session
    ↓
Split: older entries | RECENT_HISTORY (most recent N turns)
    ↓
Older entries → LLM summarization call (via llm_service.py)
    ↓
Summary written to: memory_store/<session_id>.txt
    ↓
Recent turns retained in FAISS index unchanged
```

### Prompt Injection Order

```
1. System instructions
2. Rolling summary       ← distilled context from older turns
3. Retrieved memories    ← semantically relevant recent turns
4. Document chunks       ← relevant PDF sections (parent-expanded)
5. Current question
```

### Benefits

| Property | Effect |
|---|---|
| Bounded context | Prompt size stays predictable regardless of session length |
| Distilled recall | Key facts preserved without full history |
| Persistence | Summary survives server restarts |
| Scalability | Enables arbitrarily long conversations |

---

## Session Management

Sessions are created explicitly via `POST /rag/sessions` and carry a UUID primary key. Each session owns its linked documents (via `SessionDocument`), FAISS memory index, and optional summary file.

- **Create** with a custom name via `POST /rag/sessions`
- **List** all active sessions via `GET /rag/sessions`
- **Browse** paginated history via `GET /rag/session/{session_id}`
- **Delete** all session state via `DELETE /rag/session/{session_id}` — removes the DB rows, FAISS index, JSON metadata, and summary file; orphaned documents with no remaining session links are cleaned up automatically
- **Search** prior turns semantically via `POST /rag/session/search`

---

## Semantic Session Search

Users can retrieve relevant prior conversation turns by meaning, not keyword.

```
POST /rag/session/search

Query: "What did we discuss about caching?"
    ↓
Embed query with BAAI/bge-base-en-v1.5
    ↓
Cosine similarity search over session FAISS index
    ↓
Return top-K matching conversation turns
```

This enables accurate recall even when the user cannot remember exact prior phrasing — useful for long research sessions or multi-day workflows.

---

## Retrieval Pipeline

### Parent-Child Chunking

Documents are indexed at two granularities:

- **Child chunks** — small, precise units used for embedding and retrieval
- **Parent chunks** — larger surrounding context expanded and passed to the LLM

When a child chunk is retrieved, its parent is resolved and the full parent block is included in the prompt, giving the LLM broader context without embedding noise from large chunks.

### Hybrid Retrieval

Each query runs two searches in parallel:

```
Query
    ├── pgvector ANN search (dense semantic similarity)
    └── PostgreSQL full-text search (BM25 keyword matching)
        ↓
Results merged by score
```

### Multi-Query Expansion + RRF

```
Original question
    ↓
LLM generates N expanded query variants (preserving intent)
    ↓
Each variant: embed → hybrid retrieval (parallel asyncio)
    ↓
All result sets merged via Reciprocal Rank Fusion (RRF)
    ↓
Deduplicate chunks → resolve parent blocks → build LLM context
```

RRF rewards chunks that rank consistently well across multiple query variants, improving recall for ambiguous or complex questions.

---

## Async Indexing

PDF uploads return immediately. All indexing happens as a background task.

```
POST /rag/upload
    ↓
Compute SHA-256 checksum
    ├─ Match found → reuse existing Document + chunks, return deduplicated: true
    └─ No match ↓
Save PDF to uploads/
    ↓
Return { "status": "processing" }  ← immediate response

    ↓ (background)
Extract text via PyMuPDF
    ↓
Parent-child chunker
    ↓
Embed child chunks with BAAI/bge-base-en-v1.5
    ↓
Store chunks + embeddings in PostgreSQL (pgvector)
    ↓
Update Document.status = "ready"
```

`GET /rag/sessions/{session_id}/documents` returns only documents with `status = ready`.

---

## Background Tasks

The following operations run via FastAPI `BackgroundTasks` after the HTTP response is returned to the client:

| Task | Trigger |
|---|---|
| Exact cache write (Redis) | After every LLM response |
| Semantic cache write (pgvector) | After every LLM response |
| Memory persistence (FAISS + JSON) | After each conversation turn |
| Memory summarization | When session memory exceeds `SUMMARY_TRIGGER` |
| Document ingestion (embed + store) | After PDF upload |

---

## Why PostgreSQL + pgvector

The system originally used FAISS indexes and a local filesystem for document vector storage. This was replaced with PostgreSQL + pgvector to enable:

| Capability | Local FAISS (before) | PostgreSQL + pgvector (now) |
|---|---|---|
| Multi-worker sharing | Not possible (in-process) | Native (shared DB) |
| Hybrid BM25 + dense search | Separate BM25 pickle files | Single query via `tsvector` + `pgvector` |
| Document deduplication | Not supported | SHA-256 checksum column |
| Session-scoped filtering | Filename prefix hacks | `document_id` FK with proper joins |
| Cascade deletes | Manual file cleanup | DB-level ON DELETE CASCADE |
| Persistence guarantees | Background file writes | ACID transactions |

FAISS `IndexFlatIP` is still used for per-session conversation memory, where the index is small, bounded, and does not need to be shared across workers.

---

## Why SentenceTransformers

`BAAI/bge-base-en-v1.5` is used as the single embedding model across every subsystem:

| Subsystem | Use |
|---|---|
| PDF chunk embeddings | Semantic document retrieval |
| Query expansion embeddings | Multi-query parallel retrieval |
| Semantic cache | Question similarity matching |
| Conversation memory | Turn-level semantic search |

Using one model across all subsystems ensures vectors from different contexts live in the same semantic space — enabling meaningful comparisons across document chunks, cached questions, and memory entries without remapping.

---

## Why a Centralized LLM Service

Previously, Groq client initialization was duplicated across multiple service files.

**Before:**

```
extract_service.py  → owns Groq client
rag_service.py      → owns Groq client
memory_service.py   → owns Groq client
```

**After:**

```
extract_service.py  ─┐
rag_service.py       ├──→ llm_service.py → Groq API
memory_service.py   ─┘
```

**Benefits:**

| Concern | Before | After |
|---|---|---|
| Model config | Duplicated in N files | One env var, one place |
| Retry logic | Must add to each service | Add once in `llm_service.py` |
| Logging/tracing | Scattered | Centralized |
| Provider swap | Edit N files | Edit one file |
| Unit testing | Mock N clients | Mock one interface |

---

## Scalability

### Current Design (Single Node)

| Component | Current Constraint |
|---|---|
| pgvector | Single PostgreSQL instance |
| Redis cache | Single instance |
| Memory store | Local FAISS filesystem |
| Embeddings | CPU inference |
| LLM | Groq API (rate-limited) |

### Scaling Path

| Bottleneck | Recommended Solution |
|---|---|
| PostgreSQL single-node | Read replicas or migrate vector search to Qdrant / Pinecone |
| Redis single instance | Redis Cluster or Redis Sentinel |
| CPU embeddings | GPU inference server or hosted embedding API |
| LLM rate limits | Multiple API keys or OpenAI fallback via `llm_service.py` |
| FAISS memory store | Migrate to pgvector (same DB, already available) |
| Single FastAPI process | Gunicorn multi-worker — pgvector is safely shared, unlike FAISS |
| Background tasks | Celery + Redis or RabbitMQ broker |

The centralized `llm_service.py` and modular service layer keep most of these migrations localized to a single file or service.

---

## Future Improvements

- [x] Hybrid retrieval (BM25 + dense vectors)
- [x] Streaming LLM responses (SSE)
- [x] Multi-document retrieval across files in one query
- [x] Postgres persistence for metadata and memory
- [ ] Cross-encoder reranking
- [ ] Multi-user authentication (JWT / OAuth2)
- [ ] Celery distributed background workers
- [ ] Memory importance scoring and pruning
- [ ] GPU-accelerated embeddings
- [ ] Metadata filtering on retrieval
- [ ] Async Redis client (`aioredis`)
- [ ] Memory summarization quality evaluation

---

## License

MIT
