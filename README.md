# AI Support Ticket + Conversational RAG API

A production-style FastAPI backend for AI-powered support ticket extraction and conversational document Q&A — featuring centralized LLM abstraction, semantic caching, long-term vector memory with rolling summarization, session management, and async parallel retrieval.

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
- [Async Indexing](#async-indexing)
- [Background Tasks](#background-tasks)
- [Why FAISS](#why-faiss)
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

- Async document indexing via background tasks
- FAISS `IndexHNSWFlat` for document vectors (approximate, scalable)
- FAISS `IndexFlatIP` for memory vectors (exact, per-session)
- Session-based long-term memory with rolling summarization
- Semantic chunk retrieval with cosine similarity
- Parallel retrieval via `asyncio.gather()`

---

### Centralized LLM Service

All LLM invocations are routed through a single `llm_service.py`.

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

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| LLM Abstraction | `app/services/llm_service.py` |
| Embeddings | Sentence Transformers — `BAAI/bge-base-en-v1.5` |
| Document Vectors | FAISS `IndexHNSWFlat` |
| Memory Vectors | FAISS `IndexFlatIP` |
| Caching | Redis (exact) + FAISS embeddings (semantic) |
| Rate Limiting | SlowAPI (5 req/min) |
| Async Processing | `asyncio.gather()`, `asyncio.to_thread()` |
| Background Tasks | FastAPI `BackgroundTasks` |

---

## Project Structure

```
app/
│
├── controllers/
│   ├── rag_controller.py           # RAG + session HTTP handlers
│   └── extract_controller.py       # Ticket extraction HTTP handlers
│
├── services/
│   ├── llm_service.py              # Centralized Groq client + generation methods
│   ├── rag_service.py              # PDF retrieval + prompt construction
│   ├── memory_service.py           # Long-term FAISS memory + summarization
│   ├── semantic_cache_service.py   # Embedding-based cache layer
│   ├── vector_service.py           # FAISS index management for documents
│   ├── extract_service.py          # Ticket extraction orchestration
│   ├── cache_service.py            # Redis exact cache logic
│   └── redis_service.py            # Redis client initialization
│
├── models/
│   ├── rag_model.py                # Pydantic request/response models
│   └── extract_model.py            # Pydantic extraction models
│
├── utils/
│   ├── file_utils.py               # File I/O helpers
│   ├── hash_utils.py               # MD5 cache key generation
│   └── path_utils.py               # Consistent path resolution
│
├── config/
│   └── rag_config.py               # CHUNK_SIZE, SUMMARY_TRIGGER, RECENT_HISTORY
│
├── vector_store/
│   ├── indexes/                    # Per-document FAISS IndexHNSWFlat files
│   └── metadata/                   # Per-document chunk metadata JSON
│
├── memory_store/                   # Per-session FAISS index + metadata + summary
│
├── uploads/                        # Raw uploaded PDF files
│
└── main.py
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Redis
- [Groq API key](https://console.groq.com)

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_key
MODEL=llama-3.3-70b-versatile

REDIS_HOST=localhost
REDIS_PORT=6379
```

### Run Redis

```bash
docker run -p 6379:6379 redis
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Server

```bash
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

### Document Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/rag/upload` | Upload a PDF for async background indexing |
| `GET` | `/rag/documents` | List all fully indexed documents |
| `POST` | `/rag/ask` | Ask a question against an indexed document |

**Upload response:**

```json
{ "message": "File uploaded", "status": "processing" }
```

**Ask request:**

```json
{
  "filename": "resume.pdf",
  "question": "What backend technologies does he know?",
  "session_id": "session1"
}
```

**Ask response:**

```json
{ "answer": "The document mentions FastAPI, Redis, and FAISS..." }
```

---

### Session Management Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/rag/sessions` | List all sessions with stored memory |
| `GET` | `/rag/session/{session_id}` | Paginated conversation history for a session |
| `DELETE` | `/rag/session/{session_id}` | Delete session memory, FAISS index, and summary |
| `POST` | `/rag/session/search` | Semantic search over a session's memory |

**List sessions:**

```json
{ "sessions": ["session1", "session2"] }
```

**Get session history:**

```
GET /rag/session/session1?page=1&page_size=10
```

```json
{
  "session_id": "session1",
  "page": 1,
  "results": [
    {
      "question": "What databases does he know?",
      "answer": "The document mentions Redis and FAISS."
    }
  ]
}
```

**Delete session:**

```json
{ "message": "Session session1 deleted." }
```

**Semantic search request:**

```json
{
  "session_id": "session1",
  "query": "What did we discuss about databases?"
}
```

**Semantic search response:**

```json
{
  "results": [
    {
      "question": "Which DB technologies are mentioned?",
      "answer": "Redis and FAISS are used in this project."
    }
  ]
}
```

---

## Caching Architecture

Two independent cache layers reduce redundant LLM calls at different levels of granularity.

### Layer 1 — Exact Cache (Redis)

```
Cache key: md5(filename + question)

HIT  → return cached answer immediately (sub-millisecond)
MISS → proceed to semantic cache
```

Handles identical repeated queries with near-zero overhead.

### Layer 2 — Semantic Cache (FAISS + Embeddings)

```
Incoming question
    ↓
Embed with BAAI/bge-base-en-v1.5
    ↓
Cosine similarity search over stored question embeddings
    ↓
Similarity ≥ threshold?
    ├─ YES → return cached answer
    └─ NO  → LLM call → save to both caches (background)
```

Handles paraphrased or semantically equivalent questions without an LLM call.

**Example matches:**

| Incoming | Cached Hit |
|---|---|
| `"What databases does he know?"` | `"Which DB technologies are mentioned?"` |
| `"Summarize his work experience"` | `"What jobs has he had?"` |

Both cache writes happen asynchronously via `BackgroundTasks` after the response is returned.

---

## Conversational Memory

Each session has a dedicated FAISS index that persists across server restarts.

### Memory Files (per session)

```
memory_store/
    session1.index          # FAISS IndexFlatIP (exact cosine similarity)
    session1.json           # Metadata: question, answer, timestamp per turn
    session1_summary.txt    # LLM-generated rolling summary of older turns
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
Summary written to: memory_store/session1_summary.txt
    ↓
Recent turns retained in FAISS index unchanged
```

### Prompt Injection Order

```
1. System instructions
2. Rolling summary       ← distilled context from older turns
3. Retrieved memories    ← semantically relevant recent turns
4. Document chunks       ← relevant PDF sections
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

Sessions are identified by a `session_id` string supplied with each `/rag/ask` request. Each session owns its FAISS index, metadata JSON, and optional summary file. All state is stored on the local filesystem and persists across restarts.

- **List** all active sessions via `GET /rag/sessions`
- **Browse** paginated history via `GET /rag/session/{session_id}`
- **Delete** all session state via `DELETE /rag/session/{session_id}` — removes the FAISS index, JSON metadata, and summary file
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
Cosine similarity search over session1.index
    ↓
Return top-K matching conversation turns
```

This enables accurate recall even when the user cannot remember exact prior phrasing — useful for long research sessions or multi-day workflows.

---

## Async Indexing

PDF uploads return immediately. All indexing happens as a background task.

```
POST /rag/upload
    ↓
Save PDF to uploads/
    ↓
Return { "status": "processing" }  ← immediate response

    ↓ (background)
Extract text via PyMuPDF
    ↓
Chunk text (word-based, CHUNK_SIZE = 400)
    ↓
Embed chunks with BAAI/bge-base-en-v1.5
    ↓
Build FAISS IndexHNSWFlat
    ↓
Save to: vector_store/indexes/<filename>.index
         vector_store/metadata/<filename>.json
```

`GET /rag/documents` returns only files for which both an index and metadata file exist, confirming completed indexing.

---

## Background Tasks

The following operations run via FastAPI `BackgroundTasks` after the HTTP response is returned to the client, keeping request latency minimal:

| Task | Trigger |
|---|---|
| Exact cache write (Redis) | After every LLM response |
| Semantic cache write (FAISS) | After every LLM response |
| Memory persistence (FAISS + JSON) | After each conversation turn |
| Memory summarization | When session memory exceeds `SUMMARY_TRIGGER` |

---

## Why FAISS

FAISS (Facebook AI Similarity Search) provides efficient dense vector search in-process, with no external vector database dependency.

| Index Type | Used For | Rationale |
|---|---|---|
| `IndexHNSWFlat` | Document chunks | Approximate nearest neighbor — sub-linear search time; scales to large document collections |
| `IndexFlatIP` | Session memory | Exact inner product search — session indexes are small; precision preferred over approximation |

Both use L2-normalized vectors so inner product equals cosine similarity. `faiss.normalize_L2()` is applied before all add and search operations.

Running FAISS in-process means zero network overhead and no infrastructure dependency for vector search.

---

## Why SentenceTransformers

`BAAI/bge-base-en-v1.5` is used as the single embedding model across every subsystem:

| Subsystem | Use |
|---|---|
| PDF chunk embeddings | Semantic document retrieval |
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
| FAISS indexes | In-process, single node |
| Redis cache | Single instance |
| Memory store | Local filesystem |
| Embeddings | CPU inference |
| LLM | Groq API (rate-limited) |

### Scaling Path

| Bottleneck | Recommended Solution |
|---|---|
| FAISS single-node | Migrate to Qdrant, Weaviate, or Pinecone |
| Redis single instance | Redis Cluster or Redis Sentinel |
| CPU embeddings | GPU inference server or hosted embedding API |
| LLM rate limits | Multiple API keys or OpenAI fallback via `llm_service.py` |
| Filesystem memory store | Postgres + pgvector |
| Single FastAPI process | Gunicorn multi-worker + async Redis |
| Background tasks | Celery + Redis or RabbitMQ broker |

The centralized `llm_service.py` and modular service layer keep most of these migrations localized to a single file or service.

---

## Future Improvements

- [ ] Hybrid retrieval (BM25 + dense vectors)
- [ ] Streaming LLM responses (SSE)
- [ ] Cross-encoder reranking
- [ ] Multi-document retrieval across files in one query
- [ ] Multi-user authentication (JWT / OAuth2)
- [ ] Postgres persistence for metadata and memory
- [ ] Celery distributed background workers
- [ ] Memory importance scoring and pruning
- [ ] WebSocket chat interface
- [ ] GPU-accelerated embeddings
- [ ] Metadata filtering on retrieval
- [ ] Async Redis client (`aioredis`)
- [ ] Distributed vector store
- [ ] Memory summarization quality evaluation

---

## License

MIT
