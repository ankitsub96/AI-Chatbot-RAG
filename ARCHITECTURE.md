# Architecture Overview

## System Classification

```
Conversational Retrieval-Augmented Generation (RAG) System

Properties:
  - Centralized LLM abstraction layer
  - Two-layer semantic caching (Redis exact + FAISS semantic)
  - Long-term vector memory with rolling summarization
  - Async document ingestion pipeline
  - Session-scoped FAISS memory stores
  - Parallel async retrieval
  - Background task offloading
  - Local filesystem persistence (zero external vector DB dependency)
```

---

## Table of Contents

- [High-Level Component Map](#high-level-component-map)
- [Subsystem Index](#subsystem-index)
  - [1. LLM Service Layer](#1-llm-service-layer)
  - [2. Document Ingestion Pipeline](#2-document-ingestion-pipeline)
  - [3. Retrieval Pipeline](#3-retrieval-pipeline)
  - [4. Conversational Memory Lifecycle](#4-conversational-memory-lifecycle)
  - [5. Memory Summarization Lifecycle](#5-memory-summarization-lifecycle)
  - [6. Cache Lifecycle](#6-cache-lifecycle)
  - [7. Vector Persistence Lifecycle](#7-vector-persistence-lifecycle)
  - [8. Session Management System](#8-session-management-system)
  - [9. Semantic Session Search](#9-semantic-session-search)
  - [10. Async Execution Model](#10-async-execution-model)
  - [11. Background Task Model](#11-background-task-model)
  - [12. Prompt Architecture](#12-prompt-architecture)
  - [13. Vector Database Design](#13-vector-database-design)
  - [14. Rate Limiting](#14-rate-limiting)
  - [15. File Storage Architecture](#15-file-storage-architecture)
- [Scaling Bottlenecks](#scaling-bottlenecks)
- [Future Distributed Architecture](#future-distributed-architecture)
- [Design Philosophy](#design-philosophy)

---

## High-Level Component Map

```
                        ┌─────────────────────┐
                        │        Client        │
                        └──────────┬──────────┘
                                   │ HTTP
                                   ▼
                        ┌─────────────────────┐
                        │   FastAPI + SlowAPI  │
                        │     Controllers      │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
           │  RAG Service │ │Extract Service│ │Memory Service│
           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                  │                │                  │
                  └────────────────┼──────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     LLM Service      │
                        │   (llm_service.py)   │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │      Groq API        │
                        └─────────────────────┘

Supporting infrastructure:

  ┌────────────┐   ┌────────────────┐   ┌────────────────┐
  │   Redis    │   │  FAISS Indexes │   │  Memory Store  │
  │ Exact Cache│   │  (documents)   │   │  (per-session) │
  └────────────┘   └────────────────┘   └────────────────┘
```

---

## Subsystem Index

### 1. LLM Service Layer

**File:** `app/services/llm_service.py`

**Purpose:** Centralize all Groq client initialization and LLM invocation behind a single interface.

#### Architecture

```
All services
    │
    ▼
llm_service.py
    ├── initialize Groq client (once at startup)
    ├── load MODEL from environment
    ├── generate_response(messages, **kwargs)
    └── generate_tool_response(messages, tools, **kwargs)
    │
    ▼
Groq API
```

#### Interface

| Method | Purpose |
|---|---|
| `generate_response(messages)` | Standard chat completion |
| `generate_tool_response(messages, tools)` | Function/tool calling |

#### Design Rationale

Previously, Groq client initialization was duplicated in `extract_service.py`, `rag_service.py`, and `memory_service.py`. Centralizing this:

- Eliminates N-client instantiation on startup
- Makes the provider fully swappable by editing one file
- Centralizes retry policy, timeout config, and observability hooks
- Allows the model to be reconfigured at one point without touching business logic

---

### 2. Document Ingestion Pipeline

**Trigger:** `POST /rag/upload`

**Execution:** Async background task (non-blocking)

#### Full Pipeline

```
Client → POST /rag/upload (multipart PDF)
    ↓
Save file to: uploads/<filename>.pdf
    ↓
Return HTTP 200: { "status": "processing" }   ← client unblocked here
    ↓  (BackgroundTask begins)
PyMuPDF: extract full text from all pages
    ↓
Word-based chunker:
    - CHUNK_SIZE = 400 words (configured in rag_config.py)
    - produces list of text chunks with page metadata
    ↓
SentenceTransformers: embed all chunks
    - Model: BAAI/bge-base-en-v1.5
    - Output: float32 numpy arrays
    ↓
faiss.normalize_L2(embeddings)
    ↓
Build FAISS IndexHNSWFlat
    - M = 32 (HNSW connectivity parameter)
    - efConstruction = 200
    ↓
Persist:
    vector_store/indexes/<filename>.index
    vector_store/metadata/<filename>.json
    ↓
Document available in GET /rag/documents
```

#### Completion Detection

`GET /rag/documents` checks for the co-existence of both the `.index` and `.json` files. A document is only listed if both exist, preventing queries against partially indexed files.

---

### 3. Retrieval Pipeline

**Trigger:** `POST /rag/ask`

#### Full Pipeline

```
POST /rag/ask { filename, question, session_id }
    ↓
─────────────── Cache Layer ───────────────
    ↓
Check Redis exact cache: md5(filename + question)
    ├─ HIT  → return cached answer immediately
    └─ MISS ↓
Check semantic cache: embed question → cosine similarity search
    ├─ HIT  → return cached answer
    └─ MISS ↓
─────────────── Parallel Retrieval ────────────────
    ↓
asyncio.gather():
    ├── asyncio.to_thread(load_faiss_index, filename)
    ├── asyncio.to_thread(load_metadata, filename)
    ├── retrieve_memories(session_id, question)
    └── load_summary(session_id)
    ↓
─────────────── Retrieval ─────────────────
    ↓
Embed question
    ↓
FAISS IndexHNSWFlat search → top-K document chunks
    ↓
─────────────── Prompt Construction ───────────────
    ↓
Assemble prompt:
    1. System instructions
    2. Rolling memory summary (if exists)
    3. Relevant retrieved memories
    4. Top-K document chunks
    5. Current question
    ↓
─────────────── LLM Call ──────────────────
    ↓
llm_service.generate_response(messages)
    ↓
─────────────── Post-Processing ───────────
    ↓
BackgroundTasks:
    ├── write exact cache (Redis)
    ├── write semantic cache (FAISS)
    └── persist memory turn (FAISS + JSON)
    ↓
Return answer to client
```

---

### 4. Conversational Memory Lifecycle

**Scope:** Per `session_id`

**Storage:** Local filesystem (`memory_store/`)

#### Data Model (per turn)

```json
{
  "question": "What databases does he know?",
  "answer": "Redis and FAISS are used in this project.",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Memory Write

```
Question + Answer
    ↓
Concatenate: "{question} {answer}"
    ↓
Embed with BAAI/bge-base-en-v1.5
    ↓
faiss.normalize_L2(vector)
    ↓
IndexFlatIP.add(vector)
    ↓
Append to session JSON metadata
    ↓
Persist both to disk (BackgroundTask)
```

#### Memory Read

```
Incoming question
    ↓
Embed question
    ↓
faiss.normalize_L2(question_vector)
    ↓
IndexFlatIP.search(question_vector, top_k)
    ↓
Retrieve metadata entries by returned indices
    ↓
Return list of { question, answer } pairs
```

#### Why IndexFlatIP for Memory

Session memory indexes are small (bounded by `SUMMARY_TRIGGER` + `RECENT_HISTORY`). `IndexFlatIP` provides exact cosine similarity (via L2-normalized inner product) with no approximation error — appropriate where precision matters more than search speed at scale.

---

### 5. Memory Summarization Lifecycle

**Trigger:** `len(session_memories) > SUMMARY_TRIGGER`

**Execution:** Background task

#### Summarization Flow

```
Session memory grows beyond SUMMARY_TRIGGER
    ↓
Split memories:
    ├── older_entries = memories[:-RECENT_HISTORY]
    └── recent_entries = memories[-RECENT_HISTORY:]
    ↓
Build summarization prompt:
    "Summarize these conversation turns concisely: {older_entries}"
    ↓
llm_service.generate_response(summarization_prompt)
    ↓
Write summary to: memory_store/session_id_summary.txt
    ↓
Rebuild FAISS index from recent_entries only
    ↓
Persist updated index and JSON
```

#### Summary Injection in Prompts

```
Prompt assembly:

[System Instructions]
[Rolling Summary]          ← loaded from session_id_summary.txt
[Retrieved Memories]       ← top-K from FAISS semantic search
[Document Chunks]          ← top-K from document FAISS index
[Current Question]
```

#### Bounds Analysis

| Config | Effect |
|---|---|
| `SUMMARY_TRIGGER` | Maximum number of full turns before compression |
| `RECENT_HISTORY` | Number of recent turns always kept verbatim |
| Summary file | Replaces the compressed older turns; size bounded by LLM summarization quality |

The combination means prompt memory usage grows as `O(RECENT_HISTORY)` turns + one summary block, regardless of total conversation length.

---

### 6. Cache Lifecycle

#### Layer 1 — Redis Exact Cache

```
Request arrives
    ↓
key = md5(filename + question)
    ↓
Redis GET key
    ├─ HIT  → return value, skip all further processing
    └─ MISS → continue pipeline
                    ↓
            (after LLM response)
                    ↓
            BackgroundTask: Redis SET key value
```

**Characteristics:**
- O(1) lookup
- Hash collision probability negligible at this scale
- TTL configurable per deployment

#### Layer 2 — Semantic Cache

```
Question
    ↓
Embed with BAAI/bge-base-en-v1.5
    ↓
faiss.normalize_L2(question_vector)
    ↓
Semantic cache FAISS index search
    ↓
Cosine similarity ≥ threshold?
    ├─ YES → return stored answer
    └─ NO  → LLM call
                    ↓
            BackgroundTask:
                - Add question embedding to semantic cache index
                - Store answer in semantic cache metadata
```

**Threshold tuning:**
- Higher threshold = fewer false cache hits, more LLM calls
- Lower threshold = more aggressive caching, risk of slightly mismatched answers
- Threshold should be tuned based on observed query variation in production

#### Combined Cache Hit Rate

```
Request
    ├─ Exact cache hit    → ~0ms
    ├─ Semantic cache hit → ~5–15ms (embedding + FAISS search)
    └─ LLM call           → ~800–2000ms (Groq API)
```

---

### 7. Vector Persistence Lifecycle

#### Document Vectors

```
Ingestion complete
    ↓
FAISS IndexHNSWFlat (in memory)
    ↓
faiss.write_index(index, path)
    → vector_store/indexes/<filename>.index

Chunk metadata list
    ↓
json.dump(metadata)
    → vector_store/metadata/<filename>.json
```

#### Memory Vectors

```
Conversation turn complete
    ↓
IndexFlatIP updated (in memory)
    ↓
BackgroundTask:
    faiss.write_index(index, path)
    → memory_store/<session_id>.index

    json.dump(metadata)
    → memory_store/<session_id>.json
```

#### Load on Request

Indexes are loaded from disk on each relevant request via `asyncio.to_thread()` to avoid blocking the event loop during file I/O.

---

### 8. Session Management System

**State per session:**

```
memory_store/
    <session_id>.index          FAISS IndexFlatIP
    <session_id>.json           Turn metadata array
    <session_id>_summary.txt    LLM rolling summary (optional)
```

#### Endpoint Behavior

| Endpoint | Operation |
|---|---|
| `GET /rag/sessions` | List all session IDs by scanning `memory_store/` for `.json` files |
| `GET /rag/session/{id}` | Load `.json`, return paginated slice |
| `DELETE /rag/session/{id}` | Delete `.index`, `.json`, `_summary.txt` |
| `POST /rag/session/search` | Embed query, search `.index`, return matched turns |

Sessions are fully self-contained on the filesystem. Deleting a session is a filesystem operation with no orphaned state.

---

### 9. Semantic Session Search

**Purpose:** Enable meaning-based retrieval over a session's conversation history.

#### Flow

```
POST /rag/session/search
    { session_id, query }
    ↓
Embed query with BAAI/bge-base-en-v1.5
    ↓
faiss.normalize_L2(query_vector)
    ↓
Load memory_store/<session_id>.index
    ↓
IndexFlatIP.search(query_vector, top_k)
    ↓
Retrieve matching entries from <session_id>.json by index
    ↓
Return list of { question, answer } pairs
```

This uses the same FAISS index maintained for prompt memory injection — no additional index is needed.

---

### 10. Async Execution Model

All I/O-bound and CPU-bound operations are handled to avoid blocking the FastAPI event loop.

#### Parallel Retrieval (per RAG request)

```python
results = await asyncio.gather(
    asyncio.to_thread(load_faiss_index, filename),
    asyncio.to_thread(load_metadata, filename),
    retrieve_memories(session_id, question),     # async
    load_summary(session_id)                      # async
)
```

| Task | Execution Model | Reason |
|---|---|---|
| `load_faiss_index` | `asyncio.to_thread()` | CPU + file I/O; blocking |
| `load_metadata` | `asyncio.to_thread()` | File I/O; blocking |
| `retrieve_memories` | `async` coroutine | Embedding + FAISS search |
| `load_summary` | `async` coroutine | File read |

All four run concurrently. Total retrieval latency is bounded by the slowest task, not their sum.

#### Document Indexing

Ingestion runs as a FastAPI `BackgroundTask` — the HTTP response is returned before indexing begins. Embedding and FAISS construction are CPU-bound and do not block the event loop.

---

### 11. Background Task Model

FastAPI `BackgroundTasks` are used for all non-critical writes. These execute after the HTTP response is sent to the client.

```
HTTP Response returned to client
    ↓  (simultaneously, in background)
    ├── Redis exact cache write
    ├── Semantic cache FAISS write + metadata append
    ├── Memory FAISS write + JSON append
    └── Memory summarization (if triggered)
```

**Latency impact:** Zero — all cache and persistence writes are invisible to the request-response cycle.

**Risk:** If the process exits immediately after a response, background tasks may not complete. For production deployments with persistence requirements, migrate to Celery workers.

---

### 12. Prompt Architecture

Each LLM call constructs a prompt from up to five layers:

```
┌──────────────────────────────────────┐
│         SYSTEM INSTRUCTIONS          │  ← Task framing, behavior rules
├──────────────────────────────────────┤
│          ROLLING SUMMARY             │  ← Distilled older turns (if exists)
├──────────────────────────────────────┤
│        RETRIEVED MEMORIES            │  ← Top-K semantically relevant turns
├──────────────────────────────────────┤
│        DOCUMENT CHUNKS               │  ← Top-K PDF sections from FAISS
├──────────────────────────────────────┤
│         CURRENT QUESTION             │  ← User's current input
└──────────────────────────────────────┘
```

#### Example

```
System:
  You are a helpful assistant. Answer using only the provided context.

Rolling Summary:
  The user has been asking about backend technologies in a software engineering
  resume. Key topics covered: Redis, FAISS, FastAPI, Groq.

Retrieved Memories:
  Q: What databases does he know?
  A: The document mentions Redis and FAISS.

Document Chunks:
  [Page 3] "...experience with Redis for caching and FAISS for vector search..."
  [Page 5] "...built production APIs using FastAPI and Uvicorn..."

Current Question:
  Does he have experience with async Python?
```

The rolling summary prevents the prompt from needing full conversation history while preserving key accumulated context.

---

### 13. Vector Database Design

#### Document Index — `IndexHNSWFlat`

| Property | Value |
|---|---|
| Index type | Hierarchical Navigable Small World (HNSW) |
| Search type | Approximate nearest neighbor |
| Similarity | Cosine (via L2 normalization + inner product) |
| Construction parameter M | 32 (controls graph connectivity) |
| Search parameter efSearch | 50 (controls recall vs. speed tradeoff) |
| Use case | Large document collections with many chunks |

**Why HNSW for documents:** Document indexes grow proportionally to document size. HNSW provides sub-linear search time O(log n) at the cost of approximate results — an acceptable tradeoff when retrieving the top-5 relevant chunks from thousands.

#### Memory Index — `IndexFlatIP`

| Property | Value |
|---|---|
| Index type | Flat (brute-force) |
| Search type | Exact nearest neighbor |
| Similarity | Cosine (via L2 normalization + inner product) |
| Use case | Per-session memory (small, bounded by SUMMARY_TRIGGER) |

**Why FlatIP for memory:** Session memory indexes are small and bounded by `SUMMARY_TRIGGER`. Exact search ensures no relevant memory is missed. The cost of brute-force search is negligible at this scale.

#### Normalization

All vectors are L2-normalized before insertion and search:

```python
faiss.normalize_L2(vectors)
index.add(vectors)     # IndexFlatIP inner product = cosine similarity
```

This unifies the similarity metric across both index types.

---

### 14. Rate Limiting

**Library:** SlowAPI (Starlette-compatible wrapper around `limits`)

**Default:** 5 requests/minute per client IP

Applied at the controller level. Configurable per endpoint. Returns HTTP 429 on breach.

---

### 15. File Storage Architecture

All state is stored on the local filesystem in four directories:

```
uploads/
    <filename>.pdf                      Raw uploaded PDFs

vector_store/
    indexes/
        <filename>.index                FAISS IndexHNSWFlat (document chunks)
    metadata/
        <filename>.json                 Chunk metadata: { page, text }

memory_store/
    <session_id>.index                  FAISS IndexFlatIP (conversation turns)
    <session_id>.json                   Turn metadata: { question, answer, timestamp }
    <session_id>_summary.txt            LLM rolling summary of older turns
```

#### Persistence Guarantees

| Operation | Persistence Timing |
|---|---|
| PDF upload | Synchronous (before response) |
| Document FAISS index | Background task after ingestion |
| Memory FAISS index | Background task after each turn |
| Memory summary | Background task when threshold exceeded |
| Cache writes | Background task after LLM response |

All indexes are reloaded from disk on each request — there is no in-memory index state between requests. This makes the application stateless from a process perspective and resilient to restarts.

---

## Scaling Bottlenecks

| Bottleneck | Current Behavior | At Scale |
|---|---|---|
| FAISS in-process | Fast, no network | Single node; not horizontally scalable |
| Local filesystem memory store | Zero-latency reads | Cannot be shared across multiple worker processes |
| CPU embeddings | Adequate for low-medium traffic | Becomes throughput ceiling under load |
| Redis single instance | Fast for small cache | Needs clustering for high availability |
| BackgroundTasks | Works for single process | Tasks lost on crash; no retry |
| Groq API rate limits | Fine for development | Needs key pooling or fallback under load |

---

## Future Distributed Architecture

```
                     ┌──────────────┐
                     │  Load Balancer│
                     └──────┬───────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ FastAPI  │  │ FastAPI  │  │ FastAPI  │
        │ Worker 1 │  │ Worker 2 │  │ Worker 3 │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             └─────────────┼─────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌────────────┐  ┌──────────────┐  ┌──────────────┐
   │ Redis      │  │  Qdrant /    │  │  Postgres +  │
   │ Cluster    │  │  Pinecone    │  │  pgvector    │
   │ (cache)    │  │  (vectors)   │  │  (memory)    │
   └────────────┘  └──────────────┘  └──────────────┘
                           │
                    ┌──────┴──────┐
                    │   Celery    │
                    │  Workers    │
                    │ (bg tasks)  │
                    └─────────────┘
```

**Migration path:**

| Current | Distributed Replacement |
|---|---|
| `llm_service.py` → Groq | Add OpenAI / Anthropic fallback; no changes elsewhere |
| FAISS IndexHNSWFlat | Qdrant (drop-in semantic search, distributed) |
| FAISS IndexFlatIP (memory) | pgvector in Postgres |
| Local memory JSON | Postgres table |
| FastAPI BackgroundTasks | Celery + Redis broker |
| Single Redis | Redis Cluster |
| CPU SentenceTransformers | Hosted embedding API or GPU inference server |

The service layer design means most of these are single-file replacements.

---

## Design Philosophy

| Principle | Implementation |
|---|---|
| Low latency on the hot path | Two cache layers; async parallel retrieval; background writes |
| Modularity | Each service owns one concern; LLM, memory, cache, and vector are fully independent |
| Local-first | FAISS and filesystem persistence require no external infrastructure |
| Semantic everywhere | Embeddings used for retrieval, caching, memory, and session search |
| Bounded context | Memory summarization prevents unbounded prompt growth |
| Provider independence | `llm_service.py` abstracts the LLM provider behind a stable interface |
| Production patterns | Rate limiting, async indexing, background tasks, structured logging hooks |

The system is designed to run entirely on a single machine for development and small-scale production, while following architectural patterns that translate cleanly to a distributed deployment with minimal refactoring.
