# Architecture Overview

## System Classification

```
Conversational Retrieval-Augmented Generation (RAG) System

Properties:
  - Centralized LLM abstraction layer
  - Session-scoped document ownership with UUID primary keys
  - SHA-256 checksum deduplication on upload
  - Parent-child chunking for context-aware retrieval
  - Hybrid BM25 + pgvector dense retrieval
  - Multi-query expansion with Reciprocal Rank Fusion (RRF) reranking
  - LLM thinking traces streamed via SSE
  - Two-layer semantic caching (Redis exact + pgvector semantic)
  - Long-term FAISS vector memory with rolling summarization
  - Async document ingestion pipeline
  - Parallel async retrieval across expanded queries
  - Background task offloading
  - PostgreSQL + pgvector persistence (document chunks, embeddings)
  - Local FAISS persistence (per-session conversation memory)
```

---

## Table of Contents

- [High-Level Component Map](#high-level-component-map)
- [Data Model](#data-model)
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
  - [13. Storage Design](#13-storage-design)
  - [14. Rate Limiting](#14-rate-limiting)
  - [15. Migration: Local FAISS → PostgreSQL + pgvector](#15-migration-local-faiss--postgresql--pgvector)
- [Scaling Bottlenecks](#scaling-bottlenecks)
- [Future Distributed Architecture](#future-distributed-architecture)
- [Design Philosophy](#design-philosophy)

---

## High-Level Component Map

```
                        ┌─────────────────────┐
                        │    React Frontend    │
                        └──────────┬──────────┘
                                   │ HTTP / SSE
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
           │  LangChain   │ │               │ │              │
           │  RAG Service │ │               │ │              │
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

  ┌────────────┐   ┌──────────────────────┐   ┌────────────────┐
  │   Redis    │   │  PostgreSQL          │   │  FAISS Memory  │
  │ Exact Cache│   │  pgvector            │   │  Store         │
  └────────────┘   │  (chunks, embeddings │   │  (per-session) │
                   │   sessions, docs)    │   └────────────────┘
                   └──────────────────────┘
```

---

## Data Model

```
Session (UUID PK, name, created_at)
    │
    └──< SessionDocument (session_id FK, document_id FK)
              │
              └──> Document (UUID PK, checksum SHA-256, status, filename)
                       │
                       └──< DocumentChunk (UUID PK, document_id FK,
                                           parent_id, child_id, chunk_index,
                                           text, embedding vector)

ConversationMemory (session_id, question, answer, timestamp, embedding vector)
ConversationSummary (session_id, summary_text, updated_at)
```

**Ownership rules:**

- A `Document` is created once per unique SHA-256 checksum. Re-uploading an identical file returns the existing `Document` — chunks and embeddings are reused.
- `SessionDocument` is the many-to-many join between sessions and documents. Deleting a session unlinks its `SessionDocument` rows; if a `Document` has no remaining `SessionDocument` references, it is eligible for orphan cleanup.
- `DocumentChunk` has both a `parent_id` (points to a larger context chunk) and a `child_id` (points to the precise indexed unit). Retrieval targets child chunks; prompt assembly expands to parent chunks.

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
    ├── initialize Groq client (once at startup, via groq_client.py)
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
Client → POST /rag/upload (multipart PDF + session_id)
    ↓
Compute SHA-256 checksum of file bytes
    ├─ Existing Document found (same checksum)
    │       ↓
    │   Create SessionDocument link only
    │   Return HTTP 200: { "status": "ready", "deduplicated": true }
    │
    └─ No match
            ↓
        Save file to: uploads/<filename>.pdf
        Create Document row (status = "processing")
        Create SessionDocument link
        Return HTTP 200: { "status": "processing" }
            ↓  (BackgroundTask begins)
        PyMuPDF (pdf_service.py): extract full text
            ↓
        Parent-child chunker:
            - Split into parent chunks (larger context windows)
            - Split each parent into child chunks (embedding units)
            - child.parent_id → parent chunk UUID
            ↓
        ai_service.py: embed all child chunks
            - Model: BAAI/bge-base-en-v1.5
            - Output: float32 vectors stored in pgvector column
            ↓
        Persist all DocumentChunk rows to PostgreSQL
            ↓
        Update Document.status = "ready"
            ↓
        Document available in GET /rag/sessions/{session_id}/documents
```

#### Deduplication

SHA-256 is computed before any processing. An identical file uploaded to a second session costs only a single DB insert (`SessionDocument`) — no re-embedding, no new chunks.

#### Completion Detection

`GET /rag/sessions/{session_id}/documents` queries `Document` rows via the `SessionDocument` join where `status = ready`. Partially indexed documents are not returned.

---

### 3. Retrieval Pipeline

**Trigger:** `POST /rag/ask`

#### Full Pipeline

```
POST /rag/ask { session_id, document_ids, question, stream }
    ↓
─────────────── Cache Layer ───────────────
    ↓
Check Redis exact cache: md5(document_ids + question)
    ├─ HIT  → return cached answer immediately
    └─ MISS ↓
Check semantic cache: embed question → pgvector cosine similarity search
    ├─ HIT  → return cached answer
    └─ MISS ↓
─────────────── Query Expansion ───────────────
    ↓
LLM: generate N query variants from original question
    (preserving intent; reducing topic drift)
    ↓
─────────────── Parallel Retrieval ────────────────
    ↓
For each expanded query (asyncio.gather across all variants):
    ├── Embed query variant (ai_service.py)
    └── Hybrid retrieval (vector_search_service.py):
            ├── pgvector ANN search (dense semantic similarity)
            └── PostgreSQL full-text search (BM25 keyword matching)
    ↓
─────────────── RRF Reranking ──────────────────
    ↓
Merge all per-query result sets via Reciprocal Rank Fusion
    ↓
Deduplicate chunks (by chunk UUID)
    ↓
─────────────── Parent Expansion ───────────────
    ↓
For each retrieved child chunk:
    └── Resolve parent chunk → fetch full parent text
    ↓
Build structured LLM context blocks
    ↓
─────────────── Memory Retrieval ────────────────
    ↓
asyncio.gather():
    ├── retrieve_memories(session_id, question)
    └── load_summary(session_id)
    ↓
─────────────── Prompt Construction ───────────────
    ↓
Assemble prompt:
    1. System instructions
    2. Rolling memory summary (if exists)
    3. Relevant retrieved memories
    4. Parent-expanded document blocks
    5. Current question
    ↓
─────────────── LLM Call + Streaming ──────────────
    ↓
llm_service.generate_response(messages, stream=True)
    ↓
SSE stream:
    ├── event: thinking  { "step": "..." }   ← reasoning trace events
    └── event: token     { "token": "..." }  ← answer tokens
    ↓
─────────────── Post-Processing ───────────
    ↓
BackgroundTasks:
    ├── write exact cache (Redis)
    ├── write semantic cache (pgvector)
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
  "answer": "Redis and pgvector are used in this project.",
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
Write summary to: memory_store/<session_id>.txt
    ↓
Rebuild FAISS index from recent_entries only
    ↓
Persist updated index and JSON
```

#### Summary Injection in Prompts

```
Prompt assembly:

[System Instructions]
[Rolling Summary]          ← loaded from <session_id>.txt
[Retrieved Memories]       ← top-K from FAISS semantic search
[Document Chunks]          ← parent-expanded blocks from pgvector retrieval
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
key = md5(document_ids + question)
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
pgvector cosine similarity search over cached question vectors
    ↓
Cosine similarity ≥ threshold?
    ├─ YES → return stored answer
    └─ NO  → LLM call
                    ↓
            BackgroundTask:
                - Insert question embedding into semantic cache table
                - Store answer in semantic cache metadata
```

**Threshold tuning:**
- Higher threshold = fewer false cache hits, more LLM calls
- Lower threshold = more aggressive caching, risk of slightly mismatched answers

#### Combined Cache Hit Rate

```
Request
    ├─ Exact cache hit    → ~0ms
    ├─ Semantic cache hit → ~5–15ms (embedding + pgvector search)
    └─ LLM call           → ~800–2000ms (Groq API)
```

---

### 7. Vector Persistence Lifecycle

#### Document Vectors (PostgreSQL + pgvector)

```
Ingestion complete
    ↓
DocumentChunk rows (in memory)
    ↓
Bulk insert into PostgreSQL:
    document_chunk table
        - text, chunk_index, parent_id, child_id
        - embedding: vector column (pgvector)
        - document_id FK
    ↓
Document.status updated to "ready"
```

#### Memory Vectors (FAISS — local filesystem)

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

Document chunks are retrieved directly from PostgreSQL via SQL queries — no file I/O. Memory indexes are loaded from disk via `asyncio.to_thread()` to avoid blocking the event loop.

---

### 8. Session Management System

**State per session:**

```
PostgreSQL:
    sessions table          UUID PK, name, created_at
    session_documents       session_id FK, document_id FK

memory_store/ (filesystem):
    <session_id>.index      FAISS IndexFlatIP
    <session_id>.json       Turn metadata array
    <session_id>.txt        LLM rolling summary (optional)
```

#### Endpoint Behavior

| Endpoint | Operation |
|---|---|
| `POST /rag/sessions` | Insert session row; return UUID + name |
| `GET /rag/sessions` | Query sessions table; return all rows |
| `GET /rag/session/{id}` | Load `.json`, return paginated slice |
| `DELETE /rag/session/{id}` | Delete session row (cascade to SessionDocument); delete FAISS files; run orphan document cleanup |
| `POST /rag/session/search` | Embed query, search `.index`, return matched turns |
| `GET /rag/sessions/{id}/documents` | Join Session → SessionDocument → Document; filter status = ready |
| `DELETE /rag/sessions/{id}/documents/{doc_id}` | Remove SessionDocument row; run orphan cleanup |

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
# Phase 1: expand query, then run retrieval across all variants in parallel
expanded_queries = await expand_query(question)

retrieval_results = await asyncio.gather(*[
    hybrid_retrieve(q, document_ids) for q in expanded_queries
])

# Phase 2: memory retrieval in parallel with parent expansion
memories, summary = await asyncio.gather(
    retrieve_memories(session_id, question),
    load_summary(session_id)
)
```

| Task | Execution Model | Reason |
|---|---|---|
| `hybrid_retrieve` | `async` coroutine | PostgreSQL I/O (pgvector + FTS) |
| `expand_query` | `async` coroutine | LLM call via Groq |
| `retrieve_memories` | `async` coroutine | Embedding + FAISS search |
| `load_summary` | `async` coroutine | File read |

All phases run concurrently within their gather groups. Total retrieval latency is bounded by the slowest task in each group, not their sum.

#### Document Indexing

Ingestion runs as a FastAPI `BackgroundTask` — the HTTP response is returned before indexing begins. Embedding and PostgreSQL writes are handled off the critical path.

---

### 11. Background Task Model

FastAPI `BackgroundTasks` are used for all non-critical writes. These execute after the HTTP response is sent to the client.

```
HTTP Response returned to client
    ↓  (simultaneously, in background)
    ├── Redis exact cache write
    ├── Semantic cache pgvector write
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
│     PARENT-EXPANDED DOC BLOCKS       │  ← RRF-ranked child chunks, expanded to parent context
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
  resume. Key topics covered: Redis, pgvector, FastAPI, Groq.

Retrieved Memories:
  Q: What databases does he know?
  A: The document mentions Redis and pgvector.

Document Blocks:
  [Page 3 — Parent Context]
  "...experience with Redis for caching and pgvector for semantic search,
   previously worked with FAISS for local vector search before migrating
   to PostgreSQL..."

Current Question:
  Does he have experience with async Python?
```

The rolling summary prevents the prompt from needing full conversation history while preserving key accumulated context. Parent-expanded blocks give the LLM broader context than child chunks alone.

---

### 13. Storage Design

#### Document Chunks — PostgreSQL + pgvector

| Property | Value |
|---|---|
| Storage | PostgreSQL table `document_chunk` |
| Vector column | `pgvector` extension — native ANN search |
| Similarity | Cosine (via `<=>` operator) |
| Text search | PostgreSQL `tsvector` / `tsquery` (BM25) |
| Filtering | `document_id` FK — efficient per-document scoping |
| Deduplication | `Document.checksum` (SHA-256) |

**Why PostgreSQL for documents:** Document vectors must be shared across multiple FastAPI workers, filterable by `document_id` and `session_id`, and durable across restarts with ACID guarantees. pgvector provides all of this inside a single infrastructure component that also handles BM25 full-text search, eliminating the separate BM25 pickle files used in the pre-migration architecture.

#### Memory Index — FAISS `IndexFlatIP` (per-session filesystem)

| Property | Value |
|---|---|
| Index type | Flat (brute-force) |
| Search type | Exact nearest neighbor |
| Similarity | Cosine (via L2 normalization + inner product) |
| Use case | Per-session memory (small, bounded by SUMMARY_TRIGGER) |

**Why FAISS for memory:** Session memory indexes are small and bounded by `SUMMARY_TRIGGER`. The session-local nature means multi-worker sharing is not required. Exact search ensures no relevant memory is missed. The cost of brute-force search is negligible at this scale.

#### Normalization (FAISS memory indexes)

All FAISS vectors are L2-normalized before insertion and search:

```python
faiss.normalize_L2(vectors)
index.add(vectors)     # IndexFlatIP inner product = cosine similarity
```

pgvector's `<=>` operator handles cosine similarity natively without manual normalization.

---

### 14. Rate Limiting

**Library:** SlowAPI (Starlette-compatible wrapper around `limits`)

**Default:** 5 requests/minute per client IP

Applied at the controller level. Configurable per endpoint. Returns HTTP 429 on breach.

---

### 15. Migration: Local FAISS → PostgreSQL + pgvector

The system originally used FAISS `IndexHNSWFlat` for document vectors and a local filesystem for chunk metadata. This was replaced in full. The migration rationale:

| Concern | Local FAISS (original) | PostgreSQL + pgvector (current) |
|---|---|---|
| Multi-worker sharing | Not possible — in-process index | Native — shared DB connection pool |
| Hybrid retrieval | Separate BM25 pickle files | Single SQL query (`tsvector` + `<=>`) |
| Document deduplication | Not supported | `checksum` column + unique constraint |
| Session-scoped filtering | Filename prefix conventions | `document_id` FK with proper joins |
| Cascade deletes | Manual file cleanup code | `ON DELETE CASCADE` on FK constraints |
| Persistence guarantees | Background file writes (lossy on crash) | ACID transactions |
| Index availability detection | Co-presence of `.index` + `.json` files | `Document.status = ready` column |

**What was preserved:** FAISS `IndexFlatIP` was kept for per-session conversation memory. Memory indexes are session-local, small, bounded, and do not need to be shared across workers — the properties that made FAISS unsuitable for documents do not apply here.

---

## Scaling Bottlenecks

| Bottleneck | Current Behavior | At Scale |
|---|---|---|
| PostgreSQL single instance | Fast; ACID; pgvector ANN search | Needs read replicas or dedicated vector DB under heavy write load |
| Local filesystem FAISS (memory) | Zero-latency reads; session-local | Cannot be shared across multiple worker processes |
| CPU embeddings | Adequate for low-medium traffic | Becomes throughput ceiling under load |
| Redis single instance | Fast for small cache | Needs clustering for high availability |
| BackgroundTasks | Works for single process | Tasks lost on crash; no retry |
| Groq API rate limits | Fine for development | Needs key pooling or fallback under load |

---

## Future Distributed Architecture

```
                     ┌──────────────┐
                     │ Load Balancer │
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
   │ (cache)    │  │  (if needed) │  │  (primary)   │
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
| PostgreSQL + pgvector | Add read replicas; or migrate ANN search to Qdrant if vector load dominates |
| FAISS `IndexFlatIP` (memory) | pgvector table — already available in the same DB |
| Local memory JSON | Postgres `conversation_memory` table |
| FastAPI `BackgroundTasks` | Celery + Redis broker |
| Single Redis | Redis Cluster |
| CPU SentenceTransformers | Hosted embedding API or GPU inference server |

The service layer design means most of these are single-file replacements.

---

## Design Philosophy

| Principle | Implementation |
|---|---|
| Low latency on the hot path | Two cache layers; async parallel retrieval across expanded queries; background writes |
| Modularity | Each service owns one concern; LLM, memory, cache, and vector are fully independent |
| Shared persistence | PostgreSQL + pgvector replaces in-process FAISS for documents; safe under multiple workers |
| Semantic everywhere | Embeddings used for retrieval, caching, memory, session search, and query expansion |
| Bounded context | Memory summarization + parent-child chunking prevent unbounded prompt growth |
| Provider independence | `llm_service.py` abstracts the LLM provider behind a stable interface |
| Deduplication by default | SHA-256 checksums on upload eliminate redundant embeddings at the storage layer |
| Production patterns | Rate limiting, async indexing, background tasks, SSE streaming, structured DB models |

The system is designed to run entirely on a single machine for development and small-scale production, while following architectural patterns that translate cleanly to a distributed deployment with minimal refactoring.
