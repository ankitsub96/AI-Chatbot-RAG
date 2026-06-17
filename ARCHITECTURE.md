# Architecture Overview

## System Classification

```
Conversational Retrieval-Augmented Generation (RAG) System
with Multi-Strategy Agentic Research Layer

Properties:
  - Centralized LLM abstraction layer
  - Session-scoped document ownership with UUID primary keys
  - SHA-256 checksum deduplication on upload
  - Parent-child chunking for context-aware retrieval
  - Hybrid BM25 + pgvector dense retrieval
  - Multi-query expansion with Reciprocal Rank Fusion (RRF) reranking
  - LLM thinking traces streamed via SSE
  - Two-layer semantic caching (Redis exact + pgvector semantic)
  - PostgreSQL + pgvector long-term conversational memory with rolling
    summarization (migrated from local FAISS — see Subsystem 15)
  - Async document ingestion pipeline
  - Parallel async retrieval across expanded queries
  - Background task offloading
  - PostgreSQL + pgvector persistence (document chunks, embeddings, memory)
  - Four interchangeable agentic strategies over one shared tool layer:
    plain RAG, LangChain RAG, single-pass agentic RAG, and three
    research-agent variants (ReAct, Planner, Hybrid)
  - Forced-parallel tool execution (not LLM-driven) for predictable latency
  - Tool layer shared across all agentic endpoints — zero duplicated
    retrieval/generation/evaluation logic
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
  - [16. Agentic Research Layer](#16-agentic-research-layer)
  - [17. Shared Tool Layer](#17-shared-tool-layer)
  - [18. Shared Graph Nodes](#18-shared-graph-nodes)
  - [19. ReAct Agent](#19-react-agent)
  - [20. Planner Agent](#20-planner-agent)
  - [21. Hybrid Research Agent](#21-hybrid-research-agent)
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
        ┌──────────────┬──────────┼──────────┬──────────────┬──────────────┐
        ▼              ▼          ▼          ▼              ▼              ▼
 ┌──────────────┐┌──────────────┐┌────────┐┌──────────────┐┌──────────────┐┌──────────────┐
 │  RAG Service ││  LangChain   ││Agentic ││ React Agent  ││Planner Agent ││Hybrid Agent  │
 │ (rag_service)││  RAG Service ││  RAG   ││  Service     ││  Service     ││  Service     │
 └──────┬───────┘└──────┬───────┘└───┬────┘└──────┬───────┘└──────┬───────┘└──────┬───────┘
        │                │            │            │               │               │
        │                │            └────────────┴───────────────┴───────┬───────┘
        │                │                                                  │
        │                │                                    ┌─────────────▼─────────────┐
        │                │                                    │   shared_rag_nodes.py      │
        │                │                                    │  (cache/memory/react loop/  │
        │                │                                    │   plan/synthesize/evaluate) │
        │                │                                    └─────────────┬─────────────┘
        │                │                                                  │
        │                │                                    ┌─────────────▼─────────────┐
        │                │                                    │   research_tools.py        │
        │                │                                    │  (9 pure tool functions:    │
        │                │                                    │   document_search,          │
        │                │                                    │   page_lookup, web_search,  │
        │                │                                    │   query_expander,           │
        │                │                                    │   question_decomposer,      │
        │                │                                    │   answer_generator,         │
        │                │                                    │   answer_synthesizer,       │
        │                │                                    │   answer_evaluator,         │
        │                │                                    │   memory_search)            │
        │                │                                    └─────────────┬─────────────┘
        │                │                                                  │
        └────────────────┴──────────────────┬───────────────────────────────┘
                                              ▼
                                   ┌─────────────────────┐
                                   │ vector_search_service│
                                   │   memory_service      │
                                   │  semantic_cache_service│
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │     LLM Service      │
                                   │   (llm_service.py)   │
                                   └──────────┬───────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │      Groq API        │
                                   └─────────────────────┘

Supporting infrastructure:

  ┌────────────┐   ┌──────────────────────┐   ┌──────────────────┐
  │   Redis    │   │  PostgreSQL          │   │  DuckDuckGo       │
  │ Exact Cache│   │  pgvector            │   │  (web_search tool,│
  └────────────┘   │  (chunks, embeddings,│   │   no API key)     │
                    │   memory, sessions,  │   └──────────────────┘
                    │   docs)              │
                    └──────────────────────┘
```

The five orchestration services on the top row (`rag_service`, `langchain_rag_service`,
`agentic_rag_service`, plus the three new research services) are **untouched
relative to each other** — `react_rag_service.py`, `planner_rag_service.py`, and
`hybrid_rag_service.py` are additive. None of the pre-existing five files
(`rag_service.py`, `langchain_rag_service.py`, `agentic_rag_service.py`,
`vector_search_service.py`, `memory_service.py`, `semantic_cache_service.py`) were
modified to support the new agents; the new agents call the same underlying
service functions through a shared tool layer instead.

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

ConversationMemory (session_id, question, answer, text, embedding vector,
                     thoughts JSON, created_at)
ConversationSummary (session_id, summary, embedding vector, created_at)
```

**Ownership rules:**

- A `Document` is created once per unique SHA-256 checksum. Re-uploading an identical file returns the existing `Document` — chunks and embeddings are reused.
- `SessionDocument` is the many-to-many join between sessions and documents. Deleting a session unlinks its `SessionDocument` rows; if a `Document` has no remaining `SessionDocument` references, it is eligible for orphan cleanup.
- `DocumentChunk` has both a `parent_id` (points to a larger context chunk) and a `child_id` (points to the precise indexed unit). Retrieval targets child chunks; prompt assembly expands to parent chunks.
- `ConversationMemory.thoughts` stores the full reasoning trace (`trace` list) for any agentic endpoint that produced the turn — `/rag/ask/agent`, `/rag/react/ask`, `/rag/planner/ask`, `/rag/research/ask`. Turns from `/rag/ask` and `/rag/ask/langchain` persist with `thoughts = null`, since those pipelines do not emit a structured trace.

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
    ├── generate_response(messages, temperature, tools, tool_choice, max_tokens)
    └── expand_query_sync(question, memory_context, n)
    │
    ▼
Groq API
```

#### Interface

| Method | Purpose |
|---|---|
| `generate_response(messages, ...)` | Standard chat completion; returns the raw `ChatCompletion` object, used by every orchestration and research service |
| `expand_query_sync(question, memory_context, n)` | Generates `n` reworded query variants preserving original intent, used by query expansion and the ReAct/research agents' `query_expander` tool |

#### Design Rationale

Previously, Groq client initialization was duplicated in `extract_service.py`, `rag_service.py`, and `memory_service.py`. Centralizing this:

- Eliminates N-client instantiation on startup
- Makes the provider fully swappable by editing one file
- Centralizes retry policy, timeout config, and observability hooks
- Allows the model to be reconfigured at one point without touching business logic
- Every new research agent (Subsystems 19–21) reuses this same single entry point — no agent owns its own LLM client

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
    │   Return HTTP 200: { "status": "ready", "reused": true }
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

This pipeline is the foundation that every other agentic endpoint (Subsystems
16–21) builds on top of. `tool_document_search` in `research_tools.py`
(Subsystem 17) is a direct repackaging of the hybrid retrieval + parent expansion
steps below into a single callable unit.

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
For each expanded query (ThreadPoolExecutor across all variants):
    ├── Embed query variant (vector_search_service.create_embedding)
    └── Hybrid retrieval (vector_search_service.hybrid_search):
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
retrieve_relevant_memories(session_id, question)
    — runs summary + recent-memory pgvector lookups in parallel internally
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
    └── persist memory turn (PostgreSQL — see Subsystem 4)
    ↓
Return answer to client
```

---

### 4. Conversational Memory Lifecycle

**Scope:** Per `session_id`

**Storage:** PostgreSQL — `conversation_memories` and `conversation_summaries` tables (migrated from local FAISS + JSON files; see Subsystem 15 for migration rationale)

#### Data Model (per turn — `ConversationMemory` row)

```json
{
  "session_id": "a1b2c3d4-...",
  "question": "What databases does he know?",
  "answer": "Redis and pgvector are used in this project.",
  "text": "USER:\n...\n\nASSISTANT:\n...",
  "embedding": [0.0123, -0.045, ...],
  "thoughts": [ { "node": "...", "message": "...", "data": {...}, "ts": 1234.5 } ],
  "created_at": "2026-06-17T10:30:00Z"
}
```

`thoughts` is populated for any turn produced by an agentic endpoint
(`/rag/ask/agent`, `/rag/react/ask`, `/rag/planner/ask`, `/rag/research/ask`);
it is `null` for `/rag/ask` and `/rag/ask/langchain`.

#### Memory Write

```
Question + Answer (+ optional thoughts trace)
    ↓
Concatenate: "USER:\n{question}\n\nASSISTANT:\n{answer}"
    ↓
Embed with BAAI/bge-base-en-v1.5 (asyncio.to_thread, off the event loop)
    ↓
INSERT INTO conversation_memories (..., embedding, thoughts)
    ↓
If len(session memories) > SUMMARY_TRIGGER → trigger summarization (Subsystem 5)
```

#### Memory Read

```
Incoming question
    ↓
Embed question
    ↓
Two parallel pgvector queries (ThreadPoolExecutor):
    ├── conversation_summaries  ORDER BY embedding <=> :embedding LIMIT SUMMARY_TOP_K
    └── conversation_memories   ORDER BY embedding <=> :embedding LIMIT MEMORY_TOP_K
    ↓
Concatenate matched summaries + memories into a single context string
    ↓
Return context string (consumed directly by prompt assembly, or via the
`tool_memory_search` tool — see Subsystem 17 — for the new research agents)
```

#### Why pgvector for Memory (current)

Memory now lives in the same PostgreSQL instance as document chunks, giving it
the same multi-worker safety, ACID durability, and `<=>` cosine-distance
operator used everywhere else in the system — eliminating the separate
per-session FAISS index files and JSON metadata the system previously relied on.

---

### 5. Memory Summarization Lifecycle

**Trigger:** `len(session_memories) > SUMMARY_TRIGGER`

**Execution:** Runs inline inside `save_conversation_turn` (awaited via
`asyncio.to_thread` for the LLM call and embedding step), immediately after the
new turn is inserted — not a separate background task.

#### Summarization Flow

```
New memory row inserted
    ↓
Load all ConversationMemory rows for the session, ordered by created_at
    ↓
len(memories) > SUMMARY_TRIGGER ?
    ↓ yes
older_memories = memories[:-RECENT_HISTORY]
    ↓
summarize_conversation(older_memories)  → LLM call via llm_service
    ↓
Embed the new summary text
    ↓
INSERT INTO conversation_summaries (session_id, summary, embedding)
```

#### Summary Injection in Prompts

```
Prompt assembly:

[System Instructions]
[Rolling Summary]          ← top-K from conversation_summaries (pgvector)
[Retrieved Memories]       ← top-K from conversation_memories (pgvector)
[Document Chunks]          ← parent-expanded blocks from pgvector retrieval
[Current Question]
```

#### Bounds Analysis

| Config | Effect |
|---|---|
| `SUMMARY_TRIGGER` | Maximum number of full turns before compression |
| `RECENT_HISTORY` | Number of recent turns always kept verbatim |
| `SUMMARY_TOP_K` / `MEMORY_TOP_K` | Number of summary/memory rows retrieved per question, bounding prompt size regardless of total table size |

The combination means prompt memory usage grows as `O(MEMORY_TOP_K + SUMMARY_TOP_K)` retrieved rows per question, regardless of total conversation length — old `ConversationMemory` rows are currently retained rather than deleted after summarization (see commented-out deletion code in `memory_service.py`), so summarization bounds prompt size but not table size.

---

### 6. Cache Lifecycle

#### Layer 1 — Redis Exact Cache

```
Request arrives
    ↓
key = session_id + "|" + sorted(document_ids) joined
normalized_question = normalize_question(question)
cache_key = exact_cache_key(key, normalized_question)
    ↓
Redis GET cache_key
    ├─ HIT  → return value, skip all further processing
    └─ MISS → continue pipeline
                    ↓
            (after LLM response)
                    ↓
            Redis SETEX cache_key TTL value
```

**Characteristics:**
- O(1) lookup
- TTL configurable per deployment (`CACHE_TTL`)

#### Layer 2 — Semantic Cache

```
Question
    ↓
Embed with BAAI/bge-base-en-v1.5
    ↓
Scan Redis keys matching `semantic_cache::{key}::*`
    ↓
Compute cosine similarity between query embedding and each cached embedding
    ↓
Best similarity ≥ SEMANTIC_CACHE_THRESHOLD?
    ├─ YES → return stored answer
    └─ NO  → LLM call
                    ↓
            Redis SETEX semantic_cache_key TTL { question, embedding, answer }
```

**Threshold tuning:**
- Higher threshold = fewer false cache hits, more LLM calls
- Lower threshold = more aggressive caching, risk of slightly mismatched answers

This same two-layer check runs at the start of every agentic endpoint
(`check_cache_agent` in `shared_rag_nodes.py` — Subsystem 18), not just the
plain `/rag/ask` pipeline.

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

#### Memory Vectors (PostgreSQL + pgvector)

```
Conversation turn complete
    ↓
INSERT INTO conversation_memories (..., embedding)
    ↓
If summarization triggers:
INSERT INTO conversation_summaries (..., embedding)
```

Both document and memory vectors now live in the same PostgreSQL instance,
queried with the same `<=>` cosine-distance operator — no separate file-based
persistence step is required for either.

#### Load on Request

All retrieval — documents, memory, and summaries — is read directly from
PostgreSQL via SQL queries. There is no file I/O on the request path for either
subsystem.

---

### 8. Session Management System

**State per session:**

```
PostgreSQL:
    sessions table              UUID PK, name, created_at
    session_documents           session_id FK, document_id FK
    conversation_memories       session_id FK, question, answer, text,
                                 embedding, thoughts, created_at
    conversation_summaries      session_id FK, summary, embedding, created_at
```

#### Endpoint Behavior

| Endpoint | Operation |
|---|---|
| `POST /rag/sessions` | Insert session row; return UUID + name |
| `GET /rag/sessions` | Query sessions table; return all rows |
| `GET /rag/sessions/{id}` | Paginated conversation history, including persisted `thoughts` per turn where available |
| `DELETE /rag/sessions/{id}` | Delete session row (cascade to SessionDocument); delete associated memory rows; run orphan document cleanup |
| `POST /rag/sessions/{id}/search` | Embed query, pgvector search over the session's memory, return matched turns |
| `GET /rag/sessions/{id}/documents` | Join Session → SessionDocument → Document; filter status = ready |
| `DELETE /rag/sessions/{id}/documents/{doc_id}` | Remove SessionDocument row; run orphan cleanup |

---

### 9. Semantic Session Search

**Purpose:** Enable meaning-based retrieval over a session's conversation history.

#### Flow

```
POST /rag/sessions/{session_id}/search
    { query }
    ↓
Embed query with BAAI/bge-base-en-v1.5
    ↓
pgvector cosine similarity search over conversation_memories
    WHERE session_id = :session_id
    ↓
Return top-K matching { question, answer } pairs
```

This uses the same `conversation_memories` table maintained for prompt memory
injection — no additional index is needed.

---

### 10. Async Execution Model

All I/O-bound and CPU-bound operations are handled to avoid blocking the FastAPI event loop, using a mix of `asyncio.gather`, `asyncio.to_thread`, and `ThreadPoolExecutor` depending on the call site.

#### Parallel Retrieval (per RAG request)

```python
# Hybrid retrieval across expanded queries — ThreadPoolExecutor, not asyncio.gather,
# since hybrid_search itself is a sync function calling sync DB sessions
with ThreadPoolExecutor(max_workers=min(len(combos), 8)) as executor:
    futures = {executor.submit(search_one, combo): combo for combo in combos}
    for future in as_completed(futures):
        all_result_sets.append(future.result())
```

| Task | Execution Model | Reason |
|---|---|---|
| `hybrid_search` (per expanded query) | `ThreadPoolExecutor` | Sync PostgreSQL I/O (pgvector + FTS) |
| `vector_search` + `keyword_search` (within one hybrid call) | `ThreadPoolExecutor` (2 workers) | Run dense and BM25 search concurrently |
| `expand_query_sync` | sync call inside async node | LLM call via Groq, wrapped by the calling node where needed |
| `retrieve_relevant_memories` | `ThreadPoolExecutor` (2 workers) internally | Summary + recent-memory pgvector queries run concurrently |
| `save_conversation_turn` | `async def` + `asyncio.to_thread` for embedding/summarization | Keeps the event loop free during CPU-bound embedding |
| `run_tools_parallel` (research agents) | `ThreadPoolExecutor`, `max_workers=min(len(tasks), 8)` | Generic parallel tool dispatcher shared by all 3 research agents — see Subsystem 17 |

All phases run concurrently within their respective executor pools. Total
retrieval latency is bounded by the slowest task in each group, not their sum.

#### Document Indexing

Ingestion runs as a FastAPI `BackgroundTask` — the HTTP response is returned before indexing begins. Embedding and PostgreSQL writes are handled off the critical path.

---

### 11. Background Task Model

FastAPI `BackgroundTasks` are used for non-critical writes on the plain `/rag/ask` pipeline. These execute after the HTTP response is sent to the client.

```
HTTP Response returned to client
    ↓  (simultaneously, in background)
    ├── Redis exact cache write
    ├── Semantic cache pgvector write
    └── Memory persistence (conversation_memories insert)
            └── Memory summarization (if triggered)
```

**Note on agentic endpoints:** `/rag/ask/agent`, `/rag/react/ask`,
`/rag/planner/ask`, and `/rag/research/ask` perform cache and memory writes
synchronously inside their final graph node (`save_turn_agent` in
`shared_rag_nodes.py`) rather than via `BackgroundTasks`, since these run
inside a LangGraph node invoked through `asyncio.to_thread` already, after the
streaming response has been fully consumed by the client.

**Latency impact:** Zero on the plain `/rag/ask` path — all cache and persistence writes are invisible to the request-response cycle.

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

For the three research agents (Subsystems 19–21), the same five-layer shape
applies per sub-question rather than once per request — `tool_answer_generator`
(Subsystem 17) builds this prompt for each plan step or ReAct iteration, and
`tool_answer_synthesizer` builds a sixth, simpler layer on top: original
question + all sub-answers, with no document context, to produce the final
combined answer.

#### Strictness Levels

The three research agents additionally accept a `strictness` parameter
(`strict` | `balanced` | `creative`) that changes the system instruction
injected into `tool_answer_generator` and `tool_answer_synthesizer`:

| Level | Instruction |
|---|---|
| `strict` | Use ONLY document context; do not infer beyond what is retrieved |
| `balanced` | Prefer document context; may connect ideas using general knowledge |
| `creative` | Use document context and general knowledge freely |

`strictness=creative` combined with `use_web=False` is an intentional
combination — it permits LLM inference beyond the retrieved documents without
calling out to the web.

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
  "...experience with Redis for caching and pgvector for semantic search..."

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

**Why PostgreSQL for documents:** Document vectors must be shared across multiple FastAPI workers, filterable by `document_id` and `session_id`, and durable across restarts with ACID guarantees. pgvector provides all of this inside a single infrastructure component that also handles BM25 full-text search.

#### Conversation Memory — PostgreSQL + pgvector

| Property | Value |
|---|---|
| Storage | PostgreSQL tables `conversation_memories`, `conversation_summaries` |
| Vector column | `pgvector` extension — native ANN search |
| Similarity | Cosine (via `<=>` operator) |
| Filtering | `session_id` — efficient per-session scoping |
| Trace storage | `conversation_memories.thoughts` (JSON), populated only for agentic endpoints |

**Why PostgreSQL for memory (current):** Memory was migrated off local FAISS +
JSON files for the same reasons documents were — multi-worker safety, ACID
durability, and a single query surface shared with document retrieval. See
Subsystem 15 for the full before/after comparison. FAISS may still be present
in the codebase as a legacy dependency, but it is no longer in the active
memory read/write path.

---

### 14. Rate Limiting

**Library:** SlowAPI (Starlette-compatible wrapper around `limits`)

**Default:** 5 requests/minute per client IP

Applied at the controller level. Configurable per endpoint. Returns HTTP 429 on breach.

---

### 15. Migration: Local FAISS → PostgreSQL + pgvector

The system originally used FAISS for both document vectors (`IndexHNSWFlat`)
and per-session conversation memory (`IndexFlatIP`), backed by a local
filesystem for chunk and turn metadata. **Both have since been migrated to
PostgreSQL + pgvector.** Documents migrated first; conversation memory and
summarization followed, moving from `memory_store/<session_id>.index` +
`.json` + `.txt` files into the `conversation_memories` and
`conversation_summaries` tables described in Subsystem 4.

| Concern | Local FAISS (original) | PostgreSQL + pgvector (current) |
|---|---|---|
| Multi-worker sharing | Not possible — in-process index | Native — shared DB connection pool |
| Hybrid retrieval | Separate BM25 pickle files | Single SQL query (`tsvector` + `<=>`) |
| Document deduplication | Not supported | `checksum` column + unique constraint |
| Session-scoped filtering | Filename prefix conventions | `document_id` / `session_id` FK with proper joins |
| Cascade deletes | Manual file cleanup code | `ON DELETE CASCADE` on FK constraints |
| Persistence guarantees | Background file writes (lossy on crash) | ACID transactions |
| Index availability detection | Co-presence of `.index` + `.json` files | `Document.status = ready` column |
| Memory reasoning trace | Not stored | `conversation_memories.thoughts` JSON column |

**Current state:** No part of the active retrieval or memory path depends on
FAISS or local index files. FAISS-related dependencies and directories
(`vector_store/indexes`, `vector_store/bm25`, `memory_store/`) may still exist
in the repository as legacy artifacts; they are not written to or read from by
any endpoint described in this document.

---

### 16. Agentic Research Layer

**New endpoints:**

| Endpoint | Strategy |
|---|---|
| `POST /rag/react/ask` | ReAct — LLM reasons and picks one tool per iteration, in a loop, up to `REACT_MAX_ITERATIONS` |
| `POST /rag/planner/ask` | Planner — decomposes the question into sub-questions upfront, executes all of them in parallel, synthesizes |
| `POST /rag/research/ask` | Hybrid — plans upfront like Planner, then runs a mini ReAct loop per sub-question in parallel (LangGraph subgraphs) |

All three sit alongside the pre-existing `POST /rag/ask/agent` (single-pass
agentic RAG with retry-on-low-confidence) without modifying it. The four
agentic endpoints share one tool layer (`research_tools.py`, Subsystem 17) and
one node layer (`shared_rag_nodes.py`, Subsystem 18) — no retrieval, generation,
caching, memory, or evaluation logic is duplicated between them. Differences
between the three new agents are expressed entirely in graph wiring
(`react_rag_service.py`, `planner_rag_service.py`, `hybrid_rag_service.py`),
each of which contains only state type definitions, `StateGraph` construction,
and the public `run_*_agent` entry point.

#### Common Request Shape

```json
{
  "session_id": "a1b2c3d4-...",
  "question": "How do Redis caching and pgvector retrieval interact in this system?",
  "document_ids": null,
  "stream": true,
  "use_web": false,
  "strictness": "balanced"
}
```

`document_ids: null` resolves to all documents linked to the session via
`SessionDocument`, identical to the plain `/rag/ask` behavior.

#### Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Tool layer | Single `research_tools.py` shared by all agentic services | No duplication; every agent imports the same functions |
| Parallel execution | Forced parallel (not LLM-driven) | Reliable, faster, easier to reason about and debug than letting the LLM decide concurrency |
| Tool result shape | Summary + top-3 chunks + metadata + full context, separately | Balances what the LLM sees for reasoning (summary) against what answer generation needs (full context) |
| Memory retrieval | Upfront fixed node, not an LLM-selectable tool | Consistent with `/rag/ask/agent`; faster, no wasted iteration deciding whether to check memory |
| Web search | DuckDuckGo via `duckduckgo-search`, only when `use_web=true` | No API key required; sufficient snippet quality for this use case |
| Web fetch | Not implemented | Snippets from search results are sufficient; avoids added complexity and failure surface |
| Tool history in state | Names + args kept; result text truncated to 500 characters | Prevents unbounded context window growth across ReAct iterations |
| `REACT_MAX_ITERATIONS` | 5 | Hard cap on the ReAct loop, applies per-agent-run for `/rag/react/ask` and per-step for `/rag/research/ask` subgraphs |
| `PLANNER_MAX_STEPS` | 5 | Hard cap on planner-generated sub-questions |
| `WEB_SEARCH_MAX_RESULTS` | 5 | DuckDuckGo results requested per query |

---

### 17. Shared Tool Layer

**File:** `app/services/research_tools.py`

**Purpose:** Pure functions, no LangGraph state, no side effects beyond their
own DB/LLM/web calls. Every tool returns one of nine typed `ToolResult` shapes
(`app/models/chunk_types.py`) with a consistent structure:

```python
class ToolResult(TypedDict):
    name: str
    success: bool
    summary: str                 # short description for LLM reasoning
    top_chunks: list[ChunkDict]   # top 3 chunks, full text
    full_context: str             # full context for answer generation
    data: dict                    # tool-specific metadata
    truncated_result: str         # <=500 chars, safe to store in tool history
    ts: float
```

#### The Nine Tools

| Tool | Reuses | Purpose |
|---|---|---|
| `tool_document_search` | `create_embedding`, `hybrid_search`, `group_chunks_by_parent`, `select_top_parents`, `expand_parent_chunks`, `build_parent_context_blocks` | Combines hybrid retrieval + parent expansion into one callable — the same logic as `/rag/ask`'s retrieval pipeline, repackaged |
| `tool_page_lookup` | Direct `DocumentChunk` query | Fetch all chunks on a specific page, for targeted lookups |
| `tool_web_search` | DuckDuckGo (`DDGS`) | Web snippets when `use_web=true`; fails gracefully (returns `success=false`) on rate limits or network errors rather than crashing the agent |
| `tool_query_expander` | `expand_query_sync` | Reuses the same query-rewriting LLM call used by `/rag/ask` |
| `tool_question_decomposer` | `generate_response` | Splits a question into independent sub-questions, capped at `PLANNER_MAX_STEPS` |
| `tool_answer_generator` | `generate_response` | Produces one answer from one question + context + memory + strictness level |
| `tool_answer_synthesizer` | `generate_response` | Combines multiple sub-answers into one coherent final answer |
| `tool_answer_evaluator` | `generate_response` | Groundedness/confidence scoring, identical logic to `/rag/ask/agent`'s evaluation step |
| `tool_memory_search` | `retrieve_relevant_memories` | Wraps memory retrieval as a tool-shaped result for consistency with the other 8 |

#### Parallel Dispatcher

```python
def run_tools_parallel(
    tasks: list[tuple[callable, tuple, dict]],
    max_workers: int = 8,
) -> list[AnyToolResult]:
```

A generic `ThreadPoolExecutor`-based dispatcher — takes a list of
`(function, args, kwargs)` tuples, runs them concurrently, and returns results
in the original order regardless of completion order. Any individual tool
failure is caught and converted into a `success=false` result rather than
raising, so one failed web search or document search never aborts the whole
batch. Used by the Planner agent's parallel search/answer-generation steps and
the Hybrid agent's parallel per-step subgraph execution.

---

### 18. Shared Graph Nodes

**File:** `app/services/shared_rag_nodes.py`

**Purpose:** Every LangGraph node used by any agentic endpoint lives here —
including ones used by only a single agent — so that no service file
(`react_rag_service.py`, `planner_rag_service.py`, `hybrid_rag_service.py`)
contains anything beyond state definitions, graph wiring, and the public entry
point. Nodes wrap tools (Subsystem 17) with LangGraph state read/write logic.

```
research_tools.py     — pure functions, no state, callable from anywhere
        │
        ▼
shared_rag_nodes.py   — all nodes; import tools, manage state in/out
        │
        ▼
react_rag_service.py / planner_rag_service.py / hybrid_rag_service.py
        — StateGraph construction + run_*_agent entry point only
```

| Node | Used by | Role |
|---|---|---|
| `check_cache_agent` | all 4 agentic endpoints | Exact + semantic cache check, identical to `/rag/ask`'s cache layer |
| `serve_cache_agent` | all 4 | Returns cached answer, short-circuits the graph to `END` |
| `route_cache_agent` | all 4 | Conditional edge: cache hit → `serve_cache_agent`, miss → `retrieve_memory_agent` |
| `retrieve_memory_agent` | all 4 | Upfront memory retrieval via `tool_memory_search` |
| `save_turn_agent` | all 4 | Cache write + `save_conversation_turn` with the full `trace`, final node before `END` |
| `reason_agent` / `act_agent` / `respond_agent` / `route_loop_agent` | `/rag/react/ask` | The agent-level ReAct loop: decide next tool, execute it, decide whether to answer or continue |
| `reason_step_agent` / `act_step_agent` / `respond_step_agent` / `route_step_agent` | `/rag/research/ask` subgraph | The same ReAct loop pattern, scoped to one plan step instead of the whole question |
| `create_plan_agent` | `/rag/planner/ask`, `/rag/research/ask` | Calls `tool_question_decomposer`, streamed as a `plan_creation` thinking event before execution begins |
| `search_parallel_agent` | `/rag/planner/ask` | Runs `tool_document_search` (and `tool_web_search` if `use_web`) for every plan step via `run_tools_parallel` |
| `answer_parallel_agent` | `/rag/planner/ask` | Runs `tool_answer_generator` for every plan step via `run_tools_parallel` |
| `synthesize_agent` | `/rag/planner/ask`, `/rag/research/ask` | Calls `tool_answer_synthesizer` over all collected sub-answers |
| `evaluate_agent` | all 4 | Calls `tool_answer_evaluator`; identical evaluation prompt and JSON schema as `/rag/ask/agent` |
| `execute_steps_agent` | `/rag/research/ask` | Runs one `step_subgraph` invocation per plan step concurrently (`ThreadPoolExecutor`) |

The `evaluate_agent` node intentionally reuses the exact same evaluation prompt
as `/rag/ask/agent`'s pre-existing evaluation node — there is exactly one
evaluation prompt string in the codebase, used by all four agentic endpoints.

---

### 19. ReAct Agent

**Endpoint:** `POST /rag/react/ask`
**File:** `app/services/react_rag_service.py`
**State:** `ReactState` (`app/models/agent_states.py`)

#### Graph

```
check_cache
    │
    ├── hit  → serve_cache → END
    │
    └── miss
         │
    retrieve_memory
         │
    reason ──────────────────┐
         │                    │
    (tool needed) ────► act ──┘
         │
    (ready to answer)
         │
    respond
         │
    evaluate
         │
    ┌────┴────┐
    │ retry?  │── yes ──► reason (loop, bounded by REACT_MAX_ITERATIONS)
    └────┬────┘
         │ no
    save_turn → END
```

`reason_agent` chooses between `document_search`, `query_expander`, and
(if `use_web`) `web_search` on each iteration, or signals `finish` to move to
`respond_agent`. The loop is hard-capped at `REACT_MAX_ITERATIONS` regardless
of the LLM's own decisions.

---

### 20. Planner Agent

**Endpoint:** `POST /rag/planner/ask`
**File:** `app/services/planner_rag_service.py`
**State:** `PlannerState` (`app/models/agent_states.py`)

#### Graph

```
check_cache
    │
    ├── hit  → serve_cache → END
    │
    └── miss
         │
    retrieve_memory
         │
    create_plan              ← tool_question_decomposer, streamed as a
         │                     thinking event before execution starts
    search_parallel           ← document_search (+ web_search) for every
         │                      step, all in parallel via run_tools_parallel
    answer_parallel           ← answer_generator for every step, all in
         │                      parallel via run_tools_parallel
    synthesize                ← answer_synthesizer combines all sub-answers
         │
    evaluate
         │
    save_turn → END
```

Dependency tracking between plan steps (e.g. "how do X and Y relate?" needing
both X's and Y's answers) is explicitly out of scope for v1 — all steps run
independently in parallel, and the synthesizer has access to every sub-answer
regardless of inter-step dependency, which is sufficient in practice since the
synthesis step sees the full set of sub-answers together.

---

### 21. Hybrid Research Agent

**Endpoint:** `POST /rag/research/ask`
**File:** `app/services/hybrid_rag_service.py`
**State:** `HybridState` + per-step `StepState` (`app/models/agent_states.py`)

#### Graph

```
check_cache
    │
    ├── hit  → serve_cache → END
    │
    └── miss
         │
    retrieve_memory
         │
    create_plan               ← same tool_question_decomposer as Planner
         │
    execute_steps              ← for each plan step, concurrently:
         │                        run a full step_subgraph (reason_step →
         │                        act_step → respond_step, looped, capped
         │                        at REACT_MAX_ITERATIONS per step)
    synthesize                 ← answer_synthesizer combines all sub-answers
         │
    evaluate
         │
    save_turn → END
```

#### Step Subgraph

```python
def build_step_subgraph() -> CompiledGraph:
    step_graph = StateGraph(StepState)
    step_graph.add_node("reason", RunnableLambda(timer(reason_step_agent)))
    step_graph.add_node("act", RunnableLambda(timer(act_step_agent)))
    step_graph.add_node("respond", RunnableLambda(timer(respond_step_agent)))
    step_graph.add_conditional_edges("reason", route_step_agent, {"act": "act", "respond": "respond"})
    step_graph.add_edge("act", "reason")
    step_graph.add_edge("respond", END)
    return step_graph.compile()
```

This is the most structurally complex endpoint — it runs N independent mini
ReAct loops (one per plan step) concurrently. Concurrency is achieved by
invoking each compiled subgraph synchronously inside a `ThreadPoolExecutor`
worker (`_run_step`, called from `execute_steps_agent`), rather than running N
separate asyncio event loops — `step_subgraph.invoke(...)` is a blocking call
per thread, which is the simplest correct way to parallelize multiple
independent LangGraph executions without nested event loop issues.

---

## Scaling Bottlenecks

| Bottleneck | Current Behavior | At Scale |
|---|---|---|
| PostgreSQL single instance | Fast; ACID; pgvector ANN search for documents AND memory | Needs read replicas or dedicated vector DB under heavy write load |
| CPU embeddings | Adequate for low-medium traffic | Becomes throughput ceiling under load |
| Redis single instance | Fast for small cache | Needs clustering for high availability |
| BackgroundTasks | Works for single process (plain `/rag/ask` path only) | Tasks lost on crash; no retry |
| Groq API rate limits | Fine for development | Needs key pooling or fallback under load; especially relevant for the research agents, which can issue 6–12 LLM calls per request (see timing table below) |
| Hybrid/Research agent latency | 12–25s typical for `/rag/research/ask` due to N parallel subgraphs each making multiple sequential LLM calls | Per-step latency is bounded by `REACT_MAX_ITERATIONS`, but total wall-clock time is bounded by the slowest step, not the sum — still a meaningfully higher latency tier than the other endpoints |
| Tool history token growth | Truncated to 500 chars per entry, but a full 5-iteration ReAct run can still accumulate 8–10k tokens of context | Worth monitoring on rate-limited tiers; reducible by lowering `REACT_MAX_ITERATIONS` |

#### Timing Estimates (typical, single request)

| Endpoint | LLM calls | Parallelism | Approx wall-clock time |
|---|---|---|---|
| `/rag/react/ask` | 3–7 | Tool calls run sequentially within the loop | 6–15s |
| `/rag/planner/ask` | 4–8 | Search + answer generation parallel across steps | 8–15s |
| `/rag/research/ask` | 6–12 | Per-step ReAct loops run in parallel | 12–25s |

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
   │ (cache)    │  │  (if needed) │  │  (primary —  │
   │            │  │              │  │  docs + mem) │
   └────────────┘  └──────────────┘  └──────────────┘
                           │
                    ┌──────┴──────┐
                    │   Celery    │
                    │  Workers    │
                    │ (bg tasks,  │
                    │  research   │
                    │  agent runs)│
                    └─────────────┘
```

**Migration path:**

| Current | Distributed Replacement |
|---|---|
| `llm_service.py` → Groq | Add OpenAI / Anthropic fallback; no changes elsewhere — every agentic endpoint already routes through this one function |
| PostgreSQL + pgvector (documents + memory) | Add read replicas; or migrate ANN search to Qdrant if vector load dominates |
| FastAPI `BackgroundTasks` | Celery + Redis broker |
| Single Redis | Redis Cluster |
| CPU SentenceTransformers | Hosted embedding API or GPU inference server |
| Synchronous research agent requests | Offload `/rag/research/ask` (highest latency, 12–25s) to a Celery task + polling/webhook pattern if request-response SSE becomes impractical at scale |

The service layer design means most of these are single-file replacements —
the shared tool and node layers (Subsystems 17–18) mean a future change such as
swapping the LLM provider or the web search backend touches one function, not
four agent implementations.

---

## Design Philosophy

| Principle | Implementation |
|---|---|
| Low latency on the hot path | Two cache layers; parallel retrieval across expanded queries; background writes on the plain path |
| Modularity | Each service owns one concern; LLM, memory, cache, and vector are fully independent |
| Shared persistence | PostgreSQL + pgvector handles both documents and memory; safe under multiple workers |
| Semantic everywhere | Embeddings used for retrieval, caching, memory, session search, and query expansion |
| Bounded context | Memory summarization + parent-child chunking + truncated tool history prevent unbounded prompt growth |
| Provider independence | `llm_service.py` abstracts the LLM provider behind a stable interface, used by every agent including the three research services |
| Deduplication by default | SHA-256 checksums on upload eliminate redundant embeddings at the storage layer |
| Zero duplication across agentic strategies | One tool layer (`research_tools.py`), one node layer (`shared_rag_nodes.py`); the three research services differ only in graph wiring |
| Forced parallelism over LLM-driven concurrency | Parallel tool execution is decided by code, not by the LLM, for predictable latency and easier debugging |
| Production patterns | Rate limiting, async indexing, background tasks, SSE streaming, structured DB models, graceful tool failure handling |

The system is designed to run entirely on a single machine for development and small-scale production, while following architectural patterns that translate cleanly to a distributed deployment with minimal refactoring.
