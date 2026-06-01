# RAG Refactor Plan v2

## Session-Based Documents + Checksum Deduplication + Parent/Child Retrieval

---

# Goal

Refactor the current architecture from:

```text
Session
    ↓
Filenames
    ↓
Chunks
```

to:

```text
Session
    ↓
SessionDocument
    ↓
Document
    ↓
DocumentChunk
```

while adding:

* checksum-based document deduplication
* embedding reuse
* parent/child chunking
* session-scoped memory
* parent-aware retrieval

---

# Design Principles

## 1. Documents are global

A document should exist only once in the system.

Same file:

```text
resume.pdf
resume.pdf
resume.pdf
```

uploaded 100 times should produce:

```text
1 Document
1 set of chunks
1 set of embeddings
```

---

## 2. Sessions own documents through mapping

A session can contain:

```text
Resume.pdf
OfferLetter.pdf
Contract.pdf
```

A document can belong to multiple sessions.

Relationship:

```text
Session
    ↔
SessionDocument
    ↔
Document
```

---

## 3. Chunks belong to documents only

Chunks must never contain:

```python
session_id
```

Chunks belong only to:

```python
document_id
```

---

## 4. Memory remains session-scoped

Conversation memory remains attached to:

```python
session_id
```

and NOT:

```python
document_id
```

---

# Phase 1

# Database Model Refactor

Goal:

Create normalized ownership structure.

---

## Step 1.1

Create Session model

New table:

```python
Session
```

Fields:

```python
id
title
created_at
updated_at
```

Notes:

* UUID preferred
* title optional
* becomes parent of memory and document mappings

---

## Step 1.2

Create Document model

New table:

```python
Document
```

Fields:

```python
id

checksum
original_filename
stored_filename

status

page_count
chunk_count

created_at
updated_at
```

Requirements:

checksum unique

Example:

```text
checksum=abc123
```

must exist once only.

---

## Step 1.3

Create SessionDocument model

New table:

```python
SessionDocument
```

Fields:

```python
id

session_id FK
document_id FK

created_at
```

Requirements:

Unique constraint:

```python
(session_id, document_id)
```

to prevent duplicate mappings.

---

## Step 1.4

Modify DocumentChunk

Current:

```python
filename
```

Replace with:

```python
document_id FK
```

Add:

```python
parent_id
child_id
```

Optional:

```python
chunk_index
```

Useful later for sibling retrieval.

---

## Step 1.5

Migration

Create migration:

```text
sessions
documents
session_documents
```

Modify:

```text
document_chunks
```

Backfill if required.

---

# Phase 2

# Upload Flow Refactor

Goal:

Prevent duplicate embedding generation.

---

## Current Flow

```text
upload
    ↓
embed
```

---

## New Flow

```text
upload
    ↓
checksum
    ↓
document exists?
```

---

## Case A

Document exists

```text
create SessionDocument
return
```

No:

* extraction
* chunking
* embeddings

---

## Case B

Document missing

```text
create Document
extract
chunk
embed
store chunks
create SessionDocument
```

---

## Step 2.1

Create checksum utility

Function:

```python
calculate_checksum()
```

Requirements:

Stable SHA256.

---

## Step 2.2

Modify upload endpoint

Before embedding:

```python
lookup checksum
```

---

## Step 2.3

Document reuse logic

If checksum found:

```python
reuse document
```

Only create mapping.

---

## Step 2.4

Modify build_vector_database

Current:

```python
filename
```

Future:

```python
document_id
```

or

```python
document object
```

---

# Phase 3

# Parent / Child Chunking

Goal:

Store retrieval hierarchy.

---

## Parent Definition

Parent:

```text
grouped page window
```

Example:

```text
pages 1-2
```

---

## Child Definition

Child:

```text
actual searchable chunk
```

---

## Step 3.1

Modify chunk_pages()

Generate:

```python
parent_id
child_id
```

---

## Step 3.2

Store hierarchy

Inside chunk metadata and/or columns:

```python
parent_id
child_id
```

---

## Step 3.3

Update chunk inserts

Persist hierarchy fields.

---

# Phase 4

# Retrieval Refactor

Goal:

Search documents attached to a session.

---

## Current

Retriever receives:

```python
filenames
```

---

## New

Retriever receives:

```python
session_id
```

---

## Step 4.1

Resolve document ids

Query:

```text
SessionDocument
```

to obtain:

```python
document_ids
```

---

## Step 4.2

Replace filename filtering

Current:

```python
filename=...
```

Future:

```python
document_id=...
```

Affected:

* vector_search
* keyword_search
* hybrid_search

---

## Step 4.3

Update returned metadata

Include:

```python
document_id
checksum
original_filename
```

instead of filename-only assumptions.

---

# Phase 5

# Parent-Aware Retrieval

Goal:

Search children.
Answer using parent context.

---

## Current

```text
query
    ↓
hybrid search
    ↓
top chunks
```

---

## New

```text
query
    ↓
child retrieval
    ↓
group by parent
    ↓
expand context
    ↓
LLM
```

---

## Step 5.1

Return parent metadata

Search results must include:

```python
parent_id
child_id
```

---

## Step 5.2

Parent grouping

After retrieval:

```python
group_by(parent_id)
```

---

## Step 5.3

Parent scoring

Possible:

```python
parent_score=max(child_scores)
```

Initial implementation recommended.

---

## Step 5.4

Parent selection

Select:

```python
top N parents
```

instead of:

```python
top N chunks
```

---

## Step 5.5

Context expansion

For selected parents:

retrieve:

```python
all children
```

or:

```python
neighbor children
```

---

## Step 5.6

Build final context blocks

Instead of:

```text
chunk
chunk
chunk
```

Produce:

```text
Parent A
  chunk1
  chunk2

Parent B
  chunk1
  chunk2
```

---

# Phase 6

# Search Quality Improvements

Do NOT implement until previous phases are stable.

---

## Step 6.1

Deduplication

Remove:

```text
near-identical chunks
```

---

## Step 6.2

Parent diversity

Prevent:

```text
10 results from same parent
```

---

## Step 6.3

MMR

Add:

```text
relevance + diversity
```

selection.

---

## Step 6.4

Cross-Encoder Reranking

Optional.

Only after retrieval is stable.

---

# Phase 7

# Session APIs

Goal:

Support session lifecycle.

---

## Create Session

```http
POST /sessions
```

---

## List Sessions

```http
GET /sessions
```

---

## Get Session Documents

```http
GET /sessions/{id}/documents
```

---

## Add Document To Session

Handled by upload endpoint.

---

## Remove Document From Session

```http
DELETE /sessions/{id}/documents/{document_id}
```

Only removes mapping.

Must NOT delete:

```text
Document
DocumentChunk
```

unless orphan cleanup runs.

---

# Phase 8

# Cleanup Logic

Goal:

Prevent orphan accumulation.

---

## Orphan Document Cleanup

Document with:

```text
0 SessionDocument references
```

can be deleted.

---

## Cascade Delete

Deleting session:

```text
SessionDocument
ConversationMemory
ConversationSummary
```

should cascade.

---

# Recommended Atomic Execution Order

Execute in this exact order:

1. Create Session model
2. Create Document model
3. Create SessionDocument model
4. Modify DocumentChunk → document_id
5. Create migrations
6. Refactor upload flow
7. Implement checksum reuse
8. Refactor build_vector_database()
9. Implement parent_id / child_id
10. Persist hierarchy
11. Refactor retrieval to session_id
12. Refactor vector search
13. Refactor keyword search
14. Refactor hybrid search
15. Implement parent grouping
16. Implement parent expansion
17. Refactor context building
18. Add session APIs
19. Add orphan cleanup
20. Add MMR / reranking (optional)

---

# Prompt Strategy For Implementation

Use prompts in this format:

```text
Execute Phase 1 Step 1.1 only.

Rules:
- Do not touch any other phase.
- Show all files changed.
- Explain migration impact.
- Wait for approval before next step.
```

Then proceed step-by-step through the plan.
