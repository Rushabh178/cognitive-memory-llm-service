# Cognitive Memory LLM Service — Project Documentation

[← Back to README](../README.md)

This document explains how the Python LLM service works: what each part does, how data moves through it, every endpoint, the storage schema, the tuning constants and how to run, test and troubleshoot it.

---

## Table of contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation and running](#3-installation-and-running)
4. [Configuration](#4-configuration)
5. [API reference](#5-api-reference)
6. [Core pipelines](#6-core-pipelines)
7. [Data storage](#7-data-storage)
8. [Module reference](#8-module-reference)
9. [Tuning constants](#9-tuning-constants)
10. [Testing](#10-testing)
11. [Maintenance scripts](#11-maintenance-scripts)
12. [Security](#12-security)
13. [Failure handling](#13-failure-handling)
14. [Known limitations](#14-known-limitations)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Overview

An LLM has no memory between calls. This service builds one around it:

1. **Write.** Each user message is split into short, self-contained **atomic facts** (`"User works at Yardi"`). Each fact is stored twice:
   - as an embedding in **ChromaDB**, for semantic search;
   - as a node in a **PostgreSQL knowledge graph**, with structure: domain, status, timeline, importance and edges to related facts.
2. **Read.** Before a reply, the caller asks for relevant memories from three layers: vector search, graph context and a rolling summary.
3. **Reply.** `/ai/chat` sends the long-term context, the current session's turns and the new message to the LLM.
4. **Maintain.** Background jobs keep memory accurate. They mark tasks completed, replace revised facts, merge duplicates and build summaries.

The service is one of three in the **AI Cognitive Memory** system:

| Service | Role |
|---|---|
| React frontend | UI. Talks only to Spring Boot. |
| Spring Boot API | Users, auth (JWT), chat sessions and messages. **Orchestrates** the chat pipeline: decides when to retrieve and store memory, and calls this service. |
| **This service** | Everything "intelligent": fact extraction, embeddings, vector search, the knowledge graph, summarization and the LLM reply. |

---

## 2. Architecture

### 2.1 System context

```
┌───────────────────┐  JWT   ┌───────────────────┐  Bearer token  ┌────────────────────────────┐
│ React frontend    │ ─────► │ Spring Boot API   │ ─────────────► │ Python LLM service (:8000) │
│ :5173             │ ◄───── │ :8080             │ ◄───────────── │ FastAPI + Uvicorn          │
└───────────────────┘        └─────────┬─────────┘                └──────┬─────────┬─────────┬─┘
                                       │ JPA                             │         │         │
                                       ▼                                 ▼         ▼         ▼
                         ┌──────────────────────────────────────────────────┐ ┌──────────┐ ┌──────────┐
                         │ PostgreSQL "cognitive_memory" (shared database)  │ │ ChromaDB │ │ Groq API │
                         │  Spring tables: users, chat_sessions, ...        │ │ on disk  │ │ (LLM)    │
                         │  Python tables: graph_nodes, graph_edges,        │ └──────────┘ └──────────┘
                         │                 user_summaries                   │
                         └──────────────────────────────────────────────────┘
                                                   Redis: optional embedding cache, off by default
```

Spring Boot and this service share one PostgreSQL database but use separate tables. Spring sends the numeric user id as a string `userId` in each request.

### 2.2 How a chat message flows through the system

This is the sequence Spring Boot runs when a user sends a message:

```
1. Spring snapshots the last 10 turns of the session (sessionHistory).
2. If the message is personal (not "explain X" / "write code"):
       POST /memory/retrieve   → semantically similar facts
       POST /graph/context     → scored graph nodes + "graph_context_text"
3. Always:
       GET  /graph/summary/{userId} → rolling digest, if one exists
4. Spring joins those into one labelled "context" string.
5. POST /ai/chat {message, context, sessionHistory} → answer
6. POST /memory/store for the user message (and the assistant reply).
       → facts are stored in ChromaDB immediately
       → graph processing for each fact runs in the background
```

### 2.3 Inside this service

```
HTTP request
   │
   ▼
main.py ── log_requests middleware (timing + scheduler load counter)
   │        require_auth() → 401 if the bearer token is wrong
   │
   ├── /memory/store ──► atomic_extractor ──► embeddings ──► memory_store (ChromaDB)
   │                                                      └► BackgroundTask: graph_memory.process_memory_intent
   ├── /memory/retrieve ► embeddings ──► graph_memory.classify_domain_by_embedding ──► memory_store
   ├── /ai/chat ───────► ai_service.call_llm (Groq)
   └── /graph/* ───────► graph_routes ──► graph_memory (PostgreSQL)

scheduler.py (APScheduler, every 10 min) ──► summarization.generate_user_summary
```

---

## 3. Installation and running

### 3.1 Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | The project venv currently uses 3.14. |
| PostgreSQL | The database in `DB_NAME` must already exist. Tables and indexes are created on startup. |
| Groq API key | Free tier at [console.groq.com](https://console.groq.com). |
| Redis | Optional. Only used if `REDIS_ENABLED=true`. |
| Disk | About 80 MB for the embedding model, downloaded on first start, plus ChromaDB data. |

### 3.2 Setup

```bat
:: Windows
setup.bat
```

```bash
# Linux / macOS
chmod +x setup.sh && ./setup.sh
```

Both scripts create `venv/`, install `requirements.txt` and copy `.env.example` to `.env` if `.env` doesn't exist yet. Their final message mentions `ANTHROPIC_API_KEY`; that is out of date. The key you need is `GROQ_API_KEY`.

### 3.3 Run

```bash
# activate the venv first
uvicorn main:app --host 0.0.0.0 --port 8000
# or
python main.py      # runs uvicorn on config.PORT, without auto-reload
```

On startup the service:

1. Validates `GROQ_API_KEY` and `API_BEARER_TOKEN`. It raises an error and exits if either is missing.
2. Loads the embedding model and opens ChromaDB at `CHROMA_PERSIST_PATH`.
3. Creates or migrates the graph tables (`initialize_tables()`). If PostgreSQL is down, it logs an error and keeps running: the graph is not on the critical path.
4. Starts the summarization scheduler.

> **Run exactly one worker.** Don't use `uvicorn --workers N`. The scheduler's load gate keeps state in process memory, and ChromaDB's persistent client must not be written by two processes at once.

### 3.4 Interactive API docs

While the server is running, FastAPI serves:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

---

## 4. Configuration

All configuration comes from environment variables, loaded from `.env` by `config.py`. Real environment variables take precedence over `.env`.

### Required

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key. |
| `API_BEARER_TOKEN` | Shared secret. Every protected request must send `Authorization: Bearer <value>`. Must match the Spring Boot config. Generate one with `openssl rand -hex 32`. |

### LLM

| Variable | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `gpt-oss-20b` | Model for the chat reply, summaries and internal yes/no judgements (task matching, domain resolution, clustering). |
| `FAST_LLM_MODEL` | `openai/gpt-oss-20b` | Smaller model for structured extraction: fact splitting and graph metadata. |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per chat reply. |
| `LLM_TEMPERATURE` | `0.7` | Default temperature for chat, summaries and clustering. Task matching and domain resolution use `0`; fact splitting uses `0.1`. |

### Storage

| Variable | Default | Description |
|---|---|---|
| `CHROMA_PERSIST_PATH` | `./chroma_data` | ChromaDB directory. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model name. Changing it makes existing vectors incompatible. |
| `DB_HOST` | `localhost` | PostgreSQL host. |
| `DB_PORT` | `5432` | PostgreSQL port. |
| `DB_NAME` | `cognitive_memory` | Database name (shared with Spring Boot). |
| `DB_USER` | `postgres` | Database user. |
| `DB_PASSWORD` | *(empty)* | Database password. |
| `GRAPH_PERSIST_PATH` | `./graph_data` | Legacy: from when the graph was stored as JSON files. Only logged now. |

### Optional

| Variable | Default | Description |
|---|---|---|
| `REDIS_ENABLED` | `false` | Turn on the Redis embedding cache (24-hour TTL). |
| `REDIS_HOST` | `localhost` | Redis host. |
| `REDIS_PORT` | `6379` | Redis port. |
| `PORT` | `8000` | Port used by `python main.py`. |

---

## 5. API reference

### Conventions

- **Auth:** every endpoint except `GET /health` requires `Authorization: Bearer <API_BEARER_TOKEN>`. A missing or wrong token returns `401`.
- **Validation errors:** a malformed body returns `422`, generated by FastAPI/Pydantic.
- **Dependency failures:** if ChromaDB, PostgreSQL or the LLM fails, endpoints return `503` with a `detail` message.
- **Unhandled errors** return `500` with `{"error": "Internal server error", "detail": "..."}`.
- `userId` is always a string.

### 5.1 Memory endpoints

#### `POST /memory/store`

Store one message. For `role: "user"` the text is split into atomic facts, and each fact is stored as its own entry. For `role: "assistant"` the text is stored as-is.

**Request**

```json
{
  "userId": "42",
  "sessionId": "optional-session-id",
  "text": "My name is Rushabh and I work at Yardi",
  "role": "user"
}
```

**Response `200`**

```json
{
  "status": "stored",
  "ids": ["42_1759400000_a1b2c3d4", "42_1759400000_e5f6a7b8"],
  "facts_count": 2
}
```

If nothing in the message is worth storing (for example, it was only a question), the response is `{"status": "filtered", "ids": [], "facts_count": 0}`.

The response returns as soon as the facts are in ChromaDB. Graph processing for each fact continues in the background (see [6.2](#62-background-graph-processing)).

#### `POST /memory/retrieve`

Semantic search over one user's facts.

**Request**

```json
{ "userId": "42", "query": "Where do I work?", "topK": 5 }
```

**Response `200`**

```json
{ "memories": ["User works at Yardi"] }
```

Only `role: "user"` entries are searched. Matches weaker than the relevance threshold are dropped, and so are superseded (old) values. An empty list is a normal result.

#### `GET /memory/all/{userId}`

Every stored entry for a user, oldest first, including assistant messages and superseded facts.

```json
{
  "userId": "42",
  "count": 2,
  "memories": [
    { "text": "User works at Yardi", "role": "user", "sessionId": "", "timestamp": 1759400000.12, "supersededBy": null }
  ]
}
```

#### `DELETE /memory/{userId}`

Delete one user's data from **both** ChromaDB and the graph, in place. No restart needed, and other users are untouched.

```json
{ "userId": "42", "vectorMemoriesDeleted": 12, "graphNodesDeleted": 9 }
```

### 5.2 Chat endpoint

#### `POST /ai/chat`

Generate a reply. This endpoint does **not** retrieve or store memory itself. The caller passes in the context it already retrieved.

**Request**

```json
{
  "userId": "42",
  "message": "Any tips for my interview next week?",
  "context": "Related past memories:\nUser has an Amazon interview next week\n\nMemory Graph Context:\n...",
  "sessionHistory": [
    { "role": "user", "content": "Hi!" },
    { "role": "assistant", "content": "Hey! How can I help?" }
  ]
}
```

| Field | Meaning |
|---|---|
| `context` | Long-term memory: retrieved facts, graph context text and the summary, joined into one string. Can be empty. |
| `sessionHistory` | Prior turns of the **current** session, oldest first. Sent to the LLM as real chat turns. Entries with a role other than `user`/`assistant` are dropped. |

**Response `200`**

```json
{ "answer": "Good luck with Amazon! Since you've been practising DSA..." }
```

How the prompt is built is described in [6.4](#64-prompt-construction).

### 5.3 Health

#### `GET /health` (no auth)

```json
{ "status": "ok", "chromadb": "ok", "embeddings": "ok", "llm": "groq/gpt-oss-20b", "graph": "ok" }
```

`status` is `ok` only if ChromaDB, the embedding model and the PostgreSQL connection all work. `llm` just reports the configured model; it is not a live check.

### 5.4 Graph endpoints

All graph endpoints are mounted under `/graph`.

#### `POST /graph/process`

Add one memory to the graph directly. `/memory/store` already does this in the background for every user fact, so you only need this endpoint for manual or external writes.

**Request**

```json
{ "userId": "42", "text": "User has an Amazon interview next week", "memoryId": "42_1759400000_a1b2c3d4" }
```

`memoryId` should be the id returned by `/memory/store`. It becomes the graph node id, which keeps the two stores linked.

**Response `200`**

```json
{
  "status": "stored",
  "node_id": "42_1759400000_a1b2c3d4",
  "entity_name": "Amazon interview",
  "domain": "job",
  "sub_domain": "interview preparation",
  "status_label": "ongoing",
  "timeline_label": "October 2026",
  "importance_score": 0.6
}
```

If the text is a question rather than a statement, nothing is written and the response has `"status": "skipped"`, an empty `node_id` and placeholder fields.

#### `POST /graph/context`

The most relevant current (non-archived) graph nodes for a query, their neighbours, summary counts and a ready-made prompt string.

**Request**

```json
{ "userId": "42", "query": "How is my Amazon prep going?", "topN": 5 }
```

**Response `200`**

```json
{
  "nodes": [
    {
      "node_id": "...", "text": "User has an Amazon interview next week",
      "entity_name": "Amazon interview", "domain": "job", "sub_domain": "interview preparation",
      "status": "ongoing", "timeline_label": "October 2026", "importance_score": 0.6
    }
  ],
  "neighbors": [],
  "timeline_summary": { "this_month": 4, "ongoing": 6, "completed": 2, "domains": { "job": 3, "hobbies": 5 } },
  "graph_context_text": "Memory Graph Context:\n\nRecent memories (4 this month):\n- [ONGOING] Job: Amazon interview (October 2026)"
}
```

Scoring is explained in [6.3](#63-memory-retrieval).

#### `GET /graph/summary/{userId}`

The latest rolling summary. Having no summary yet is normal, not an error.

```json
{
  "userId": "42",
  "hasSummary": true,
  "summary": "Active goals:\n- Preparing for Amazon interview\nRecent completions:\n- Finished AWS certification\nStanding facts:\n- Works at Yardi",
  "generatedAt": "2026-10-02T09:30:00",
  "sourceNodeCount": 14
}
```

or `{ "userId": "42", "hasSummary": false, "summary": null }`.

#### `GET /graph/history/{userId}`

How facts have changed over time. Spring calls this when the user asks a history question such as "who was my favorite player before?".

| Query param | Default | Meaning |
|---|---|---|
| `query` | — | Used to pick a domain if `domain` isn't given. |
| `domain` | — | Restrict to one domain. |
| `limit` | `5` | Max chains returned (capped at 20). |

```json
{
  "userId": "42",
  "domain": "hobbies",
  "count": 1,
  "history_context_text": "- Favorite cricket players: previously Rohit Sharma, replaced by Virat Kohli (current)"
}
```

#### `GET /graph/timeline/{userId}?domain=job`

Full history grouped by month, then domain, then status. Archived nodes are included and labelled with what replaced them. Meant for UI views, not chat context.

```json
{
  "userId": "42",
  "domain": null,
  "timeline": {
    "October 2026": {
      "hobbies": { "ongoing": ["Virat Kohli", "Rohit Sharma (replaced by Virat Kohli)"], "completed": [] },
      "job": { "ongoing": ["Amazon interview"], "completed": [] }
    }
  }
}
```

#### `POST /graph/status-update`

Set a node's status manually.

```json
{ "userId": "42", "nodeId": "42_1759400000_a1b2c3d4", "newStatus": "completed" }
```

```json
{ "updated": true, "nodeId": "42_1759400000_a1b2c3d4", "newStatus": "completed" }
```

`newStatus` must be `ongoing` or `completed`; any other value returns `400`.

#### `GET /graph/all?include_archived=false`

**Admin/debug only.** Returns every node for **all users**. `include_archived=true` also returns merged and revised nodes.

#### `DELETE /graph/{userId}`

Delete only the user's graph (nodes and edges). Their ChromaDB memories are kept. Use `DELETE /memory/{userId}` to clear both stores.

```json
{ "userId": "42", "graphNodesDeleted": 9, "graphEdgesDeleted": 14 }
```

---

## 6. Core pipelines

### 6.1 Memory write (`POST /memory/store`)

```
user text
  │
  ├─ 1. atomic_extractor.extract_facts_with_metadata(text)       ← one FAST_LLM_MODEL call
  │      Splits into facts AND classifies each one together:
  │      [{text: "User works at Yardi", metadata: {domain: "job", status: "ongoing", ...}}, ...]
  │      • questions/requests are never turned into facts → [] if the message is only a question
  │      • on any failure (bad JSON, 429, ...) → falls back to the whole message, marked fallback=True
  │
  ├─ 2. Fallback cleanup: drop_question_sentences() removes question sentences from the raw message
  │
  ├─ 3. Quality filter: is_fact_worth_storing()
  │      drops facts under 4 words and meta statements ("user asked", "user wants to know", ...)
  │      nothing left → return status "filtered"
  │
  ├─ 4. embeddings.encode_batch(facts)                          ← one batched model call
  │
  ├─ 5. memory_store.store_memory() for each fact, in parallel on a 4-thread pool
  │      id = "{userId}_{unixTime}_{8 hex chars}"
  │
  └─ 6. Respond, then schedule process_graph_background() once per fact (user role only)
```

**Why split and classify in one call?** When each fact was classified on its own, sibling facts lost their shared context. In "For football my favorite is Messi and for tennis it's Federer", the tennis half was sometimes classified as a completed task. Classifying them together keeps sibling facts consistent.

### 6.2 Background graph processing

`main.process_graph_background()` runs after the response is sent, once per fact. Any failure is logged and ignored, so the user never sees it.

`graph_memory.process_memory_intent()` decides what to do with the fact:

| Situation | Action |
|---|---|
| Text is a question (`is_declarative()` is false) | **skipped**: nothing is written |
| Extraction failed (degraded input) | **created**: a plain node with `needs_reprocessing = TRUE`. It never matches, edits or archives other nodes. |
| Reports that an ongoing task finished (`progression` + `completed`) | **updated**: the existing task node is set to `completed` |
| Adds new detail to an ongoing task (`progression` + `ongoing`) | **updated**: the task node's content is edited in place |
| Gives a single-value fact a new value (`revision`), e.g. favorite player, phone number, job title | **superseded**: a new node is created and the old node is archived with `superseded_by` and `supersede_reason = 'revision'` |
| Anything else | **created**: a new node with edges to related nodes |

Matching against existing nodes (`find_matching_task_node()`) is an LLM judgement at temperature 0. It only counts as a match at confidence ≥ `TASK_MATCH_THRESHOLD` (0.85), and only considers ongoing nodes in the same domain.

After the decision, the background task also:

- copies the resolved `domain` onto the fact's ChromaDB entry (`update_memory_domain`), so domain-filtered retrieval can find it;
- on **superseded**, flags the old value's ChromaDB entry with `superseded_by`, so retrieval stops returning it (the entry is kept, not deleted);
- on **created** (not degraded), runs `resolve_conflicting_nodes()` for that domain. An LLM clusters up to the 20 most recent active nodes, keeps the newest node in each cluster and archives the rest with `supersede_reason = 'merge'`.

**Domain classification** (inside `extract_graph_metadata` / `normalize_graph_metadata`):

1. `domain` must be one of `job, education, hobbies, health, finance, relationships, travel, food, general`. Finer detail goes in the free-form `sub_domain`.
2. Out-of-set labels are mapped with a fixed alias table first (`career → job`, `sports → hobbies`, ...).
3. Then `classify_domain_by_embedding()` compares the text's embedding to the centroid of each of the user's existing domains. Cosine ≥ 0.70 is accepted without an LLM call.
4. Otherwise `resolve_domain()` asks the LLM, accepting confidence ≥ 0.80, and falls back to `general`.

**Edges** connect a new node to existing nodes in the same domain and month. The type is `LED_TO` when the node completes something that was ongoing, otherwise `RELATED_TO`. Strength = `0.5 × cosine(text_a, text_b) + 0.5 × heuristic`, with a heuristic of 0.7 for `LED_TO` and 0.6 for `RELATED_TO`.

### 6.3 Memory retrieval

**Vector layer (`POST /memory/retrieve`)**

1. Embed the query.
2. Classify the query's domain with the embedding-centroid classifier. The LLM is never used here, to keep retrieval fast. No confident match means all domains are searched.
3. Query ChromaDB for `topK × 2` candidates, filtered to this `userId`, `role = "user"` and the domain if there is one.
4. Keep candidates with distance below `1.8` (squared L2; see [9](#9-tuning-constants)) that are not superseded, and return up to `topK`.

**Graph layer (`POST /graph/context`)**

Each active node gets a score:

```
score = importance × 0.4 + recency × 0.3 + keyword_match × 0.3

recency:        same month 1.0 · last month 0.7 · within 3 months 0.4 · older 0.1
keyword_match:  share of query words (longer than 2 chars) that appear in the node
```

A node is kept only if `score ≥ 0.45` **and** it matches at least one query keyword (when the query has keywords). The top `topN` nodes are returned with their edge neighbours, and their `last_accessed_at` is updated.

**Summary layer (`GET /graph/summary/{userId}`)**: the latest digest. See [6.5](#65-background-summarization).

### 6.4 Prompt construction

`ai_service.call_llm()` sends this to Groq:

```
[system]    build_system_prompt(context, has_session_history)
[user]      sessionHistory turn 1          ┐
[assistant] sessionHistory turn 2          │ the current session, as real turns
...                                        ┘
[user]      message
```

The system prompt:

- **With no context:** a plain helpful-assistant prompt.
- **With context:** includes the context as *background* and tells the model to use it naturally, never saying "based on my memory".
- **With context and session history:** also tells the model that the current conversation is more authoritative than the background, so a recent statement always beats an older memory.
- **Context-fidelity rule:** don't reinterpret stored facts into unrelated categories, and don't call something unanswered if the context already answers it.
- **Formatting rule:** Markdown only, no HTML tags such as `<br>` (the frontend doesn't render raw HTML), and one line per table cell.

### 6.5 Background summarization

`scheduler.py` runs a tick every **10 minutes**:

1. Find users with graph nodes not yet summarized (`summarized_at IS NULL`, not archived).
2. For each user, but at most 5 per tick, check `is_low_load(userId)`:
   - no more than **2** HTTP requests in flight across the whole service, and
   - the user hasn't called store, retrieve or chat for at least **5 minutes**.
3. `summarization.generate_user_summary()` takes up to 50 of the oldest pending nodes and asks the LLM for a three-section digest: *Active goals*, *Recent completions*, *Standing facts*.
4. The `user_summaries` insert and the `summarized_at` update on those nodes happen in **one transaction**, so a crash can't lose track of which nodes were covered. Re-running is safe.

---

## 7. Data storage

### 7.1 ChromaDB

- Persistent client at `CHROMA_PERSIST_PATH`, with one collection named `memories` shared by all users.
- Distance: Chroma's default, **squared L2**. The collection was not created with cosine space.
- Every query filters by `userId`. That filter is the privacy boundary between users.

| Field | Content |
|---|---|
| `id` | `{userId}_{unixTime}_{8 hex}`. The graph node created from it has the same id. |
| `document` | The fact text, or the assistant message text. |
| `embedding` | 384-dimensional MiniLM vector. |
| `metadata.userId` | Owner. |
| `metadata.sessionId` | Session, or `""`. |
| `metadata.role` | `user` or `assistant`. |
| `metadata.timestamp` | Unix time (float). |
| `metadata.domain` | Added later by the background graph task. |
| `metadata.superseded_by` | Set when a newer value replaced this fact. |

### 7.2 PostgreSQL

Created and migrated automatically on startup by `graph_memory.initialize_tables()`. Every statement uses `IF NOT EXISTS`, so it is safe to run repeatedly.

**`graph_nodes`**

| Column | Type | Notes |
|---|---|---|
| `id` | VARCHAR PK | Same as the ChromaDB id. |
| `user_id` | VARCHAR | Owner (indexed). |
| `entity_name` | VARCHAR | Main subject, e.g. `Amazon interview`. |
| `domain` | VARCHAR | One of the fixed top-level domains. |
| `sub_domain` | VARCHAR | Free-form, e.g. `favorite cricket players`. |
| `status` | VARCHAR | `ongoing` / `completed`. |
| `timeline_year`, `timeline_month`, `timeline_label` | INT, INT, VARCHAR | e.g. `2026`, `10`, `October 2026`. |
| `importance_score` | FLOAT | 0.0–1.0. |
| `text` | TEXT | The fact. |
| `created_at`, `last_updated`, `last_accessed_at` | TIMESTAMP | `last_accessed_at` is updated on every retrieval. |
| `superseded_by` | VARCHAR FK → `graph_nodes.id` | Set when the node is archived. |
| `archived_at` | TIMESTAMP | When it was archived. |
| `supersede_reason` | VARCHAR | `revision` or `merge`. |
| `summarized_at` | TIMESTAMP | When it was folded into a summary. |
| `needs_reprocessing` | BOOLEAN | Written from fallback metadata. |

**`graph_edges`**

| Column | Type | Notes |
|---|---|---|
| `id` | SERIAL PK | |
| `user_id` | VARCHAR | |
| `source_node_id`, `target_node_id` | VARCHAR FK (cascade delete) | |
| `relationship` | VARCHAR | `RELATED_TO` or `LED_TO`. |
| `strength` | FLOAT | See [6.2](#62-background-graph-processing). |
| `created_at` | TIMESTAMP | |

**`user_summaries`**

| Column | Type |
|---|---|
| `id` | SERIAL PK |
| `user_id` | VARCHAR |
| `summary_text` | TEXT |
| `generated_at` | TIMESTAMP |
| `source_node_count` | INT |

Archived nodes are never deleted. Current-state reads (`/graph/context`, matching, merging, summarization) exclude them. `/graph/timeline` and `/graph/history` show them as history.

---

## 8. Module reference

| Module | Responsibility | Key functions |
|---|---|---|
| `main.py` | App, middleware, auth, core endpoints, background graph task | `require_auth`, `store_memory_endpoint`, `retrieve_memory_endpoint`, `ai_chat_endpoint`, `process_graph_background`, `health_endpoint` |
| `graph_routes.py` | `/graph/*` router | One handler per graph endpoint, `_require_auth` |
| `config.py` | Load `.env`, defaults, startup validation | — |
| `models.py` | Pydantic request/response contracts | `MemoryStoreRequest`, `AiChatRequest`, `GraphContextResponse`, ... |
| `ai_service.py` | System prompt and Groq chat call | `build_system_prompt`, `call_llm` |
| `atomic_extractor.py` | Fact splitting, classification and filters | `extract_facts_with_metadata`, `drop_question_sentences`, `is_fact_worth_storing` |
| `embeddings.py` | Load the model once; optional Redis cache | `encode`, `encode_batch` |
| `memory_store.py` | The only module that touches ChromaDB | `store_memory`, `retrieve_memories`, `update_memory_domain`, `mark_memory_superseded`, `delete_user_memories`, `get_all_memories` |
| `graph_memory.py` | Knowledge graph: connection pool, schema, classification, matching, revision, merging, scoring, history | `initialize_tables`, `process_memory_intent`, `classify_domain_by_embedding`, `resolve_domain`, `find_matching_task_node`, `add_to_graph`, `supersede_node`, `resolve_conflicting_nodes`, `get_graph_context`, `get_history_context`, `get_timeline_summary`, `delete_user_graph` |
| `summarization.py` | Rolling per-user digest | `get_users_with_pending_nodes`, `generate_user_summary`, `get_latest_summary` |
| `scheduler.py` | Low-load gate and APScheduler job | `mark_request_start/end`, `mark_user_active`, `is_low_load`, `start`, `shutdown` |
| `backfill_legacy_domains.py` | One-off domain backfill | `main` |

### Every LLM call

| Where | Model | Temperature | Purpose |
|---|---|---|---|
| `atomic_extractor.extract_facts_with_metadata` | `FAST_LLM_MODEL` | 0.1 | Split and classify facts |
| `graph_memory.extract_graph_metadata` | `FAST_LLM_MODEL` | `LLM_TEMPERATURE` | Classify one fact when no extractor metadata is available |
| `graph_memory.resolve_domain` | `GROQ_MODEL` | 0 | Map a stray domain label onto the fixed set |
| `graph_memory.find_matching_task_node` | `GROQ_MODEL` | 0 | Is this an update to an existing node? |
| `graph_memory._cluster_nodes_via_llm` | `GROQ_MODEL` | `LLM_TEMPERATURE` | Cluster duplicates for merging |
| `summarization.generate_user_summary` | `GROQ_MODEL` | `LLM_TEMPERATURE` | Rolling digest |
| `ai_service.call_llm` (chat) | `GROQ_MODEL` | `LLM_TEMPERATURE` | The reply |

---

## 9. Tuning constants

These live in code, not in `.env`.

| Constant | File | Value | Effect |
|---|---|---|---|
| `RELEVANCE_THRESHOLD` | `memory_store.py` | `1.8` | Max ChromaDB distance (squared L2). Measured: related pairs about 1.4–1.8, unrelated about 1.8–2.1. **Re-tune if the collection is ever recreated with cosine space** (a value around 0.7 would fit then). |
| `RELEVANCE_THRESHOLD` | `graph_memory.py` | `0.45` | Min graph score. Raise it if unrelated context leaks into prompts; lower it if relevant memories are dropped. |
| `DOMAIN_CLASSIFIER_THRESHOLD` | `graph_memory.py` | `0.70` | Min centroid cosine to skip the domain LLM call. Calibrated on third-person facts; still unvalidated on raw queries. |
| `DOMAIN_MATCH_THRESHOLD` | `graph_memory.py` | `0.80` | Min LLM confidence to map a stray domain label. |
| `TASK_MATCH_THRESHOLD` | `graph_memory.py` | `0.85` | Min LLM confidence to treat text as an update to an existing node. Raised from 0.75 after false positives. |
| `CONFLICT_RESOLUTION_TRIGGER_LIMIT` | `graph_memory.py` | `20` | Nodes per domain checked by the per-write merge pass. |
| `EDGE_SIMILARITY_WEIGHT` | `graph_memory.py` | `0.5` | Text-similarity share of edge strength. |
| `HISTORY_CONTEXT_LIMIT` | `graph_memory.py` | `5` | Default chains for `/graph/history`. |
| `SUMMARY_TICK_INTERVAL_MINUTES` | `scheduler.py` | `10` | Scheduler period. |
| `MAX_CONCURRENT_REQUESTS_FOR_SUMMARY` | `scheduler.py` | `2` | Global load gate. |
| `USER_IDLE_SECONDS_FOR_SUMMARY` | `scheduler.py` | `300` | Per-user idle gate. |
| `MAX_USERS_PER_TICK` | `scheduler.py` | `5` | Users summarized per tick. |
| `MAX_NODES_PER_RUN` | `summarization.py` | `50` | Nodes folded into one summary. |

---

## 10. Testing

```bash
# from the service directory, with the venv active
python tests/test_fact_regressions.py
# or, if pytest is installed (it is not in requirements.txt)
pytest tests/test_fact_regressions.py
```

`tests/test_fact_regressions.py` contains **live** regression tests for fact extraction and supersession:

- They call the **real Groq API** and write to the **real PostgreSQL** tables under a throwaway user id, which is always deleted afterwards.
- ChromaDB is redirected to a temporary directory before any service module is imported, so the real vector store is never touched and the tests can run while the server is up.
- They **skip** (they don't fail) when config, the database or LLM quota is unavailable. A preflight call checks for Groq rate limits.
- `REGRESSION_RUNS` (default `3`) repeats the extraction test, because LLM output varies between runs.
- `test_failed_extraction_never_archives_existing_nodes` forces an extraction failure and needs no LLM quota.

If quota is close to the limit, the preflight can pass and later calls can still fail. Those failures show up as extractor fallbacks, not as real regressions.

---

## 11. Maintenance scripts

### `backfill_legacy_domains.py`

Gives a `domain` to ChromaDB user memories stored before domain tagging existed, so domain-filtered retrieval doesn't hide them.

```bash
python backfill_legacy_domains.py
```

For each memory without a domain, it tries the embedding classifier first, then the LLM, and writes the result to the entry's metadata. It is safe to re-run: only entries still missing a domain are touched.

### Clearing a user's data

Use the API rather than deleting files while the server is running:

```bash
curl -X DELETE http://localhost:8000/memory/42 -H "Authorization: Bearer $API_BEARER_TOKEN"
```

---

## 12. Security

- **Service-to-service auth.** One static bearer token shared with Spring Boot. End-user identity is handled by Spring; this service trusts the `userId` it receives.
- **Expose it only to Spring Boot.** Anyone holding the token can read or delete any user's data. Keep the port off the public internet.
- **Per-user isolation.** Every ChromaDB query filters by `userId`, and every graph query filters by `user_id`.
- **`GET /graph/all`** returns every user's data. Treat it as an admin endpoint.
- **Secrets** belong in `.env`, which is git-ignored. Never commit it.
- **Logs** include memory text, retrieved context and full system prompts (some at `INFO` with a `DEBUG` prefix). Handle log files as user data.

---

## 13. Failure handling

| Failure | Behaviour |
|---|---|
| Fact extraction fails (bad JSON, rate limit) | The whole message is stored as one fact, minus question sentences. Its graph node is marked `needs_reprocessing` and never edits existing nodes. |
| Background graph processing fails | Logged as a warning. The stored memory and the chat are unaffected. |
| PostgreSQL down at startup | Logged. The service still starts; `/health` reports `graph: error`. |
| Connection pool exhausted | Falls back to a direct connection. |
| ChromaDB or embeddings fail during store/retrieve | `503`. |
| Groq fails during chat | `503 "AI model unavailable"`. |
| Redis unavailable | One warning, then runs uncached. |
| Summarization fails | Transaction rolled back; the nodes stay pending for the next tick. |

---

## 14. Known limitations

- **Single process only.** The scheduler's load gate is in memory, and ChromaDB must have a single writer.
- **Merges don't flag ChromaDB.** When `resolve_conflicting_nodes` archives a duplicate (`merge`), its ChromaDB entry is not flagged and can still be retrieved. Only revisions flag the old entry. This is deliberate, because LLM clustering can be wrong.
- **The query-time domain classifier is uncalibrated.** It was tuned on third-person facts, but at query time it sees raw questions. A wrong confident match would narrow retrieval to the wrong domain.
- **Classifier cost grows with memory.** `classify_domain_by_embedding()` re-embeds a user's graph texts to build centroids.
- **The per-write merge pass is bounded.** It only checks the 20 most recent nodes per domain, so duplicates of very old nodes need a full sweep (`resolve_conflicting_nodes(userId)` with no limit).
- **Assistant messages are write-only.** They are stored, but retrieval only returns `role = "user"` entries.
- **Leftovers.** `GRAPH_PERSIST_PATH` is no longer used for storage. `main.py` has some unused imports (`click`, `typer`, `matplotlib`, `multiprocessing`).

---

## 15. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ValueError: GROQ_API_KEY is not set` on start | Fill in `.env`, or set the variable in your environment. |
| `401 Unauthorised` | The `Authorization` header must be exactly `Bearer <API_BEARER_TOKEN>`, with the same value Spring Boot uses. |
| `/health` shows `graph: error` | PostgreSQL isn't reachable or `DB_*` is wrong. Check that the database in `DB_NAME` exists. |
| First start is slow | The embedding model is downloading (about 80 MB). Later starts use the local cache. |
| `/memory/store` returns `filtered` | The message held no storable facts (questions only, or fragments under 4 words). This is expected. |
| Retrieval returns nothing for an obvious match | The memory may still be waiting for its background domain tag, or the query was classified into the wrong domain. Check the `classify_domain_by_embedding` scores in the logs. |
| Many facts stored as whole messages | Extraction is falling back, usually because of Groq rate limits (429). Check the logs for `extract_facts_with_metadata failed`. |
| No summary appears | Summaries only run when the service is idle and the user has been inactive for 5+ minutes, checked every 10 minutes. |
| ChromaDB lock or corruption errors | Make sure only one process (one server, no extra workers, no test using the real path) writes to `CHROMA_PERSIST_PATH`. |

---

[← Back to README](../README.md)
