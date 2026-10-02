# ============================================================
# FILE: main.py
#
# WHAT THIS FILE DOES:
# The entry point for the FastAPI application. Defines all four API endpoints,
# enforces bearer token authentication on protected routes, handles startup
# logging via a lifespan context manager, registers a global error handler,
# and logs every request and response with timing information.
#
# KEY CONCEPT FOR LEARNING:
# FastAPI uses Python type hints and Pydantic models to do two things at once:
# validate incoming JSON AND generate OpenAPI documentation automatically.
# When you declare a function parameter as `body: MemoryStoreRequest`, FastAPI
# parses the JSON, validates every field, and gives you a typed Python object —
# no manual json.loads() or type checking needed.
#
# WHERE IT FITS IN THE PIPELINE:
# HTTP request → main.py (auth + routing) → memory_store / ai_service → HTTP response
# ============================================================

from multiprocessing import context
import asyncio
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from click import prompt
from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from matplotlib.style import context

from typer import prompt

import config
import embeddings
import memory_store
import ai_service
import graph_routes
import scheduler
from atomic_extractor import drop_question_sentences, extract_facts_with_metadata, is_fact_worth_storing
from models import (
    MemoryStoreRequest,
    MemoryStoreResponse,
    MemoryRetrieveRequest,
    MemoryRetrieveResponse,
    AiChatRequest,
    AiChatResponse,
    HealthResponse,
)

# ---------------------------------------------------------------
# Configure logging so every log line includes a timestamp,
# the log level, and the message. This output goes to stdout,
# which Docker / systemd / your terminal can capture.
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Thread pool for running sync ChromaDB writes (memory_store.store_memory)
# concurrently. ChromaDB's client is sync, so each store call blocks a
# thread — running the per-fact stores in parallel here means storing 3
# atomic facts costs roughly one write's worth of wall time instead of
# three sequential ones.
# ---------------------------------------------------------------
_store_executor = ThreadPoolExecutor(max_workers=4)


# ---------------------------------------------------------------
# Lifespan context manager — runs startup code before the server
# accepts traffic, and shutdown code when it stops.
#
# Using @asynccontextmanager + lifespan is the modern FastAPI way
# (replaces the deprecated @app.on_event("startup") pattern).
# ---------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    # These log lines confirm that all three heavy dependencies
    # (ChromaDB, embedding model, LLM config) are ready before
    # the first request arrives.
    logger.info(f"ChromaDB initialised at {config.CHROMA_PERSIST_PATH}")
    logger.info("Embedding model loaded: all-MiniLM-L6-v2")
    logger.info(f"LLM ready: {config.LLM_MODEL}")
    logger.info(f"Graph memory initialised at {config.GRAPH_PERSIST_PATH}")
    logger.info(f"Server starting on port {config.PORT}")

    # Initialize PostgreSQL graph tables (safe to run on every startup)
    try:
        from graph_memory import initialize_tables
        initialize_tables()
        logger.info("Graph tables ready in PostgreSQL")
    except Exception as e:
        logger.error(f"Graph table init failed: {e}")
        # Don't crash startup — graph is not on the critical path

    # Background summarization scheduler (Task 1) — in-process, no broker.
    # Never blocks startup or the request path; see scheduler.py for the
    # low-load gating that decides when it's actually allowed to do work.
    scheduler.start()

    yield  # Server is now running and accepting requests

    # --- SHUTDOWN ---
    scheduler.shutdown()
    logger.info("Server shutting down.")


# ---------------------------------------------------------------
# Create the FastAPI app instance.
# The lifespan argument wires up the startup/shutdown hooks above.
# ---------------------------------------------------------------
app = FastAPI(
    title="AI Cognitive Memory System",
    description="RAG-based memory service for persistent AI conversations",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount the graph memory router under the /graph prefix.
# All endpoints in graph_routes.py are now reachable at /graph/...
app.include_router(graph_routes.router, prefix="/graph", tags=["Graph Memory"])


# ---------------------------------------------------------------
# Global exception handler.
# If ANY unhandled exception bubbles up from an endpoint,
# this catches it and returns a consistent JSON error shape
# with HTTP 500 — instead of FastAPI's default HTML error page.
# ---------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------
# Middleware for request/response logging.
# Logs every request (method + path) before it's handled, and
# every response (status code + elapsed time) after.
# This gives an audit trail without touching individual endpoints.
# ---------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log the incoming request
    logger.info(f"REQUEST  {request.method} {request.url.path}")

    # Track this request as "in flight" for the summarization scheduler's
    # low-load gate (scheduler.is_low_load()) — incremented/decremented
    # around every request regardless of path, so it reflects total service
    # load, not just chat traffic.
    scheduler.mark_request_start()
    try:
        # Pass the request to the actual endpoint handler
        response = await call_next(request)
    finally:
        scheduler.mark_request_end()

    # Calculate how long the endpoint took in milliseconds
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"RESPONSE {request.method} {request.url.path} "
        f"→ {response.status_code} ({elapsed_ms:.1f}ms)"
    )

    return response


# ---------------------------------------------------------------
# Auth helper — shared by all protected endpoints.
#
# Reads the Authorization header and checks it matches:
#   "Bearer <API_BEARER_TOKEN>"
#
# Returns nothing on success. Raises HTTP 401 on failure.
# We call this at the top of every protected endpoint handler.
# ---------------------------------------------------------------
def require_auth(authorization: Optional[str]) -> None:
    """
    Validate the Bearer token in the Authorization header.

    Raises HTTP 401 if the header is missing, malformed, or the token is wrong.
    """
    expected = f"Bearer {config.API_BEARER_TOKEN}"
    if not authorization or authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Unauthorised: missing or invalid Bearer token.",
        )


# ================================================================
# ENDPOINT 1: POST /memory/store
# Stores a single message (user or assistant) as a searchable memory.
# ================================================================
@app.post("/memory/store", response_model=MemoryStoreResponse)
async def store_memory_endpoint(
    body: MemoryStoreRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    # Auth: reject any request without a valid Bearer token
    require_auth(authorization)

    # Log which user is storing a memory (useful for debugging)
    logger.info(f"store_memory  userId={body.userId}  role={body.role}")
    scheduler.mark_user_active(body.userId)

    try:
        # Step 1: Split compound user statements into individual atomic facts
        # ("My name is Rushabh and I work at Yardi" → two facts) so each one
        # can be retrieved precisely later instead of pulling in a whole
        # sentence to get part of it. The same LLM call also classifies each
        # fact (domain/status/importance...) with the whole message as
        # context, so sibling facts are judged consistently; that metadata is
        # handed to the background graph task below instead of a second LLM
        # call per fact. Assistant responses are stored verbatim — they're
        # conversational text, not a set of facts to decompose.
        fact_metadata: list = []
        if body.role == "user":
            extracted = extract_facts_with_metadata(body.text)

            # If extraction failed, the fallback item is the whole raw message.
            # Never store a question from it as a "fact" — keep only its
            # statements (a successful extraction already drops questions).
            # A stored question is later retrieved as a past memory with no
            # answer, and the model treats it as an open question.
            cleaned = []
            for item in extracted:
                if item["fallback"]:
                    statements = drop_question_sentences(item["text"])
                    if statements != item["text"]:
                        logger.info(
                            f"Extraction fallback for userId={body.userId}: dropped question "
                            f"sentence(s); keeping {statements!r}"
                        )
                    if not statements:
                        continue
                    item = {**item, "text": statements}
                cleaned.append(item)
            extracted = cleaned

            # Filter out low-quality fragments that survived extraction —
            # too short to carry meaning on their own, or meta-cognitive
            # statements about the conversation rather than facts about the user.
            extracted = [item for item in extracted if is_fact_worth_storing(item["text"])]
            facts = [item["text"] for item in extracted]
            fact_metadata = [item["metadata"] for item in extracted]
            fact_fallback = [item["fallback"] for item in extracted]

            if not facts:
                logger.info(
                    f"All atomic facts filtered out "
                    f"for userId={body.userId} — nothing stored"
                )
                return MemoryStoreResponse(status="filtered", ids=[], facts_count=0)

            logger.info(
                f"Storing {len(facts)} facts after filtering "
                f"for userId={body.userId}: {facts}"
            )
        else:
            facts = [body.text]

        # Step 2: Embed every fact in a single batched model call — much
        # faster than calling embeddings.encode() once per fact, since the
        # model parallelises internally across the batch.
        embeddings_list = embeddings.encode_batch(facts)

        # Step 3 & 4: Store each fact as its own ChromaDB entry. ChromaDB's
        # client is sync, so run the N stores concurrently on a thread pool
        # instead of blocking on them one at a time.
        loop = asyncio.get_running_loop()
        store_tasks = [
            loop.run_in_executor(
                _store_executor,
                lambda f=fact, e=embedding: memory_store.store_memory(
                    userId=body.userId,
                    sessionId=body.sessionId or "",
                    text=f,
                    role=body.role,
                    embedding=e,
                ),
            )
            for fact, embedding in zip(facts, embeddings_list)
        ]
        stored_ids = await asyncio.gather(*store_tasks)

        # Graph processing is scheduled after storage completes. Each
        # background task is handed the SAME id store_memory() just
        # returned for that fact — not a freshly synthesized one — so the
        # graph node this creates stays in sync with its ChromaDB document
        # (the contract process_memory_intent()'s memoryId param and
        # graph_routes.py's integration guide both assume). This also lets
        # the background task patch that exact ChromaDB document's metadata
        # once domain is known (see process_graph_background).
        if body.role == "user":
            for index, fact in enumerate(facts):
                background_tasks.add_task(
                    process_graph_background,
                    body.userId,
                    fact,
                    stored_ids[index],
                    fact_metadata[index],
                    fact_fallback[index],
                )

        # Step 5: Return success response with every generated ID.
        logger.info(
            f"Memory stored  userId={body.userId}  facts_count={len(stored_ids)}  ids={stored_ids}"
        )
        return MemoryStoreResponse(status="stored", ids=stored_ids, facts_count=len(stored_ids))

    except Exception as e:
        # Step 6: Return 503 if anything fails (ChromaDB down, model error, etc.)
        # 503 = "Service Unavailable" — tells the caller to retry later.
        logger.error(f"store_memory failed: {e}")
        raise HTTPException(status_code=503, detail=f"Memory store unavailable: {str(e)}")


# ================================================================
# ENDPOINT 2: POST /memory/retrieve
# Finds the most semantically similar memories for a user's query.
# ================================================================
@app.post("/memory/retrieve", response_model=MemoryRetrieveResponse)
async def retrieve_memory_endpoint(
    body: MemoryRetrieveRequest,
    authorization: Optional[str] = Header(default=None),
):
    # Auth: reject any request without a valid Bearer token
    require_auth(authorization)

    logger.info(f"retrieve_memory  userId={body.userId}  topK={body.topK}")
    scheduler.mark_user_active(body.userId)

    try:
        # Step 1: Take the query string from the request body.
        # This is the text we'll search for semantically similar memories of.
        query = body.query

        # Step 2: Convert the query into a vector.
        # We need the same vector space as stored memories so that
        # "close vectors = similar meaning" comparisons make sense.
        query_embedding = embeddings.encode(query)

        # Step 2b: Classify which domain (if any) this query is about, using
        # the same embedding-centroid classifier extract_graph_metadata()
        # uses at write time — never the LLM fallback, since that would add
        # LLM latency to every retrieval call. A low-confidence/no-match
        # result (None) means "search across all domains", same as before
        # domain filtering existed.
        from graph_memory import classify_domain_by_embedding
        query_domain = classify_domain_by_embedding(query, body.userId)

        # Step 3 & 4: Search ChromaDB for the most similar memories.
        # The where={"userId": ...} filter inside retrieve_memories() ensures
        # we ONLY return memories belonging to this specific user.
        memories = memory_store.retrieve_memories(
            userId=body.userId,
            query_embedding=query_embedding,
            topK=body.topK,
            domain=query_domain,
        )

        # Step 5 & 6: Return the memories list.
        # An empty list is a valid result (user has no memories yet) — not an error.
        logger.info(f"Retrieved {len(memories)} memories for userId={body.userId}")
        logger.info(f"DEBUG retrieved_memories userId={body.userId} texts={memories!r}")
        return MemoryRetrieveResponse(memories=memories)

    except Exception as e:
        # Step 7: Return 503 if ChromaDB or embedding fails.
        logger.error(f"retrieve_memory failed: {e}")
        raise HTTPException(status_code=503, detail=f"Memory retrieval unavailable: {str(e)}")


# ================================================================
# ENDPOINT: GET /memory/all/{userId}
# Returns every memory stored for a user, ordered oldest-first.
# ================================================================
@app.get("/memory/all/{userId}")
async def get_all_memories_endpoint(
    userId: str,
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    logger.info(f"get_all_memories  userId={userId}")
    try:
        memories = memory_store.get_all_memories(userId)
        return {"userId": userId, "count": len(memories), "memories": memories}
    except Exception as e:
        logger.error(f"get_all_memories failed: {e}")
        raise HTTPException(status_code=503, detail=f"Memory retrieval unavailable: {str(e)}")


# ================================================================
# ENDPOINT: DELETE /memory/{userId}
# Clears one user's data from BOTH stores (ChromaDB + graph DB) in
# place, without touching any other user's data or requiring a
# server restart — the safe alternative to deleting chroma_data by
# hand while the server is running.
# ================================================================
@app.delete("/memory/{userId}")
async def clear_memory_endpoint(
    userId: str,
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    logger.info(f"clear_memory  userId={userId}")

    try:
        vector_deleted = memory_store.delete_user_memories(userId)

        from graph_memory import delete_user_graph
        graph_result = delete_user_graph(userId)
        graph_deleted = graph_result["nodes_deleted"]

        logger.info(
            f"clear_memory complete  userId={userId}  "
            f"vectorMemoriesDeleted={vector_deleted}  "
            f"graphNodesDeleted={graph_deleted}  "
            f"graphEdgesDeleted={graph_result['edges_deleted']}"
        )

        return {
            "userId": userId,
            "vectorMemoriesDeleted": vector_deleted,
            "graphNodesDeleted": graph_deleted,
        }

    except Exception as e:
        logger.error(f"clear_memory failed userId={userId}: {e}")
        raise HTTPException(status_code=503, detail=f"Clearing memory failed: {str(e)}")


async def process_graph_background(
    userId: str,
    text: str,
    memory_id: str,
    extracted_metadata: Optional[dict] = None,
    extraction_fallback: bool = False,
) -> None:
    """
    Runs after the /memory/store response is already sent to the caller, once
    per atomic fact. Writes the fact to the graph using the metadata the
    extractor already produced for it (extracted_metadata), or classifies it
    here if that's None. Failure here never affects the user.
    """
    try:
        logger.info(
            f"Background graph processing started "
            f"userId={userId} memoryId={memory_id}"
        )
        from graph_memory import (
            process_memory_intent,
            resolve_conflicting_nodes,
            CONFLICT_RESOLUTION_TRIGGER_LIMIT,
        )

        result = process_memory_intent(
            userId, text, memory_id, extracted_metadata, extraction_fallback
        )
        metadata = result["metadata"]

        # process_memory_intent() runs the is_declarative() gate itself
        # before ever calling the LLM — a question never gets extracted or
        # stored. Nothing left to do here for that case.
        if result["action"] == "skipped":
            logger.info(
                f"Background graph processing skipped "
                f"userId={userId} reason=non-declarative text"
            )
            return

        # Patch the domain this extraction just resolved onto the ChromaDB
        # entry for THIS memory_id — not result["node_id"], which for
        # action="updated" is an older, matched graph node, not the fact we
        # just stored (for "created"/"superseded" the two are the same id). This is what makes the memory domain-filterable in
        # retrieve_memories(); it wasn't known yet when store_memory() wrote
        # it, since that call happens before this background task runs.
        domain = metadata.get("domain", "general")
        memory_store.update_memory_domain(memory_id, domain)

        # A revision: this fact superseded an older value of the same slot,
        # whose graph node is now archived as history. Flag the old value's
        # Chroma entry the same way (kept, not deleted), so /memory/retrieve
        # stops returning it as current context.
        if result["action"] == "superseded":
            old = result["supersession"]["old"]
            memory_store.mark_memory_superseded(userId, old["node_id"], memory_id)
            logger.info(
                f"Fact superseded userId={userId} "
                f"old={old['entity_name'] or old['text']!r} -> "
                f"new={result['supersession']['new']['entity_name']!r}"
            )

        # We're already running off the main request/response path (this
        # function only runs after the chat response has been sent), so a
        # direct await here adds no latency the user can see. Only "created"
        # can have just introduced a new duplicate — "updated" and
        # "superseded" already went through the write-time match.
        # A degraded (fallback-metadata) node never enters the merge pass:
        # clustering unreliable data could archive real nodes into it.
        if result["action"] == "created" and not result.get("degraded"):
            await resolve_conflicting_nodes(
                userId,
                metadata.get("domain", "general"),
                CONFLICT_RESOLUTION_TRIGGER_LIMIT,
            )

        logger.info(
            f"Background graph processing complete "
            f"userId={userId} action={result['action']} "
            f"domain={metadata.get('domain', 'unknown')} "
            f"status={metadata.get('status', 'unknown')}"
        )
    except Exception as e:
        logger.warning(
            f"Background graph processing failed "
            f"userId={userId} error={e} "
            f"— chat pipeline unaffected"
        )


# ================================================================
# ENDPOINT 3: POST /ai/chat
# Sends the user's message to Claude, injecting retrieved memories
# as context so the AI can give personalised, memory-aware responses.
# Graph processing for the user's message happens separately, per atomic
# fact, when /memory/store is called — not here, to avoid graph-processing
# the same statement twice (once whole, once split into facts).
# ================================================================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat_endpoint(
    body: AiChatRequest,
    authorization: Optional[str] = Header(default=None),
):
    # Auth: reject any request without a valid Bearer token
    require_auth(authorization)

    has_context = bool(body.context.strip())
    logger.info(
        f"ai_chat  userId={body.userId}  "
        f"context={'yes' if has_context else 'no'}"
    )
    scheduler.mark_user_active(body.userId)
    logger.info(f"DEBUG ai_chat_full_context userId={body.userId} context={body.context!r}")

    try:
        # Step 1 & 2: build_system_prompt() inside call_llm() checks whether
        # context is non-empty and builds the appropriate system prompt.

        # Step 3: Call the Anthropic Claude API with the message and context.
        # ai_service.call_llm() handles prompt construction and the API call.

        answer = ai_service.call_llm(
            message=body.message,
            context=body.context,
            session_history=[turn.model_dump() for turn in body.sessionHistory],
        )

        logger.info(f"ai_chat response generated for userId={body.userId}")

        return AiChatResponse(answer=answer)

    except Exception as e:
        # Step 6: Return 503 if the Anthropic API call fails.
        logger.error(f"ai_chat failed: {e}")
        raise HTTPException(status_code=503, detail="AI model unavailable")


# ================================================================
# ENDPOINT 4: GET /health
# No auth required — used by load balancers and monitoring tools
# to check whether this service and its dependencies are running.
# ================================================================
@app.get("/health", response_model=HealthResponse)
async def health_endpoint():
    # Check ChromaDB by asking for the document count.
    # If this raises, ChromaDB is down or corrupted.
    chromadb_status = "ok"
    try:
        memory_store._collection.count()
    except Exception as e:
        logger.warning(f"Health check: ChromaDB error: {e}")
        chromadb_status = "error"

    # Check the embedding model by encoding a trivial string.
    # If this raises, the model failed to load or is out of memory.
    embeddings_status = "ok"
    try:
        embeddings.encode("health check")
    except Exception as e:
        logger.warning(f"Health check: embeddings error: {e}")
        embeddings_status = "error"

    graph_status = "ok"
    try:
        from graph_memory import _get_connection, _release_connection
        conn = _get_connection()
        _release_connection(conn)
    except Exception as e:
        logger.warning(f"Health: graph DB error: {e}")
        graph_status = "error"

    overall = (
        "ok"
        if chromadb_status == "ok" and embeddings_status == "ok" and graph_status == "ok"
        else "error"
    )

    return HealthResponse(
        status=overall,
        chromadb=chromadb_status,
        embeddings=embeddings_status,
        llm=f"groq/{config.LLM_MODEL}",
        graph=graph_status,
    )


# ---------------------------------------------------------------
# Run the server when this file is executed directly:
#   python main.py
#
# In production you'd typically run:
#   uvicorn main:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT, reload=False)
