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

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Header, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional

import config
import embeddings
import memory_store
import ai_service
import graph_routes
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

    yield  # Server is now running and accepting requests

    # --- SHUTDOWN ---
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

    # Pass the request to the actual endpoint handler
    response = await call_next(request)

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
    authorization: Optional[str] = Header(default=None),
):
    # Auth: reject any request without a valid Bearer token
    require_auth(authorization)

    # Log which user is storing a memory (useful for debugging)
    logger.info(f"store_memory  userId={body.userId}  role={body.role}")

    try:
        # Step 1: Take the text field from the request body.
        # This is the raw message content we want to remember.
        text = body.text

        # Step 2: Convert the text into a vector using the embedding model.
        # The vector captures the semantic meaning of the text so we can
        # search for similar memories later using vector similarity.
        embedding = embeddings.encode(text)

        # Step 3 & 4 & 5: Store the text, embedding, and metadata in ChromaDB.
        # memory_store.store_memory() generates the unique ID internally and
        # returns it so we can include it in the response.
        memory_id = memory_store.store_memory(
            userId=body.userId,
            sessionId=body.sessionId or "",
            text=text,
            role=body.role,
            embedding=embedding,
        )

        # Step 5: Return success response with the generated ID.
        logger.info(f"Memory stored  id={memory_id}")
        return MemoryStoreResponse(status="stored", id=memory_id)

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

    try:
        # Step 1: Take the query string from the request body.
        # This is the text we'll search for semantically similar memories of.
        query = body.query

        # Step 2: Convert the query into a vector.
        # We need the same vector space as stored memories so that
        # "close vectors = similar meaning" comparisons make sense.
        query_embedding = embeddings.encode(query)

        # Step 3 & 4: Search ChromaDB for the most similar memories.
        # The where={"userId": ...} filter inside retrieve_memories() ensures
        # we ONLY return memories belonging to this specific user.
        memories = memory_store.retrieve_memories(
            userId=body.userId,
            query_embedding=query_embedding,
            topK=body.topK,
        )

        # Step 5 & 6: Return the memories list.
        # An empty list is a valid result (user has no memories yet) — not an error.
        logger.info(f"Retrieved {len(memories)} memories for userId={body.userId}")
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


async def process_graph_background(
    userId: str,
    text: str,
    memory_id: str
) -> None:
    """
    Runs after the chat response is already sent to user.
    Extracts graph metadata and stores the node.
    Failure here never affects the user.
    """
    try:
        logger.info(
            f"Background graph processing started "
            f"userId={userId} memoryId={memory_id}"
        )
        from graph_memory import (
            extract_graph_metadata,
            add_to_graph
        )

        metadata = extract_graph_metadata(text, userId)

        add_to_graph(userId, memory_id, text, metadata)

        logger.info(
            f"Background graph processing complete "
            f"userId={userId} "
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
# ================================================================
@app.post("/ai/chat", response_model=AiChatResponse)
async def ai_chat_endpoint(
    body: AiChatRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    # Auth: reject any request without a valid Bearer token
    require_auth(authorization)

    has_context = bool(body.context.strip())
    logger.info(
        f"ai_chat  userId={body.userId}  "
        f"context={'yes' if has_context else 'no'}"
    )

    try:
        # Step 1 & 2: build_system_prompt() inside call_llm() checks whether
        # context is non-empty and builds the appropriate system prompt.

        # Step 3: Call the Anthropic Claude API with the message and context.
        # ai_service.call_llm() handles prompt construction and the API call.
        answer = ai_service.call_llm(
            message=body.message,
            context=body.context,
        )

        logger.info(f"ai_chat response generated for userId={body.userId}")

        memory_id = f"{body.userId}_{int(time.time())}"

        background_tasks.add_task(
            process_graph_background,
            body.userId,
            body.message,
            memory_id
        )

        logger.info(
            f"Graph processing scheduled as background task "
            f"memoryId={memory_id}"
        )

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
        from graph_memory import _get_connection
        conn = _get_connection()
        conn.close()
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
