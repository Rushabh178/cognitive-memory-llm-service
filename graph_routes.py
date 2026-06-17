# ============================================================
# FILE: graph_routes.py
#
# WHAT THIS FILE DOES:
# Defines all FastAPI endpoints for the graph memory layer as a
# reusable APIRouter. main.py imports this router and mounts it
# at the /graph prefix, so all endpoints here are accessible at
# /graph/process, /graph/context, etc.
#
# KEY CONCEPT FOR LEARNING:
# FastAPI's APIRouter lets you split endpoint definitions across
# multiple files instead of cramming everything into main.py.
# Think of it like Express.js routers or Spring Boot @RestController
# classes — each file owns a slice of the API surface. main.py
# just registers the slices with app.include_router().
#
# WHERE IT FITS IN THE PIPELINE:
# HTTP request → main.py routes → graph_routes.py handler
# → graph_memory.py functions → JSON response
# ============================================================

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

import config
import graph_memory
from models import (
    GraphProcessRequest,
    GraphProcessResponse,
    GraphContextRequest,
    GraphContextResponse,
    GraphStatusUpdateRequest,
    GraphStatusUpdateResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Create the router.
# main.py calls: app.include_router(graph_routes.router, prefix="/graph")
# All route paths below are therefore relative to /graph.
# ---------------------------------------------------------------
router = APIRouter()


# ---------------------------------------------------------------
# Auth helper — duplicated from main.py to avoid a circular import.
# (main.py imports graph_routes; if graph_routes imported from main.py
# that would be a circular dependency and Python would refuse to load.)
# ---------------------------------------------------------------
def _require_auth(authorization: Optional[str]) -> None:
    """Validate the Bearer token. Raises HTTP 401 if invalid or missing."""
    expected = f"Bearer {config.API_BEARER_TOKEN}"
    if not authorization or authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Unauthorised: missing or invalid Bearer token.",
        )


# ================================================================
# ENDPOINT 1: POST /graph/process
# Takes a raw memory text, extracts graph metadata via Claude,
# and adds the node (with edges) to the user's graph.
# ================================================================
@router.post("/process", response_model=GraphProcessResponse)
async def graph_process_endpoint(
    body: GraphProcessRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Process a raw memory text into a graph node.

    Intended to be called by Spring Boot immediately after POST /memory/store,
    passing the same text and the memory ID returned by that endpoint.
    The memory ID becomes the graph node ID so the two stores stay in sync.
    """
    _require_auth(authorization)
    logger.info(f"graph_process  userId={body.userId}  memoryId={body.memoryId}")

    try:
        # Step 1: Ask Claude to extract entity name, domain, status, etc.
        # This is the intelligence step — turning raw text into structured metadata.
        metadata = graph_memory.extract_graph_metadata(body.text, body.userId)

        # Step 2: Add the node (and auto-generated edges) to the graph.
        # We use the memoryId as the node_id so ChromaDB and the graph stay in sync.
        success = graph_memory.add_to_graph(
            userId=body.userId,
            node_id=body.memoryId,
            text=body.text,
            metadata=metadata,
        )

        if not success:
            raise HTTPException(
                status_code=503,
                detail="Graph store unavailable: failed to write node to disk.",
            )

        # Step 3: Return the structured metadata so Spring Boot can log or display it.
        logger.info(
            f"graph_process complete  userId={body.userId}  "
            f"entity={metadata.get('entity_name')}  domain={metadata.get('domain')}"
        )
        return GraphProcessResponse(
            status="stored",
            node_id=body.memoryId,
            entity_name=metadata.get("entity_name", ""),
            domain=metadata.get("domain", "general"),
            sub_domain=metadata.get("sub_domain", ""),
            status_label=metadata.get("status", "ongoing"),
            timeline_label=metadata.get("timeline_label", ""),
            importance_score=metadata.get("importance_score", 0.5),
        )

    except HTTPException:
        raise  # re-raise auth / 503 errors unchanged
    except Exception as e:
        logger.error(f"graph_process failed: {e}")
        raise HTTPException(status_code=503, detail=f"Graph processing failed: {str(e)}")


# ================================================================
# ENDPOINT 2: POST /graph/context
# Retrieves the most relevant graph nodes for a query and returns
# enriched context text + structured node data.
# ================================================================
@router.post("/context", response_model=GraphContextResponse)
async def graph_context_endpoint(
    body: GraphContextRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Retrieve graph-based context for a user's query.

    Spring Boot should call this alongside POST /memory/retrieve,
    then combine both results into a single context string for POST /ai/chat.
    The graph context adds relationship-aware context that vector search alone
    cannot provide (e.g. "DSA practice is PREPARATION_FOR Amazon interview").
    """
    _require_auth(authorization)
    logger.info(f"graph_context  userId={body.userId}  topN={body.topN}")

    try:
        # get_graph_context() handles all scoring, traversal, and text building.
        # It never raises — returns empty result on any failure.
        result = graph_memory.get_graph_context(
            userId=body.userId,
            query_text=body.query,
            topN=body.topN,
        )

        logger.info(
            f"graph_context  userId={body.userId}  "
            f"nodes={len(result['nodes'])}  neighbors={len(result['neighbors'])}"
        )
        return GraphContextResponse(**result)

    except Exception as e:
        logger.error(f"graph_context failed: {e}")
        raise HTTPException(status_code=503, detail=f"Graph context unavailable: {str(e)}")


# ================================================================
# ENDPOINT 3: GET /graph/timeline/{userId}
# Returns all memories grouped by month → domain → status.
# ================================================================
@router.get("/timeline/{userId}")
async def graph_timeline_endpoint(
    userId: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Return the full memory timeline for a user.

    Groups all stored graph nodes by month, then by domain (job, study, etc.),
    then separates ongoing vs completed within each domain.
    Useful for a "memory history" view in the frontend.
    """
    _require_auth(authorization)
    logger.info(f"graph_timeline  userId={userId}")

    try:
        summary = graph_memory.get_timeline_summary(userId)
        return {"userId": userId, "timeline": summary}

    except Exception as e:
        logger.error(f"graph_timeline failed: {e}")
        raise HTTPException(status_code=503, detail=f"Timeline unavailable: {str(e)}")


# ================================================================
# ENDPOINT 4: POST /graph/status-update
# Updates a node's status to "ongoing" or "completed".
# ================================================================
@router.post("/status-update", response_model=GraphStatusUpdateResponse)
async def graph_status_update_endpoint(
    body: GraphStatusUpdateRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Update the status of a graph node.

    Call this when the user reports a change — e.g. "I finished my DSA course"
    should mark the DSA course node from "ongoing" to "completed".
    Spring Boot can pair this with identifying the relevant nodeId from
    prior /graph/context responses.
    """
    _require_auth(authorization)

    # Step 1: Validate newStatus before touching the graph.
    # Only "ongoing" and "completed" are valid — reject anything else with 400.
    if body.newStatus not in ("ongoing", "completed"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{body.newStatus}'. Must be 'ongoing' or 'completed'.",
        )

    logger.info(
        f"graph_status_update  userId={body.userId}  "
        f"nodeId={body.nodeId}  newStatus={body.newStatus}"
    )

    try:
        updated = graph_memory.update_node_status(
            userId=body.userId,
            node_id=body.nodeId,
            new_status=body.newStatus,
        )

        return GraphStatusUpdateResponse(
            updated=updated,
            nodeId=body.nodeId,
            newStatus=body.newStatus,
        )

    except Exception as e:
        logger.error(f"graph_status_update failed: {e}")
        raise HTTPException(status_code=503, detail=f"Status update failed: {str(e)}")


# ================================================================
# INTEGRATION GUIDE FOR SPRING BOOT
# ================================================================
#
# This comment block explains the recommended call sequence so that
# any Spring Boot developer reading this file knows exactly how to
# integrate both the vector memory layer and the graph memory layer.
#
# ---------------------------------------------------------------
# STEP 1 — Store a memory (call both stores)
# ---------------------------------------------------------------
# When the user sends a message, call POST /memory/store first.
# It returns a memory ID (e.g. "user_123_1749123456_a1b2c3d4").
# Immediately after, call POST /graph/process with the SAME text
# and that memory ID. This creates a graph node linked to the
# ChromaDB document so both stores stay in sync.
#
#   POST /memory/store   → { "id": "user_123_..." }
#   POST /graph/process  → { "entity_name": "...", "domain": "job", ... }
#
# ---------------------------------------------------------------
# STEP 2 — Retrieve context (combine both retrievers)
# ---------------------------------------------------------------
# When building context for the AI response, call both retrievers
# in parallel (they are independent HTTP calls):
#
#   POST /memory/retrieve  → { "memories": ["text1", "text2", ...] }
#   POST /graph/context    → { "graph_context_text": "...", ... }
#
# Combine them:
#   vector_context  = memories joined with "\n"
#   graph_context   = graph_context_text
#   final_context   = vector_context + "\n\n" + graph_context
#
# ---------------------------------------------------------------
# STEP 3 — Generate AI response
# ---------------------------------------------------------------
# Pass the combined context to the chat endpoint:
#
#   POST /ai/chat  { message: "...", context: final_context }
#                → { "answer": "Claude's memory-aware response" }
#
# This gives Claude:
#   - Semantically similar memories (from ChromaDB vector search)
#   - Structured relationship context (from graph traversal)
#   - A timeline summary and status overview (ongoing vs completed)
#
# ---------------------------------------------------------------
# OPTIONAL — Update status when user reports completion
# ---------------------------------------------------------------
# If the user says "I passed my interview", identify the relevant
# nodeId from a previous /graph/context call and call:
#
#   POST /graph/status-update  { nodeId: "...", newStatus: "completed" }
#
# ================================================================
