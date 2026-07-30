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

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException

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
    background_tasks: BackgroundTasks,
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
        # process_memory_intent() handles the whole pipeline: extracting
        # metadata via the LLM, then deciding whether this text completes or
        # duplicates an existing "ongoing" node (update) or is genuinely new
        # (create). This is what stops e.g. three separate nodes being
        # created for the same "deploy to AWS" task.
        result = graph_memory.process_memory_intent(
            userId=body.userId,
            text=body.text,
            memoryId=body.memoryId,
        )

        # The text was a question, not a fact/task — process_memory_intent()
        # never called the LLM or wrote anything. GraphProcessResponse's
        # fields are all required (non-Optional) strings/floats, so we fill
        # placeholders here rather than changing that shared response
        # contract; status="skipped" + an empty node_id is how Spring Boot
        # tells "nothing was stored" apart from a real "stored" response.
        if result["action"] == "skipped":
            logger.info(f"graph_process skipped  userId={body.userId}  reason=non-declarative text")
            return GraphProcessResponse(
                status="skipped",
                node_id="",
                entity_name="",
                domain="",
                sub_domain="",
                status_label="",
                timeline_label="",
                importance_score=0.0,
            )

        if not result.get("success"):
            raise HTTPException(
                status_code=503,
                detail="Graph store unavailable: failed to write node to disk.",
            )

        metadata = result["metadata"]
        node_id = result["node_id"]

        # A brand-new node was just written — schedule a cheap, domain-scoped
        # conflict-resolution pass as a background task so it never delays
        # this response. Only "created" needs this: "updated" already went
        # through the write-time dedup check, so it can't have just
        # introduced a new duplicate.
        if result["action"] == "created":
            background_tasks.add_task(
                graph_memory.resolve_conflicting_nodes,
                body.userId,
                metadata.get("domain", "general"),
                graph_memory.CONFLICT_RESOLUTION_TRIGGER_LIMIT,
            )

        logger.info(
            f"graph_process complete  userId={body.userId}  action={result['action']}  "
            f"nodeId={node_id}  entity={metadata.get('entity_name')}  domain={metadata.get('domain')}"
        )
        return GraphProcessResponse(
            status="stored",
            node_id=node_id,
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
# ENDPOINT 5: GET /graph/all
# Admin/debug endpoint — returns every memory node stored across
# ALL users, not scoped to a single userId like the endpoints above.
# ================================================================
@router.get("/all")
async def graph_all_nodes_endpoint(
    include_archived: bool = False,
    authorization: Optional[str] = Header(default=None),
):
    """
    Return every stored memory node, across all users.

    Admin/debug view only — the response includes every user's raw memory
    text in one payload, so it is gated behind the same Bearer token as
    every other /graph endpoint. Not intended for per-user frontend use;
    use GET /graph/timeline/{userId} or POST /graph/context for that.

    include_archived=true also includes nodes that were merged into
    another node by resolve_conflicting_nodes() (superseded_by IS NOT NULL).
    """
    _require_auth(authorization)
    logger.info(f"graph_all_nodes  include_archived={include_archived}")

    try:
        nodes = graph_memory.get_all_nodes(include_archived=include_archived)
        return {"count": len(nodes), "nodes": nodes}

    except Exception as e:
        logger.error(f"graph_all_nodes failed: {e}")
        raise HTTPException(status_code=503, detail=f"Fetching all nodes failed: {str(e)}")


# ================================================================
# ENDPOINT 6: DELETE /graph/{userId}
# Clears one user's graph data ONLY (nodes + edges), leaving their
# ChromaDB vector memories untouched. Use DELETE /memory/{userId}
# in main.py instead when both stores need to be cleared together.
# ================================================================
@router.delete("/{userId}")
async def graph_clear_endpoint(
    userId: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Delete every graph_nodes and graph_edges row for a single user.

    The safe, in-place alternative to running raw SQL or deleting files —
    no other user's data is touched and no server restart is needed.
    """
    _require_auth(authorization)
    logger.info(f"graph_clear  userId={userId}")

    try:
        result = graph_memory.delete_user_graph(userId)
        logger.info(
            f"graph_clear complete  userId={userId}  "
            f"nodesDeleted={result['nodes_deleted']}  edgesDeleted={result['edges_deleted']}"
        )
        return {
            "userId": userId,
            "graphNodesDeleted": result["nodes_deleted"],
            "graphEdgesDeleted": result["edges_deleted"],
        }

    except Exception as e:
        logger.error(f"graph_clear failed userId={userId}: {e}")
        raise HTTPException(status_code=503, detail=f"Graph clear failed: {str(e)}")


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
