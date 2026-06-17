# ============================================================
# FILE: models.py
#
# WHAT THIS FILE DOES:
# Defines all Pydantic models (data shapes) for every request and response
# in the API. Pydantic automatically validates incoming JSON — if the Spring
# Boot service sends the wrong type, FastAPI returns a 422 error immediately,
# before any of our code runs.
#
# KEY CONCEPT FOR LEARNING:
# Pydantic models are like contracts. They say "this endpoint expects exactly
# this shape of data." If the caller sends a number where a string is expected,
# or forgets a required field, the error is caught at the door — not buried
# somewhere in business logic. This makes debugging much easier.
#
# WHERE IT FITS IN THE PIPELINE:
# Incoming JSON → Pydantic validates & parses → typed Python objects → endpoint logic
# ============================================================

from pydantic import BaseModel
from typing import Dict, List, Optional


class MemoryStoreRequest(BaseModel):
    """
    Data the caller sends when storing a new memory.
    Represents one message (user or assistant) that should be remembered.
    """

    userId: str          # Identifies whose memory this belongs to
    sessionId: Optional[str] = None  # Groups memories from the same conversation session
    text: str            # The actual message content to store
    role: str            # Who said it: "user" or "assistant"


class MemoryStoreResponse(BaseModel):
    """
    Data returned after a memory is successfully stored.
    Lets the caller confirm the store happened and track the generated ID.
    """

    status: str          # Always "stored" on success
    id: str              # The unique ID assigned to this memory in ChromaDB


class MemoryRetrieveRequest(BaseModel):
    """
    Data the caller sends when asking for relevant memories.
    The query is semantically searched — not exact-matched — against stored memories.
    """

    userId: str          # Only retrieve memories belonging to this user
    query: str           # The text to search for semantically similar memories
    topK: int            # How many memories to return (e.g. 5 returns the 5 most similar)


class MemoryRetrieveResponse(BaseModel):
    """
    Data returned after a memory search.
    Returns a list of the most semantically similar stored text snippets.
    """

    memories: List[str]  # The actual text of the matching memories, ordered by similarity


class AiChatRequest(BaseModel):
    """
    Data the caller sends when asking the AI to respond to a user message.
    The context field contains pre-retrieved memories that will be injected
    into the AI's system prompt.
    """

    userId: str          # Identifies the user (used for logging, not for retrieval here)
    message: str         # The user's current message to respond to
    context: str         # Memories already retrieved and joined into a single string


class AiChatResponse(BaseModel):
    """
    Data returned after the AI generates a response.
    """

    answer: str          # The AI's text response to the user's message


class HealthResponse(BaseModel):
    """
    Data returned by the health check endpoint.
    Used by load balancers, monitoring tools, and the Spring Boot service
    to verify that this service and all its dependencies are running.
    """

    status: str          # Overall health: "ok" or "error"
    chromadb: str        # ChromaDB connectivity: "ok" or "error"
    embeddings: str      # Embedding model status: "ok" or "error"
    llm: str             # Which LLM is configured (not a live check, just the model name)
    graph: str           # Graph data directory accessibility: "ok" or "error"


# ================================================================
# GRAPH MEMORY — REQUEST AND RESPONSE MODELS
# ================================================================

class GraphProcessRequest(BaseModel):
    """
    Data the caller sends when adding a memory to the graph layer.
    Should be called right after POST /memory/store using the same
    text and the memory ID returned by that endpoint.
    """

    userId: str      # Owner of this memory
    text: str        # The raw memory text (same as sent to /memory/store)
    memoryId: str    # The ID returned by /memory/store — becomes the graph node ID


class GraphProcessResponse(BaseModel):
    """
    Data returned after a memory is processed into the graph.
    Contains the structured metadata Claude extracted from the text.
    """

    status: str              # Always "stored" on success
    node_id: str             # The node ID in the graph (same as ChromaDB memory ID)
    entity_name: str         # Main entity Claude identified (e.g. "Amazon Interview")
    domain: str              # Life domain: job, study, sport, etc.
    sub_domain: str          # Specific sub-category (e.g. "DSA preparation")
    status_label: str        # "ongoing" or "completed"
    timeline_label: str      # Human-readable date label (e.g. "June 2026")
    importance_score: float  # 0.0 to 1.0 — how significant this memory is


class GraphContextRequest(BaseModel):
    """
    Data the caller sends when requesting graph-based context for a query.
    Pair this call with POST /memory/retrieve to get both vector and graph context.
    """

    userId: str       # Only retrieve memories belonging to this user
    query: str        # The user's current message or question
    topN: int = 5     # How many top-scored nodes to return (default 5)


class GraphNodeInfo(BaseModel):
    """
    Represents a single graph node in a context response.
    """

    node_id: str             # Unique node identifier
    text: str                # Original memory text
    entity_name: str         # Main entity name extracted by Claude
    domain: str              # Life domain
    sub_domain: str          # Specific sub-category
    status: str              # "ongoing" or "completed"
    timeline_label: str      # e.g. "June 2026"
    importance_score: float  # 0.0 to 1.0


class GraphTimelineSummary(BaseModel):
    """
    Summary statistics for the user's memory graph.
    Gives Claude a quick overview of the user's current life state.
    """

    this_month: int            # Number of memories stored this calendar month
    ongoing: int               # Total memories with status "ongoing"
    completed: int             # Total memories with status "completed"
    domains: Dict[str, int]    # Count per domain, e.g. {"job": 3, "study": 2}


class GraphContextResponse(BaseModel):
    """
    Full response from the graph context endpoint.
    Contains top-scored nodes, their neighbors (graph traversal result),
    summary statistics, and a pre-formatted text string for Claude.
    """

    nodes: List[GraphNodeInfo]            # Top-N nodes by relevance score
    neighbors: List[GraphNodeInfo]        # Nodes connected to top nodes by edges
    timeline_summary: GraphTimelineSummary
    graph_context_text: str               # Pre-formatted string ready to inject into Claude


class GraphStatusUpdateRequest(BaseModel):
    """
    Data the caller sends when updating a memory node's status.
    Use this when the user reports that something has finished.
    """

    userId: str      # Owner of the graph
    nodeId: str      # The node ID to update
    newStatus: str   # Must be exactly "ongoing" or "completed"


class GraphStatusUpdateResponse(BaseModel):
    """
    Data returned after a status update attempt.
    """

    updated: bool    # True if the node was found and updated
    nodeId: str      # The node ID that was targeted
    newStatus: str   # The status value that was applied
