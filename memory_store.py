# ============================================================
# FILE: memory_store.py
#
# WHAT THIS FILE DOES:
# Sets up the ChromaDB vector database and exposes two functions:
# store_memory() to save a memory, and retrieve_memories() to find the most
# semantically similar memories for a given user. All database operations
# live here — no other file touches ChromaDB directly.
#
# KEY CONCEPT FOR LEARNING:
# ChromaDB is a vector database. Unlike a regular database that matches
# exact values (WHERE name = 'Alice'), ChromaDB matches by meaning.
# You give it a vector (the embedding of your search query) and it returns
# the stored vectors that are closest to it — i.e. the most semantically
# similar pieces of text. This is what makes memory retrieval feel intelligent
# rather than keyword-based.
#
# A "collection" is ChromaDB's equivalent of a database table. It stores
# documents (text), their embeddings (vectors), IDs, and metadata together.
# We use one collection called "memories" for all users.
#
# Why persistent storage?
# Without persistence, ChromaDB stores everything in RAM and all memories
# are lost when the server restarts. PersistentClient writes to disk so
# memories survive restarts, deployments, and crashes.
#
# Why filter by userId?
# All users share one collection. Without a userId filter, user A's
# memories could appear in user B's responses — a serious privacy violation.
# The where={"userId": userId} clause ensures strict per-user isolation.
#
# WHERE IT FITS IN THE PIPELINE:
# embedding vector + metadata → memory_store.py → ChromaDB on disk (store)
# query vector + userId → memory_store.py → matching text strings (retrieve)
# ============================================================

import chromadb
import logging
import uuid
import time
from typing import List, Optional
import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Initialise the ChromaDB client ONCE at module load time.
#
# PersistentClient writes data to the path specified in config.
# If the directory doesn't exist, ChromaDB creates it automatically.
# If it already exists (server restart), ChromaDB reads existing data from it.
# ---------------------------------------------------------------
_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_PATH)

# ---------------------------------------------------------------
# Get or create the "memories" collection.
#
# get_or_create_collection is idempotent:
#   - First run: creates a new empty collection called "memories".
#   - Subsequent runs: returns the existing collection with all its data.
# This means we never accidentally wipe data on restart.
# ---------------------------------------------------------------
_collection = _client.get_or_create_collection(name="memories")

# ---------------------------------------------------------------
# Relevance threshold for retrieve_memories().
#
# IMPORTANT: this collection was created without an explicit
# `metadata={"hnsw:space": "cosine"}`, so Chroma is using its default
# distance space (squared L2), NOT cosine distance — despite "lower
# distance = more similar" still holding true either way. A cosine-style
# threshold like 0.7 does NOT apply here: it would reject every result,
# including strong matches. This value was calibrated empirically against
# the live collection — semantically related pairs measured ~1.4-1.8,
# unrelated pairs measured ~1.8-2.1. If the collection is ever recreated
# with cosine space and normalized embeddings, this threshold must be
# re-tuned (a cosine-appropriate value would be far smaller, e.g. ~0.7).
# ---------------------------------------------------------------
RELEVANCE_THRESHOLD = 1.8


def store_memory(
    userId: str,
    sessionId: str,
    text: str,
    role: str,
    embedding: List[float],
) -> str:
    """
    Save one memory (a single message) into ChromaDB.

    Args:
        userId:    The user this memory belongs to.
        sessionId: The conversation session this message came from.
        text:      The raw message text to store.
        role:      Who sent the message: "user" or "assistant".
        embedding: The pre-computed vector representation of the text.

    Returns:
        The unique ID assigned to this memory in ChromaDB.
    """

    # Step 1: Generate a unique ID for this memory.
    # We combine userId + timestamp + a short random hex suffix so the ID is:
    #   - Unique across all users and all time
    #   - Human-readable (you can tell who it belongs to and roughly when)
    #   - Safe to use as a ChromaDB document ID (no special characters)
    timestamp = time.time()
    memory_id = f"{userId}_{int(timestamp)}_{uuid.uuid4().hex[:8]}"

    # Step 2: Build the metadata dict.
    # Metadata is stored alongside the document in ChromaDB and can be used
    # for filtering (e.g. where={"userId": ...}) or for returning context
    # about a memory alongside its text.
    metadata = {
        "userId": userId,
        "sessionId": sessionId if sessionId else "",  # ChromaDB doesn't accept None in metadata
        "role": role,
        "timestamp": timestamp,  # Unix timestamp (float) — when this memory was stored
    }

    # Step 3: Add the document, its embedding, ID, and metadata to ChromaDB.
    # ChromaDB stores all four together so that when we search by vector,
    # we get back the text (documents) and context (metadatas) alongside it.
    _collection.add(
        documents=[text],       # The raw text — what we'll return to the caller
        embeddings=[embedding], # The vector — what ChromaDB searches against
        ids=[memory_id],        # The unique identifier for this entry
        metadatas=[metadata],   # Extra context stored with the document
    )

    # Step 4: Return the generated ID so the caller can confirm what was stored.
    return memory_id


def retrieve_memories(
    userId: str,
    query_embedding: List[float],
    topK: int,
    domain: Optional[str] = None,
) -> List[str]:
    """
    Find the most semantically similar memories for a given user.

    Searches ChromaDB for stored vectors that are closest (most similar) to
    query_embedding, filtered strictly to the given userId.

    Args:
        userId:          Only return memories belonging to this user.
        query_embedding: The vector of the current search query text.
        topK:            How many results to return (e.g. 5 = top 5 matches).
        domain:          If given, restrict results to memories tagged with
                          this domain (see graph_memory.classify_domain_by_embedding,
                          which the /memory/retrieve endpoint uses to classify
                          the query before calling this). Memories stored
                          before domain tagging existed, or whose background
                          graph processing hasn't patched a domain onto them
                          yet, won't match a domain filter — pass None
                          (the default) to search across all of a user's
                          memories regardless of domain, same as before this
                          parameter existed.

    Returns:
        A list of text strings (the stored memory documents), ordered from
        most to least similar. Returns an empty list if no memories exist yet.
    """

    # Step 1: Check how many documents are in the collection.
    # ChromaDB raises an error if you ask for more results than exist.
    # We cap n_results at the actual count to prevent that error.
    total_docs = _collection.count()
    if total_docs == 0:
        # No memories stored at all — return empty list, not an error.
        return []

    # Request topK * 2 candidates (capped at the actual collection size) so
    # there's a pool to filter down by relevance below — otherwise, filtering
    # out weak matches would silently shrink results below topK even when
    # better matches exist further down the ranking.
    effective_n_results = min(topK * 2, total_docs)

    # Step 2: Query ChromaDB.
    # - query_embeddings: the vector we're searching for similar entries to
    # - n_results: how many candidates to fetch (before relevance filtering)
    # - where: metadata filter — ONLY return memories for this specific userId,
    #   and ONLY role="user" memories. Assistant responses are stored for
    #   future reference but are conversational text, not facts about the
    #   user, so they must never be injected back as retrieval context.
    #   The userId filter is the privacy boundary: no user ever sees
    #   another user's memories. domain is added only when the caller passed
    #   one — most memories don't have it backfilled/patched onto them yet,
    #   so an unconditional domain clause would silently exclude them.
    where_clause = {
        "$and": [
            {"userId": {"$eq": userId}},
            {"role": {"$eq": "user"}},
        ]
    }
    if domain:
        where_clause["$and"].append({"domain": {"$eq": domain}})

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_n_results,
        where=where_clause,
        include=["documents", "distances", "metadatas"],
    )

    # Step 3: Extract the document and distance arrays from ChromaDB's response.
    # ChromaDB returns a dict with keys: "ids", "documents", "metadatas", "distances".
    # Each is a list of lists (one list per query — we sent one query), so index [0]
    # gets the results for our single query, ordered most-to-least similar.
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)

    # Step 4: Filter out weak matches — a memory is only relevant context if
    # it's actually close to the query, not just the least-bad option available.
    # Superseded entries (an older value of a revised fact, see
    # mark_memory_superseded) are history, never current context. This is
    # filtered here rather than in `where`: a Chroma `$ne`/`$exists`-style
    # clause would also drop every entry that simply lacks the key, which is
    # all of them before this flag existed.
    filtered_docs = [
        doc for doc, dist, meta in zip(documents, distances, metadatas)
        if dist < RELEVANCE_THRESHOLD and not (meta or {}).get("superseded_by")
    ]

    logger.info(
        f"Retrieved {len(filtered_docs)} relevant memories from "
        f"{len(documents)} candidates userId={userId}"
    )

    # Step 5: Return at most topK of the filtered results.
    # If nothing passed the threshold (or the user has no memories), return empty list.
    return filtered_docs[:topK]


def update_memory_domain(memory_id: str, domain: str) -> bool:
    """
    Patch the `domain` field onto an already-stored ChromaDB entry.

    Domain isn't known at store_memory() time — it's determined afterward by
    the LLM/classifier-based graph extraction pipeline, which runs as a
    background task so it never blocks the /memory/store response. This is
    called once that background task resolves a domain, so the memory
    becomes domain-filterable in retrieve_memories() shortly after — not
    immediately — after it's stored.

    ChromaDB's collection.update() merges the given metadata keys into the
    existing metadata dict rather than replacing it wholesale, so userId/
    sessionId/role/timestamp are left untouched.

    Returns True if the update was applied, False on failure (never raises —
    this runs off the main request/response path via a background task, and
    a failure here must not surface as an error anywhere).
    """
    try:
        _collection.update(ids=[memory_id], metadatas=[{"domain": domain}])
        return True
    except Exception as e:
        logger.warning(f"update_memory_domain failed memoryId={memory_id}: {e}")
        return False


def mark_memory_superseded(userId: str, old_memory_id: str, new_memory_id: str) -> bool:
    """
    Flag the Chroma entry holding a fact's OLD value as superseded, after
    graph_memory.supersede_node() archived its graph node (a "revision").

    The entry is kept (history), but retrieve_memories() skips any entry
    with `superseded_by` set, so the stale value can't be returned as current
    context. Every graph node shares its id with the Chroma entry that
    created it, so old_memory_id is simply the archived node's id.

    Only entries owned by userId are touched. Returns True if the flag was
    written. Never raises — it runs in a background task.
    """
    try:
        existing = _collection.get(ids=[old_memory_id], include=["metadatas"])
        metas = existing.get("metadatas") or []
        if not metas or (metas[0] or {}).get("userId") != userId:
            logger.warning(
                f"mark_memory_superseded userId={userId} entry {old_memory_id} not found for this user"
            )
            return False
        _collection.update(ids=[old_memory_id], metadatas=[{"superseded_by": new_memory_id}])
        logger.info(
            f"mark_memory_superseded userId={userId} {old_memory_id} superseded_by={new_memory_id}"
        )
        return True
    except Exception as e:
        logger.warning(f"mark_memory_superseded failed userId={userId} id={old_memory_id}: {e}")
        return False


def delete_user_memories(userId: str) -> int:
    """
    Delete every memory belonging to a single user from ChromaDB.

    Unlike deleting the chroma_data folder directly, this runs against the
    live collection while the server keeps serving other users — no file
    locks, no restart, and every other user's data is untouched.

    Returns the number of memories deleted (0 if the user had none).
    """
    existing = _collection.get(where={"userId": userId}, include=[])
    ids = existing.get("ids") or []
    if not ids:
        return 0

    _collection.delete(where={"userId": userId})
    return len(ids)


def get_all_memories(userId: str) -> List[dict]:
    """
    Return every memory stored for a user, ordered oldest-first.

    Each item contains the memory text, role, sessionId, and timestamp
    so the caller gets full context, not just the text strings.
    """
    total_docs = _collection.count()
    if total_docs == 0:
        return []

    results = _collection.get(
        where={"userId": userId},
        include=["documents", "metadatas"],
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    memories = [
        {
            "text": doc,
            "role": meta.get("role", ""),
            "sessionId": meta.get("sessionId", ""),
            "timestamp": meta.get("timestamp", 0),
            "supersededBy": meta.get("superseded_by"),
        }
        for doc, meta in zip(documents, metadatas)
    ]

    memories.sort(key=lambda m: m["timestamp"])
    return memories
