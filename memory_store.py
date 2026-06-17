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
import uuid
import time
from typing import List
import config

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
) -> List[str]:
    """
    Find the most semantically similar memories for a given user.

    Searches ChromaDB for stored vectors that are closest (most similar) to
    query_embedding, filtered strictly to the given userId.

    Args:
        userId:          Only return memories belonging to this user.
        query_embedding: The vector of the current search query text.
        topK:            How many results to return (e.g. 5 = top 5 matches).

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

    # Cap topK so we never ask for more results than exist in the collection.
    # e.g. if only 2 memories are stored and topK=5, ask for 2.
    effective_top_k = min(topK, total_docs)

    # Step 2: Query ChromaDB.
    # - query_embeddings: the vector we're searching for similar entries to
    # - n_results: how many matches to return
    # - where: metadata filter — ONLY return memories for this specific userId
    #   This is the privacy boundary: no user ever sees another user's memories.
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_top_k,
        where={"userId": userId},
    )

    # Step 3: Extract the document strings from ChromaDB's response.
    # ChromaDB returns a dict with keys: "ids", "documents", "metadatas", "distances".
    # results["documents"] is a list of lists (one list per query — we sent one query).
    # So results["documents"][0] is the list of matching text strings for our query.
    documents = results.get("documents", [[]])[0]

    # Step 4: Return the list of text strings.
    # If the query matched nothing (e.g. userId has no memories), return empty list.
    return documents if documents else []
