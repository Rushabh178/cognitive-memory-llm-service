# ============================================================
# FILE: graph_memory.py
#
# WHAT THIS FILE DOES:
# Stores user memories as a knowledge graph in PostgreSQL.
# Each memory becomes a node with structured metadata (domain,
# status, timeline, importance). Related memories are connected
# by typed edges (RELATED_TO, LED_TO). Each user's data is
# isolated via a user_id column — no data leaks between users.
#
# KEY CONCEPT FOR LEARNING:
# Migrating from file storage to a database is a common
# engineering task. The function signatures stay identical
# so callers (graph_routes.py) don't know or care what changed
# internally. This is called "encapsulation" — hide the
# implementation, keep the interface stable.
#
# WHERE IT FITS IN THE PIPELINE:
# chat message → extract_graph_metadata() →
# add_to_graph() → PostgreSQL (graph_nodes / graph_edges tables)
# ============================================================

import json
import re
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

import config
import ai_service
import embeddings
from ai_service import _client as _groq_client

logger = logging.getLogger(__name__)

# Minimum combined relevance score (importance*0.4 + recency*0.3 + keyword*0.3)
# a node must clear to be included in get_graph_context() results. Tune this
# if unrelated context is still leaking into prompts (raise it) or genuinely
# relevant memories are being dropped (lower it).
RELEVANCE_THRESHOLD = 0.45

# Minimum LLM-judged confidence that a candidate domain is the same life-area
# as an existing domain, for resolve_domain() to reuse the existing one
# instead of minting a new one (e.g. "job_search" should resolve to "career"
# if "career" already exists).
DOMAIN_MATCH_THRESHOLD = 0.80

# Minimum cosine similarity between a new memory's embedding and an existing
# domain's centroid embedding for classify_domain_by_embedding() to use that
# domain directly, skipping resolve_domain() (and its LLM call) entirely.
#
# CALIBRATION STATUS: an early smoke test using hand-written, first-person
# test sentences ("I have an exam next week...") scored suspiciously low and
# even inverted (same-domain lower than different-domain) — but that test
# fed the classifier text in a different style from what it ever sees in
# production. Every stored memory is rewritten by atomic_extractor.py into a
# short, third-person fact ("He works at Yardi") BEFORE extract_graph_metadata()
# ever runs, so that's the only distribution classify_domain_by_embedding()
# needs to work well on. Re-tested against the 2026-07-30 backfill
# (backfill_legacy_domains.py) using real third-person atomic facts, 18/33
# legacy memories cleared 0.70 confidently (many at 0.79-1.00). Caveat: most
# of those 18 already had an identical/near-identical text already filed
# under that domain in graph_nodes (graph processing had already run for
# them despite the id-mismatch bug fixed alongside this), which likely
# inflates those specific scores via self-similarity — that inflation can't
# happen for genuinely new text at normal write time, but it also means this
# backfill run isn't strong accuracy evidence either way. A real, still-open
# risk: classify_domain_by_embedding() is also used at QUERY time (see
# /memory/retrieve in main.py) on raw natural-language queries, which are
# NOT third-person atomic facts — a train/inference distribution mismatch
# that write-time classification doesn't have. Keep logging every call's
# top score (see classify_domain_by_embedding()) and watch real query-time
# scores specifically before trusting this threshold for retrieval.
DOMAIN_CLASSIFIER_THRESHOLD = 0.70

# Minimum LLM-judged confidence that new text is about the same logical task
# as an existing "ongoing" node, for find_matching_task_node() to treat them
# as the same thing rather than creating a duplicate node. Raised from 0.75
# to 0.85 after a production false-positive: "update my resume before
# Friday" matched "Rushabh is a software engineer at Yardi" at confidence
# 1.00 — both were domain=job, but not the same task. See the prompt in
# find_matching_task_node() for the domain-vs-task distinction this guards.
TASK_MATCH_THRESHOLD = 0.85

# Per-domain node cap used by the cheap, per-write resolve_conflicting_nodes()
# trigger (see process_memory_intent() call sites in graph_routes.py and
# main.py). Keeps that LLM clustering call bounded as a domain's history
# grows. A full sweep (limit=None) is needed to catch conflicts between very
# old dormant nodes and brand-new ones — see resolve_conflicting_nodes()'s
# docstring for that known limitation.
CONFLICT_RESOLUTION_TRIGGER_LIMIT = 20

# NOTE on matching approach (resolve_domain / find_matching_task_node):
# An earlier version of both functions used raw sentence-embedding cosine
# similarity (via embeddings.encode()). Empirically that scored the exact
# cases these thresholds are meant to catch far too low to ever fire — e.g.
# "job_search" vs "career" scored 0.39, and an obvious task/completion pair
# ("applying for a job, preparing for the interview" vs "got the offer,
# finished the interview") scored 0.71 — both under their thresholds. LLM
# judgment via ai_service handles this kind of reasoning-based matching
# (paraphrase, completion-of-the-same-task) far better than raw embedding
# distance, so both functions ask the LLM directly instead. The same reasoning
# applies to _cluster_nodes_via_llm() below (which resolve_conflicting_nodes()
# depends on) — it's clustering by the same kind of paraphrase/completion
# judgment, not content similarity.
#
# classify_domain_by_embedding() below is NOT a violation of this finding: it
# embeds MEMORY TEXT and compares to other memory text (content similarity),
# never a domain LABEL string compared to another label string. It only
# replaces resolve_domain()'s LLM call as a pre-check inside
# extract_graph_metadata() — resolve_domain(), find_matching_task_node(), and
# _cluster_nodes_via_llm() themselves are untouched and still LLM-only.


# ================================================================
# DATABASE CONNECTION
# ================================================================

_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    """
    Lazily create the module-level connection pool on first use.

    Not created at import time because DB_* config (and the DB itself)
    may not be ready yet when this module is first imported.
    """
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            cursor_factory=RealDictCursor,
        )
        logger.info("PostgreSQL connection pool created (min=2 max=10)")
    return _pool


def _get_connection():
    """
    Borrow a connection from the pool.

    Every caller MUST return it via _release_connection() in a finally
    block — the pool has a fixed max of 10, so a leaked connection
    permanently shrinks capacity until the process restarts.

    Falls back to a direct, unpooled connection if the pool is exhausted
    or unavailable (e.g. DB down when the pool was first created) — a
    request should degrade to per-call connection overhead rather than
    fail outright.
    """
    try:
        return _get_pool().getconn()
    except Exception as e:
        logger.warning(
            f"Connection pool exhausted/unavailable, opening a direct "
            f"connection instead: {e}"
        )
        return psycopg2.connect(
            host=config.DB_HOST,
            port=config.DB_PORT,
            dbname=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD,
            cursor_factory=RealDictCursor,
        )


def _release_connection(conn) -> None:
    """
    Return a connection to the pool.

    conn may be a direct fallback connection (not tracked by the pool) if
    _get_connection() fell back above — putconn() raises for those, so we
    catch that and just close the connection directly instead.
    """
    try:
        _get_pool().putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


# ================================================================
# TABLE INITIALISATION
# ================================================================

def initialize_tables() -> None:
    """
    Create graph_nodes and graph_edges tables if they don't exist.
    Called once at startup from main.py lifespan.
    Safe to call multiple times — all DDL uses IF NOT EXISTS.
    """
    ddl = [
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id              VARCHAR PRIMARY KEY,
            user_id         VARCHAR NOT NULL,
            entity_name     VARCHAR,
            domain          VARCHAR DEFAULT 'general',
            sub_domain      VARCHAR,
            status          VARCHAR DEFAULT 'ongoing',
            timeline_year   INTEGER,
            timeline_month  INTEGER,
            timeline_label  VARCHAR,
            importance_score FLOAT DEFAULT 0.5,
            text            TEXT,
            created_at      TIMESTAMP DEFAULT NOW(),
            last_updated    TIMESTAMP DEFAULT NOW(),
            last_accessed_at TIMESTAMP DEFAULT NOW(),
            superseded_by   VARCHAR REFERENCES graph_nodes(id),
            archived_at     TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS graph_edges (
            id              SERIAL PRIMARY KEY,
            user_id         VARCHAR NOT NULL,
            source_node_id  VARCHAR REFERENCES graph_nodes(id) ON DELETE CASCADE,
            target_node_id  VARCHAR REFERENCES graph_nodes(id) ON DELETE CASCADE,
            relationship    VARCHAR DEFAULT 'RELATED_TO',
            strength        FLOAT DEFAULT 0.5,
            created_at      TIMESTAMP DEFAULT NOW()
        )
        """,
        # Migration statements for graph_nodes tables that already existed
        # before last_accessed_at / superseded_by / archived_at were added.
        # CREATE TABLE IF NOT EXISTS above is a no-op on an existing table,
        # so these ADD COLUMN IF NOT EXISTS statements are what actually
        # bring a live DB up to date. Safe to run every startup.
        "ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP DEFAULT NOW()",
        "ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS superseded_by VARCHAR REFERENCES graph_nodes(id)",
        "ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS archived_at TIMESTAMP",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_user_id ON graph_nodes(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_domain   ON graph_nodes(domain)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_superseded_by ON graph_nodes(superseded_by)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_last_accessed ON graph_nodes(last_accessed_at)",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source   ON graph_edges(source_node_id)",
    ]

    conn = _get_connection()
    try:
        cur = conn.cursor()
        for statement in ddl:
            cur.execute(statement)
        conn.commit()
        cur.close()
        logger.info("Graph tables initialised in PostgreSQL")
    except Exception as e:
        conn.rollback()
        logger.error(f"initialize_tables failed: {e}")
        raise
    finally:
        _release_connection(conn)


# ================================================================
# UTILITY: safe_parse_json
# ================================================================

def safe_parse_json(text: str) -> Dict[str, Any]:
    """
    Robustly parse a JSON string that may include LLM formatting artifacts.

    Handles markdown code fences, preamble text, and postamble text —
    the three most common ways an LLM ignores "Return ONLY the JSON".

    Raises json.JSONDecodeError only if the text contains no valid JSON at all.
    The caller's try/except catches this and falls back to safe defaults.
    """
    text = re.sub(r'```json|```', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        raise


# ================================================================
# UTILITY: is_declarative
# ================================================================

# Common interrogative openers — question words and the auxiliary/modal
# verbs that start most yes/no questions in English.
_INTERROGATIVE_STARTERS = (
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how",
    "is", "are", "was", "were", "am",
    "do", "does", "did",
    "can", "could", "may", "might",
    "will", "would", "shall", "should",
    "have", "has", "had",
)


def is_declarative(text: str) -> bool:
    """
    Cheap, deterministic check for whether `text` is a statement (a fact or
    task worth remembering) rather than a question.

    This is called as the very first thing inside process_memory_intent(),
    before extract_graph_metadata() ever runs — so a question never reaches
    the LLM extraction call, let alone graph storage. This is what stops a
    message like "What are my pending reminders?" from being stored as if
    it were a fact.

    Deliberately a heuristic, not an LLM call: question-vs-statement is
    simple syntax, and every single stored message passes through here —
    spending an LLM round trip just to decide whether to spend another LLM
    round trip would defeat the purpose.

    Two signals, either one is enough to classify as a question:
      - the text ends with "?"
      - the text starts with a common interrogative word or auxiliary verb
        ("what", "is", "do", "can", "will", ...)

    Empty/whitespace-only text is also treated as NOT declarative — there's
    nothing there worth storing either way.

    Returns True if `text` looks like a statement, False if it looks like
    a question (or is empty).
    """
    stripped = text.strip()
    if not stripped:
        return False

    if stripped.endswith("?"):
        return False

    first_word_match = re.match(r"[A-Za-z']+", stripped)
    if first_word_match and first_word_match.group(0).lower() in _INTERROGATIVE_STARTERS:
        return False

    return True


# ================================================================
# UTILITY: _call_fast_llm
# ================================================================

def _call_fast_llm(prompt: str) -> str:
    """
    Same call shape as ai_service.call_llm(context="") but pinned to
    config.FAST_LLM_MODEL instead of the large chat model.

    Used only for structured extraction (JSON metadata) — a task that
    doesn't need the large model's reasoning capacity, unlike the
    domain/task-matching LLM calls elsewhere in this file, which stay on
    the large model (see the NOTE above about why those need real judgment).
    """
    logger.info(f"_call_fast_llm model={config.FAST_LLM_MODEL}")
    response = _groq_client.chat.completions.create(
        model=config.FAST_LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ================================================================
# FUNCTION 0: classify_domain_by_embedding
# ================================================================

def classify_domain_by_embedding(text: str, userId: str) -> Optional[str]:
    """
    Fast, LLM-free alternative to resolve_domain() for the common case of
    tagging a new memory with one of this user's existing domains.

    See the CALIBRATION STATUS note on DOMAIN_CLASSIFIER_THRESHOLD above —
    reasonably confident on write-time (atomic-fact) text, still unvalidated
    on query-time (natural-language) text.

    For each of the user's existing domains, computes the centroid (mean)
    of the sentence-transformer embeddings of every graph_nodes.text already
    tagged with that domain, then compares `text`'s own embedding to each
    centroid by cosine similarity.

    This is a different signal from resolve_domain()'s already-tried-and-
    failed embedding approach (see the NOTE near the top of this file):
    that compared domain LABEL strings to label strings ("job_search" vs
    "career" scored 0.39). This compares memory CONTENT to memory content —
    whether a new paragraph reads like the other paragraphs already filed
    under a domain, which is a much stronger signal.

    Returns the matched domain if the top similarity clears
    DOMAIN_CLASSIFIER_THRESHOLD, so the caller can skip resolve_domain()
    (and its LLM call) entirely. Returns None — meaning "fall back to
    resolve_domain()" — when this user has no existing domains yet, when
    nothing clears the threshold, or on any embedding/DB failure.

    Every call logs its top domain and score, even below-threshold ones,
    so a batch of real classifications can be reviewed to retune the 0.70
    cutoff empirically — the same way RELEVANCE_THRESHOLD was tuned.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT domain, text FROM graph_nodes
            WHERE user_id = %s AND superseded_by IS NULL
              AND text IS NOT NULL AND text != ''
            """,
            (userId,),
        )
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        logger.warning(
            f"classify_domain_by_embedding failed to fetch nodes userId={userId}: {e}"
        )
        return None
    finally:
        _release_connection(conn)

    texts_by_domain: Dict[str, List[str]] = {}
    for row in rows:
        d = row["domain"]
        if d:
            texts_by_domain.setdefault(d, []).append(row["text"])

    if not texts_by_domain:
        return None

    try:
        all_texts = [t for texts in texts_by_domain.values() for t in texts]
        all_vectors = np.array(embeddings.encode_batch(all_texts))

        centroids: Dict[str, np.ndarray] = {}
        offset = 0
        for domain, texts in texts_by_domain.items():
            n = len(texts)
            centroids[domain] = all_vectors[offset:offset + n].mean(axis=0)
            offset += n

        query_vector = np.array(embeddings.encode(text))
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return None

        best_domain, best_score = None, -1.0
        for domain, centroid in centroids.items():
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm == 0:
                continue
            score = float(np.dot(query_vector, centroid) / (query_norm * centroid_norm))
            if score > best_score:
                best_domain, best_score = domain, score
    except Exception as e:
        logger.warning(
            f"classify_domain_by_embedding failed to embed/score userId={userId}: {e}"
        )
        return None

    logger.info(
        f"classify_domain_by_embedding userId={userId} top_domain={best_domain!r} "
        f"score={best_score:.3f} threshold={DOMAIN_CLASSIFIER_THRESHOLD} "
        f"text={text[:80]!r}"
    )

    if best_domain is not None and best_score >= DOMAIN_CLASSIFIER_THRESHOLD:
        return best_domain
    return None


# ================================================================
# FUNCTION 1: extract_graph_metadata
# ================================================================

def extract_graph_metadata(text: str, userId: str) -> dict:
    """
    Use the LLM to extract structured metadata from a raw memory text.

    Returns a dict with domain, status, importance_score, timeline fields, etc.
    On LLM failure returns safe defaults — never raises.
    """
    prompt = (
        "Analyze this text and extract structured metadata.\n"
        "Return ONLY a JSON object with these exact fields:\n"
        "{\n"
        '  "entity_name": "the main subject/entity in this text",\n'
        '  "domain": "a short life-area label for this memory (e.g. job, study, health, '
        'finance, relationship, travel, food, hobby) — invent a new concise label if none fit",\n'
        '  "sub_domain": "specific subcategory within the domain",\n'
        '  "status": "ongoing or completed",\n'
        '  "importance_score": 0.5,\n'
        '  "related_entities": ["list of other entities mentioned"]\n'
        "}\n\n"
        "Rules for status:\n"
        '- "completed": finished/done/passed/failed/got/received/quit\n'
        '- "ongoing": preparing/studying/working/applying/waiting\n'
        '- Default to "ongoing" if unclear\n\n'
        "Rules for importance_score:\n"
        "- 0.8-1.0: major life events (job offer, diagnosis, graduation, major financial decision)\n"
        "- 0.5-0.7: significant events (interview, exam, starting course, investment)\n"
        "- 0.2-0.4: regular activities (study session, gym, movie)\n"
        "- 0.1-0.2: casual mentions (food, minor observations)\n\n"
        f'Text: "{text}"\n\n'
        "Return ONLY the JSON. No explanation. No markdown."
    )

    try:
        raw = _call_fast_llm(prompt)
        parsed: Dict[str, Any] = safe_parse_json(raw)

        domain = str(parsed.get("domain", "general")).lower().strip() or "general"
        # Try the cheap embedding classifier first — if this memory's content
        # clearly clusters with an existing domain, use it directly and skip
        # resolve_domain()'s LLM call. Only fall back to resolve_domain() (the
        # LLM-guessed candidate above) when the classifier isn't confident.
        classified_domain = classify_domain_by_embedding(text, userId)
        domain = classified_domain if classified_domain else resolve_domain(domain, userId)

        status = str(parsed.get("status", "ongoing")).lower()
        if status not in ("ongoing", "completed"):
            status = "ongoing"

        try:
            importance = float(parsed.get("importance_score", 0.5))
            importance = max(0.0, min(1.0, importance))
        except (ValueError, TypeError):
            importance = 0.5

        related = parsed.get("related_entities", [])
        if not isinstance(related, list):
            related = []
        related = [str(e) for e in related if e]

    except Exception as e:
        logger.warning(
            f"extract_graph_metadata failed for userId={userId}: {e} — using defaults"
        )
        domain = "general"
        status = "ongoing"
        importance = 0.5
        related = []
        parsed = {}

    now = datetime.now()
    return {
        "entity_name": str(parsed.get("entity_name", text[:60])),
        "domain": domain,
        "sub_domain": str(parsed.get("sub_domain", "")),
        "status": status,
        "importance_score": importance,
        "related_entities": related,
        "timeline_year": now.year,
        "timeline_month": now.month,
        "timeline_label": now.strftime("%B %Y"),
    }


# ================================================================
# FUNCTION 1B: resolve_domain
# ================================================================

def resolve_domain(candidate_domain: str, userId: str) -> str:
    """
    Resolve an LLM-suggested domain against this user's existing domains.

    Domains are no longer restricted to a fixed set — the LLM can invent any
    label. To stop near-duplicates piling up over time (e.g. "job_search" vs
    "career"), this reuses an existing domain when it's a close semantic
    match, and only mints a new domain label when nothing close already
    exists.

    Returns the resolved domain (always lowercased/stripped).
    """
    candidate = candidate_domain.lower().strip() or "general"

    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT domain FROM graph_nodes WHERE user_id = %s",
            (userId,),
        )
        existing = [row["domain"] for row in cur.fetchall() if row["domain"]]
        cur.close()
    except Exception as e:
        logger.error(f"resolve_domain failed to fetch existing domains userId={userId}: {e}")
        return candidate
    finally:
        _release_connection(conn)

    if not existing:
        return candidate

    if candidate in existing:
        return candidate

    prompt = (
        "A memory-tagging system is assigning a life-area label to a new memory.\n\n"
        f'NEW CANDIDATE LABEL: "{candidate}"\n\n'
        "EXISTING LABELS already used for this same user:\n"
        + "\n".join(f'- "{d}"' for d in existing) + "\n\n"
        "If the NEW CANDIDATE LABEL refers to essentially the same life-area as one "
        "of the EXISTING LABELS (e.g. \"job_search\" and \"career\" are the same "
        "area; \"gym\" and \"sport\" are the same area), return that existing label "
        "exactly as written. Otherwise, if it's a genuinely different area, return "
        "the NEW CANDIDATE LABEL unchanged.\n\n"
        "Return ONLY a JSON object:\n"
        '{"resolved_domain": "<an existing label, or the new candidate label>", '
        '"confidence": 0.0}\n'
        "confidence is how sure you are that resolved_domain is correct (1.0 = certain).\n"
        "No explanation. No markdown."
    )

    try:
        raw = ai_service.call_llm(message=prompt, context="")
        parsed = safe_parse_json(raw)
        resolved = str(parsed.get("resolved_domain", candidate)).lower().strip()
        confidence = float(parsed.get("confidence", 0.0))
    except Exception as e:
        logger.warning(
            f"resolve_domain LLM match failed userId={userId}: {e} — using raw candidate"
        )
        return candidate

    if resolved in existing and confidence >= DOMAIN_MATCH_THRESHOLD:
        logger.info(
            f"resolve_domain userId={userId} matched '{candidate}' -> "
            f"'{resolved}' (confidence={confidence:.2f})"
        )
        return resolved

    return candidate


# ================================================================
# FUNCTION 2: add_to_graph
# ================================================================

def add_to_graph(
    userId: str,
    node_id: str,
    text: str,
    metadata: dict,
) -> bool:
    """
    Insert a new node into graph_nodes and create edges to related
    existing nodes (same domain + same month).

    Returns True on success, False on failure.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()

        # Insert (or update on conflict — idempotent for retries)
        cur.execute(
            """
            INSERT INTO graph_nodes (
                id, user_id, entity_name, domain, sub_domain,
                status, timeline_year, timeline_month, timeline_label,
                importance_score, text, created_at, last_updated
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                last_updated = NOW(),
                status = EXCLUDED.status
            """,
            (
                node_id,
                userId,
                metadata.get("entity_name", ""),
                metadata.get("domain", "general"),
                metadata.get("sub_domain", ""),
                metadata.get("status", "ongoing"),
                metadata.get("timeline_year"),
                metadata.get("timeline_month"),
                metadata.get("timeline_label", ""),
                metadata.get("importance_score", 0.5),
                text,
            ),
        )

        # Find existing nodes in the same domain + same calendar month
        cur.execute(
            """
            SELECT id, status
            FROM graph_nodes
            WHERE user_id = %s
              AND id != %s
              AND domain = %s
              AND timeline_year = %s
              AND timeline_month = %s
              AND superseded_by IS NULL
            LIMIT 10
            """,
            (
                userId,
                node_id,
                metadata.get("domain", "general"),
                metadata.get("timeline_year"),
                metadata.get("timeline_month"),
            ),
        )
        related_nodes = cur.fetchall()

        new_status = metadata.get("status", "ongoing")
        for related in related_nodes:
            if new_status == "completed" and related["status"] == "ongoing":
                relationship, strength = "LED_TO", 0.7
            else:
                relationship, strength = "RELATED_TO", 0.6

            # Insert edge only if it doesn't already exist
            cur.execute(
                """
                INSERT INTO graph_edges (
                    user_id, source_node_id, target_node_id,
                    relationship, strength, created_at
                )
                SELECT %s, %s, %s, %s, %s, NOW()
                WHERE NOT EXISTS (
                    SELECT 1 FROM graph_edges
                    WHERE source_node_id = %s AND target_node_id = %s
                )
                """,
                (
                    userId, node_id, related["id"],
                    relationship, strength,
                    node_id, related["id"],
                ),
            )

        conn.commit()
        cur.close()
        logger.info(
            f"Graph node stored userId={userId} nodeId={node_id} "
            f"edges={len(related_nodes)}"
        )
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"add_to_graph failed userId={userId}: {e}")
        return False
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 3: get_graph_context
# ================================================================

def get_graph_context(userId: str, query_text: str, topN: int = 5) -> dict:
    """
    Score and retrieve the most relevant graph nodes for a query.

    Scoring: importance * 0.4 + recency * 0.3 + keyword_match * 0.3

    Archived nodes (superseded_by IS NOT NULL — see resolve_conflicting_nodes())
    are excluded everywhere in this function: the candidate pool, the neighbor
    traversal, and the timeline_summary built here. This function represents
    "current state", so a node that's been consolidated into another one
    shouldn't reappear or double-count. Contrast with get_timeline_summary(),
    which deliberately keeps archived nodes so historical trajectory isn't lost.

    Every node returned in `nodes` has its last_accessed_at bumped to NOW()
    in a single batch UPDATE. This is distinct from last_updated, which only
    changes when a node's content/status actually changes — last_accessed_at
    changes on every retrieval regardless of whether anything changed, and
    feeds get_stale_nodes() for future pruning of genuinely dead memories.

    Returns a dict compatible with GraphContextResponse:
      nodes           — List[GraphNodeInfo dicts]
      neighbors       — List[GraphNodeInfo dicts] (connected nodes)
      timeline_summary — {this_month, ongoing, completed, domains}
      graph_context_text — pre-formatted string for LLM prompt
    """
    empty = {
        "nodes": [],
        "neighbors": [],
        "timeline_summary": {
            "this_month": 0, "ongoing": 0, "completed": 0, "domains": {},
        },
        "graph_context_text": "",
    }

    conn = _get_connection()
    try:
        cur = conn.cursor()
        now = datetime.now()

        # ── Fetch all nodes for this user ──────────────────────────
        cur.execute(
            """
            SELECT id, entity_name, domain, sub_domain, status,
                   timeline_year, timeline_month, timeline_label,
                   importance_score, text
            FROM graph_nodes
            WHERE user_id = %s AND superseded_by IS NULL
            ORDER BY created_at DESC
            """,
            (userId,),
        )
        all_nodes = cur.fetchall()

        if not all_nodes:
            cur.close()
            return empty

        # ── Score each node ────────────────────────────────────────
        query_lower = query_text.lower()
        query_words = [w for w in query_lower.split() if len(w) > 2]
        scored: List[tuple] = []

        for row in all_nodes:
            node_year = row["timeline_year"] or now.year
            node_month = row["timeline_month"] or now.month
            months_ago = (now.year - node_year) * 12 + (now.month - node_month)

            if months_ago == 0:
                recency = 1.0
            elif months_ago == 1:
                recency = 0.7
            elif months_ago <= 3:
                recency = 0.4
            else:
                recency = 0.1

            searchable = (
                f"{row['domain'] or ''} {row['sub_domain'] or ''} "
                f"{row['text'] or ''} {row['entity_name'] or ''}"
            ).lower()
            if query_words:
                matched_words = sum(1 for word in query_words if word in searchable)
                keyword_match = matched_words / len(query_words)
            else:
                keyword_match = 0.0

            importance = float(row["importance_score"] or 0.5)
            score = importance * 0.4 + recency * 0.3 + keyword_match * 0.3

            scored.append((score, {
                "node_id": row["id"],
                "text": row["text"] or "",
                "entity_name": row["entity_name"] or "",
                "domain": row["domain"] or "general",
                "sub_domain": row["sub_domain"] or "",
                "status": row["status"] or "ongoing",
                "timeline_label": row["timeline_label"] or "",
                "importance_score": importance,
                # kept for context-text building / relevance filtering;
                # Pydantic ignores extra fields
                "_timeline_year": node_year,
                "_timeline_month": node_month,
                "_keyword_match": keyword_match,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Filter out weak matches BEFORE slicing to topN — otherwise a query
        # like "write python code to square a number" would still pull in
        # the top N nodes even if none of them are actually relevant. A score
        # threshold alone isn't enough here: importance*0.4 + recency*0.3 can
        # reach 0.7 on their own for a recent, important node with ZERO
        # keyword overlap with the query — so also require at least some
        # keyword relevance whenever the query has meaningful words.
        relevant = [
            (score, node) for score, node in scored
            if score >= RELEVANCE_THRESHOLD
            and (not query_words or node["_keyword_match"] > 0)
        ]

        if not relevant:
            cur.close()
            logger.info(
                f"get_graph_context userId={userId} no nodes cleared "
                f"RELEVANCE_THRESHOLD={RELEVANCE_THRESHOLD} "
                f"(top score={scored[0][0]:.2f})"
            )
            return empty

        top_nodes = [n for _, n in relevant[:topN]]
        top_node_ids = [n["node_id"] for n in top_nodes]

        # ── Freshness tracking ──────────────────────────────────────
        # Every node actually returned to the caller counts as "accessed",
        # regardless of whether its content changed. Single batch UPDATE
        # for all top_node_ids, not one query per node.
        cur.execute(
            "UPDATE graph_nodes SET last_accessed_at = NOW() WHERE id = ANY(%s)",
            (top_node_ids,),
        )
        conn.commit()

        # ── Fetch neighbor nodes ───────────────────────────────────
        neighbors: List[dict] = []
        if top_node_ids:
            placeholders = ",".join(["%s"] * len(top_node_ids))
            cur.execute(
                f"""
                SELECT DISTINCT
                    gn.id, gn.entity_name, gn.domain, gn.sub_domain,
                    gn.status, gn.timeline_label, gn.text,
                    gn.importance_score, ge.relationship
                FROM graph_edges ge
                JOIN graph_nodes gn ON gn.id = ge.target_node_id
                WHERE ge.source_node_id IN ({placeholders})
                  AND gn.id NOT IN ({placeholders})
                  AND gn.user_id = %s
                  AND gn.superseded_by IS NULL
                LIMIT 10
                """,
                top_node_ids + top_node_ids + [userId],
            )
            for row in cur.fetchall():
                neighbors.append({
                    "node_id": row["id"],
                    "text": row["text"] or "",
                    "entity_name": row["entity_name"] or "",
                    "domain": row["domain"] or "general",
                    "sub_domain": row["sub_domain"] or "",
                    "status": row["status"] or "ongoing",
                    "timeline_label": row["timeline_label"] or "",
                    "importance_score": float(row["importance_score"] or 0.5),
                    # extra field for context text; ignored by Pydantic
                    "_relationship": row["relationship"] or "RELATED_TO",
                })

        # ── Build timeline summary (current state, archived excluded) ──
        cur.execute(
            """
            SELECT timeline_year, timeline_month, domain, status, COUNT(*) AS cnt
            FROM graph_nodes
            WHERE user_id = %s AND superseded_by IS NULL
            GROUP BY timeline_year, timeline_month, domain, status
            """,
            (userId,),
        )
        timeline_summary: Dict[str, Any] = {
            "this_month": 0, "ongoing": 0, "completed": 0, "domains": {},
        }
        for row in cur.fetchall():
            cnt = row["cnt"]
            if row["timeline_year"] == now.year and row["timeline_month"] == now.month:
                timeline_summary["this_month"] += cnt
            if row["status"] == "ongoing":
                timeline_summary["ongoing"] += cnt
            else:
                timeline_summary["completed"] += cnt
            d = row["domain"] or "general"
            timeline_summary["domains"][d] = timeline_summary["domains"].get(d, 0) + cnt

        cur.close()

        # ── Build context text ─────────────────────────────────────
        lines: List[str] = [
            "Memory Graph Context:", "",
            f"Recent memories ({timeline_summary['this_month']} this month):",
        ]
        for node in top_nodes:
            tag = "[ONGOING]" if node["status"] == "ongoing" else "[DONE]"
            lines.append(
                f"- {tag} {node['domain'].title()}: "
                f"{node['entity_name']} ({node['timeline_label']})"
            )
        if neighbors:
            lines.append("")
            lines.append("Related context:")
            for nb in neighbors[:3]:
                rel = nb["_relationship"].lower().replace("_", " ")
                lines.append(f"- {nb['entity_name']} {rel} {top_nodes[0]['entity_name']}")

        # Strip internal underscore keys before returning
        clean_nodes = [
            {k: v for k, v in n.items() if not k.startswith("_")}
            for n in top_nodes
        ]
        clean_neighbors = [
            {k: v for k, v in nb.items() if not k.startswith("_")}
            for nb in neighbors
        ]

        return {
            "nodes": clean_nodes,
            "neighbors": clean_neighbors,
            "timeline_summary": timeline_summary,
            "graph_context_text": "\n".join(lines),
        }

    except Exception as e:
        logger.error(f"get_graph_context failed userId={userId}: {e}")
        return empty
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 3B: get_stale_nodes
# ================================================================

def get_stale_nodes(userId: str, days_threshold: int = 90) -> List[dict]:
    """
    Return active (non-archived) nodes that haven't been accessed via
    get_graph_context() in more than `days_threshold` days.

    Read-only — this never deletes or archives anything itself. It exists
    so a future pruning/archiving pass has a cheap way to find candidates:
    memories nobody has retrieved in a long time are the ones most likely
    to be safe to prune.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, entity_name, domain, sub_domain, status,
                   timeline_label, importance_score, last_accessed_at
            FROM graph_nodes
            WHERE user_id = %s
              AND superseded_by IS NULL
              AND last_accessed_at < NOW() - (%s * INTERVAL '1 day')
            ORDER BY last_accessed_at ASC
            """,
            (userId, days_threshold),
        )
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]

    except Exception as e:
        logger.error(f"get_stale_nodes failed userId={userId}: {e}")
        return []
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 4: update_node_status
# ================================================================

def update_node_status(userId: str, node_id: str, new_status: str) -> bool:
    """
    Update a graph node's status to "ongoing" or "completed".

    Returns True if the node was found and updated, False otherwise.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE graph_nodes
            SET status = %s, last_updated = NOW()
            WHERE id = %s AND user_id = %s
            """,
            (new_status, node_id, userId),
        )
        updated = cur.rowcount > 0
        conn.commit()
        cur.close()
        logger.info(f"update_node_status nodeId={node_id} → {new_status} updated={updated}")
        return updated

    except Exception as e:
        conn.rollback()
        logger.error(f"update_node_status failed userId={userId}: {e}")
        return False
    finally:
        _release_connection(conn)


# Matches entity_names like "the user", "the speaker", "the user's task",
# "the speaker's request" — generic pronoun/references rather than a real
# named entity or task. A candidate node with one of these was very likely
# mis-extracted from a question (see _is_generic_entity_name() docstring).
_GENERIC_ENTITY_PATTERN = re.compile(
    r"^the\s+(user|speaker)('s\s+(task|question|request|reminder|goal))?$",
    re.IGNORECASE,
)


def _is_generic_entity_name(name: str) -> bool:
    """
    True if `name` is a generic pronoun/reference ("the user", "the
    speaker's task") rather than a genuine named entity or task.

    Candidates with these entity_names were likely extracted from a
    question rather than a real fact, and must never be treated as
    matchable tasks. This is a stopgap for bad extraction from questions
    (Problem 3) in case that fix hasn't fully landed yet — filtering here
    means find_matching_task_node() stays safe even if extraction slips.
    """
    return bool(_GENERIC_ENTITY_PATTERN.match((name or "").strip()))


# ================================================================
# FUNCTION 5: find_matching_task_node
# ================================================================

def find_matching_task_node(userId: str, text: str, domain: str) -> Optional[str]:
    """
    Search this user's existing "ongoing" nodes in the same domain for one
    that refers to the exact same task, action, or named entity as `text`
    — not merely one that shares the same domain/life-area.

    Used to detect when a new message is really an update to an existing
    task/fact (e.g. "finished the AWS deployment") rather than a brand-new
    memory — this is what prevents duplicate/contradictory nodes for the
    same logical task.

    Candidates whose entity_name is a generic pronoun/reference (e.g. "the
    user", "the speaker's task") are excluded before they ever reach the
    LLM — see _is_generic_entity_name().

    Returns the matching node's id, or None if nothing scores above
    TASK_MATCH_THRESHOLD (including when there are no eligible candidates).
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, text, entity_name
            FROM graph_nodes
            WHERE user_id = %s AND domain = %s AND status = 'ongoing'
              AND superseded_by IS NULL
            ORDER BY last_updated DESC
            LIMIT 15
            """,
            (userId, domain),
        )
        candidates = [
            row for row in cur.fetchall()
            if (row["text"] or row["entity_name"])
            and not _is_generic_entity_name(row["entity_name"])
        ]
        cur.close()
    except Exception as e:
        logger.error(f"find_matching_task_node failed to fetch candidates userId={userId}: {e}")
        return None
    finally:
        _release_connection(conn)

    if not candidates:
        return None

    prompt = (
        "You are matching a NEW message against a user's EXISTING ongoing "
        "tasks/notes to detect if it's an update to the SAME specific task, "
        "action, or named entity — not just something from the same "
        "general life-area (domain).\n\n"
        "IMPORTANT: sharing a domain/category (e.g. both being about "
        '"job") is NOT sufficient for a match. The NEW MESSAGE must refer '
        "to the exact same underlying task or entity as an EXISTING item. "
        "A different task, a standing fact, or a question in the same "
        "domain is NOT a match.\n\n"
        "Examples:\n"
        '- MATCH: "applying for the AWS job" and "got the AWS offer" — '
        "same task, reported at a different stage.\n"
        '- NO MATCH: "I\'m a software engineer at Yardi" and "update my '
        'resume before Friday" — same domain (job), but one is a standing '
        "fact about the person and the other is a distinct, unrelated "
        "task. Do not match these.\n"
        '- NO MATCH: "what are my pending reminders" and "remind me to '
        'deploy to AWS" — a question is never a match target for a task; '
        "only match against genuine facts/tasks, never questions.\n\n"
        f'NEW MESSAGE: "{text}"\n\n'
        "EXISTING ITEMS (id: text):\n"
        + "\n".join(
            f'- {row["id"]}: "{row["text"] or row["entity_name"]}"'
            for row in candidates
        ) + "\n\n"
        "If the NEW MESSAGE is about the exact same underlying task/entity "
        "as one of the existing items (even if worded very differently), "
        "return that item's id. Otherwise — including when it's merely in "
        "the same domain — return matched_id as null.\n\n"
        'Return ONLY JSON: {"matched_id": "<id or null>", "confidence": 0.0, '
        '"reason": "<one sentence explaining the decision>"}\n'
        "confidence is how sure you are of the match (1.0 = certain).\n"
        "No explanation outside the JSON. No markdown."
    )

    try:
        raw = ai_service.call_llm(message=prompt, context="")
        parsed = safe_parse_json(raw)
        matched_id = parsed.get("matched_id")
        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", "")).strip()
    except Exception as e:
        logger.warning(f"find_matching_task_node LLM match failed userId={userId}: {e}")
        return None

    valid_ids = {row["id"] for row in candidates}
    if matched_id in valid_ids and confidence >= TASK_MATCH_THRESHOLD:
        logger.info(
            f"find_matching_task_node userId={userId} matched nodeId={matched_id} "
            f"(confidence={confidence:.2f}) reason={reason!r}"
        )
        return matched_id

    return None


# ================================================================
# FUNCTION 5B: process_memory_intent
# ================================================================

def process_memory_intent(userId: str, text: str, memoryId: Optional[str] = None) -> dict:
    """
    Single entry point for turning a raw memory text into a graph mutation.
    This replaces calling extract_graph_metadata() + add_to_graph() directly
    from the route handler, because that combination always created a new
    node — including for messages that just report a prior task finishing.

    Decides between four outcomes:
      - the text is a question, not a statement (is_declarative() is False)
        → skip entirely. Never calls extract_graph_metadata() or writes
        anything — this is what stops e.g. "What are my pending reminders?"
        from being stored as if it were a fact.
      - the text reports completion of an existing "ongoing" node → update
        its status to "completed" instead of creating a new node.
      - the text closely matches an existing "ongoing" node (duplicate
        mention of the same task/fact) → refresh its last_updated timestamp
        instead of creating a duplicate node.
      - otherwise → create a new node via add_to_graph(), as before.

    `memoryId`, when provided by the caller, is used as the new node's id so
    it stays in sync with the corresponding ChromaDB memory id (matching the
    contract graph_routes.py already relies on). If omitted, an id is
    generated the same way Spring Boot generates memory ids.

    This is the single entry point both graph_routes.py's /graph/process
    endpoint and main.py's process_graph_background (triggered from
    /ai/chat) call — the is_declarative() gate lives here, not duplicated
    in each caller, precisely so it can't end up wired into one call path
    but not the other.

    Returns:
        {
            "action": "skipped" | "updated" | "created",
            "node_id": Optional[str],  # None when action == "skipped"
            "metadata": dict,          # {} when action == "skipped"
            "success": bool,           # whether the underlying DB write succeeded
        }
    """
    if not is_declarative(text):
        logger.info(
            f"process_memory_intent userId={userId} skipped — text is a "
            f"question, not a fact/task: {text[:80]!r}"
        )
        return {"action": "skipped", "node_id": None, "metadata": {}, "success": True}

    metadata = extract_graph_metadata(text, userId)
    domain = metadata.get("domain", "general")
    status = metadata.get("status", "ongoing")

    if status == "completed":
        matched_id = find_matching_task_node(userId, text, domain)
        if matched_id:
            success = update_node_status(userId, matched_id, "completed")
            return {"action": "updated", "node_id": matched_id, "metadata": metadata, "success": success}

        logger.warning(
            f"process_memory_intent userId={userId} reported a completion with no "
            f"matching prior ongoing task — creating a new node. text={text[:80]!r}"
        )

    elif status == "ongoing":
        matched_id = find_matching_task_node(userId, text, domain)
        if matched_id:
            success = update_node_status(userId, matched_id, "ongoing")
            logger.info(
                f"process_memory_intent userId={userId} duplicate ongoing task matched "
                f"nodeId={matched_id} — refreshed instead of creating a new node"
            )
            return {"action": "updated", "node_id": matched_id, "metadata": metadata, "success": success}

    node_id = memoryId or f"{userId}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
    success = add_to_graph(userId=userId, node_id=node_id, text=text, metadata=metadata)
    return {"action": "created", "node_id": node_id, "metadata": metadata, "success": success}


# ================================================================
# FUNCTION 5C: _cluster_nodes_via_llm (private helper)
# ================================================================

def _cluster_nodes_via_llm(nodes: List[dict]) -> List[List[str]]:
    """
    Ask the LLM to group a list of graph_nodes rows (id/entity_name/text/
    status/created_at) into clusters that refer to the same underlying
    task, entity, or fact — even if worded differently or at different
    status stages (e.g. "applying for the job" and "got the offer" are the
    same underlying task at different points in time).

    Uses ai_service (LLM judgment), consistent with resolve_domain() and
    find_matching_task_node() — raw embedding cosine similarity was already
    proven (see the NOTE near the top of this file) to under-score exactly
    these cases.

    Returns a list of clusters, each a list of >=2 node ids. Nodes not
    mentioned in any returned cluster are singletons with no conflict.
    Returns [] on any failure (bad LLM output, JSON parse failure, etc.) —
    callers must treat that as "nothing to merge this pass", not an error.
    """
    listing = "\n".join(
        f'- {n["id"]}: "{n["entity_name"] or ""} — {(n["text"] or "")[:200]}" '
        f'(status={n["status"]})'
        for n in nodes
    )

    prompt = (
        "You are reviewing a user's memory graph for duplicate/conflicting "
        "entries. Below is a list of memory nodes from the SAME domain. "
        "Group together nodes that refer to the SAME underlying task, "
        "entity, or fact — even if worded very differently, or at "
        "different status stages (e.g. one node says \"applying for the "
        "job\" and another says \"got the offer\" — these are the same "
        "underlying task at different points in time, so they belong in "
        "the same group).\n\n"
        "Nodes:\n"
        f"{listing}\n\n"
        "Return ONLY a JSON object:\n"
        '{"clusters": [["id1", "id2"], ["id3", "id4", "id5"]]}\n'
        "Only include clusters with 2 or more ids — omit any node that "
        "doesn't duplicate/conflict with anything else. "
        "No explanation. No markdown."
    )

    try:
        raw = ai_service.call_llm(message=prompt, context="")
        parsed = safe_parse_json(raw)
        raw_clusters = parsed.get("clusters", [])
        if not isinstance(raw_clusters, list):
            return []

        valid_ids = {n["id"] for n in nodes}
        clusters: List[List[str]] = []
        for cluster in raw_clusters:
            if not isinstance(cluster, list):
                continue
            ids = [str(i) for i in cluster if str(i) in valid_ids]
            if len(ids) >= 2:
                clusters.append(ids)
        return clusters

    except Exception as e:
        logger.warning(f"_cluster_nodes_via_llm failed: {e}")
        return []


# ================================================================
# FUNCTION 5D: resolve_conflicting_nodes
# ================================================================

async def resolve_conflicting_nodes(
    userId: str,
    domain: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Find nodes that refer to the same underlying task/entity/fact and
    consolidate them: the most recently created node in each cluster stays
    "active", every other node in that cluster is archived (never deleted)
    by setting superseded_by + archived_at.

    This is a safety net on top of process_memory_intent()'s write-time
    dedup check (find_matching_task_node(), TASK_MATCH_THRESHOLD=0.85) —
    it catches nodes that predate that fix, or edge cases the confidence
    threshold missed at write time.

    Args:
        userId: whose graph to clean up.
        domain: if given, only this domain's active nodes are considered.
            Used by the cheap per-write trigger (see process_memory_intent
            call sites in graph_routes.py / main.py) so each pass only
            re-examines the one domain that was just touched, not the
            user's whole graph. If None, every domain the user has is
            processed — used for a full sweep.
        limit: max nodes considered per domain group, most recent first
            (ORDER BY created_at DESC). The per-write trigger always passes
            CONFLICT_RESOLUTION_TRIGGER_LIMIT (20) to keep the LLM call cheap.
            KNOWN LIMITATION: with a limit set, a conflict between a very
            old dormant node and a brand-new one won't be caught by the
            cheap per-write trigger — only a limit=None full sweep
            (call this manually or on a schedule) will catch that case,
            since it considers a domain's entire history.

    Never raises — on any failure this logs a warning and returns an
    empty-ish summary, since it always runs off the main request/response
    path and must never break the chat/write pipeline.

    Returns:
        {
            "clusters_found": int,
            "nodes_archived": [node_id, ...],
            "active_nodes": [node_id, ...],
        }
    """
    summary = {"clusters_found": 0, "nodes_archived": [], "active_nodes": []}

    conn = _get_connection()
    try:
        cur = conn.cursor()

        if domain is not None:
            domains = [domain]
        else:
            cur.execute(
                """
                SELECT DISTINCT domain FROM graph_nodes
                WHERE user_id = %s AND superseded_by IS NULL
                """,
                (userId,),
            )
            domains = [row["domain"] for row in cur.fetchall() if row["domain"]]

        for d in domains:
            query = """
                SELECT id, entity_name, text, status, created_at
                FROM graph_nodes
                WHERE user_id = %s AND domain = %s AND superseded_by IS NULL
                ORDER BY created_at DESC
            """
            params: List[Any] = [userId, d]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            cur.execute(query, params)
            domain_nodes = cur.fetchall()

            if len(domain_nodes) < 2:
                continue  # nothing to consolidate with fewer than 2 nodes

            clusters = _cluster_nodes_via_llm(domain_nodes)
            nodes_by_id = {n["id"]: n for n in domain_nodes}

            for cluster_ids in clusters:
                cluster_nodes = [nodes_by_id[i] for i in cluster_ids if i in nodes_by_id]
                if len(cluster_nodes) < 2:
                    continue

                active = max(cluster_nodes, key=lambda n: n["created_at"])
                other_ids = [n["id"] for n in cluster_nodes if n["id"] != active["id"]]
                if not other_ids:
                    continue

                cur.execute(
                    """
                    UPDATE graph_nodes
                    SET superseded_by = %s, archived_at = NOW()
                    WHERE id = ANY(%s) AND user_id = %s
                    """,
                    (active["id"], other_ids, userId),
                )
                conn.commit()

                summary["clusters_found"] += 1
                summary["nodes_archived"].extend(other_ids)
                summary["active_nodes"].append(active["id"])

                logger.info(
                    f"resolve_conflicting_nodes userId={userId} domain={d} "
                    f"merged nodeIds={other_ids} -> active nodeId={active['id']}"
                )

        cur.close()
        return summary

    except Exception as e:
        conn.rollback()
        logger.warning(
            f"resolve_conflicting_nodes failed userId={userId} domain={domain}: "
            f"{e} — skipping this pass"
        )
        return summary
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 5E: get_all_nodes
# ================================================================

def get_all_nodes(include_archived: bool = False) -> List[dict]:
    """
    Return every memory node stored in graph_nodes, across all users.

    Admin/debug function — unlike every other read function in this file,
    this is NOT scoped to a single userId. Callers must treat the result
    as sensitive: it exposes every user's stored memory text in one call.

    By default excludes archived nodes (superseded_by IS NOT NULL), matching
    get_graph_context()'s "current state" behavior. Pass include_archived=True
    to also see nodes that were merged into another node by
    resolve_conflicting_nodes().
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        query = """
            SELECT id, user_id, entity_name, domain, sub_domain, status,
                   timeline_year, timeline_month, timeline_label,
                   importance_score, text, created_at, last_updated,
                   last_accessed_at, superseded_by, archived_at
            FROM graph_nodes
        """
        if not include_archived:
            query += " WHERE superseded_by IS NULL"
        query += " ORDER BY user_id ASC, created_at DESC"

        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]

    except Exception as e:
        logger.error(f"get_all_nodes failed: {e}")
        return []
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 5F: delete_user_graph
# ================================================================

def delete_user_graph(userId: str) -> dict:
    """
    Delete every graph_edges and graph_nodes row belonging to a single user.

    Edges are deleted first and counted explicitly — graph_edges rows also
    have ON DELETE CASCADE from graph_nodes, so deleting nodes first would
    cascade-delete edges silently and make edges_deleted always read 0.

    Never raises — on failure, rolls back and returns zero counts so a
    clear-memory call fails loudly via the caller's own error handling
    rather than partially deleting data.

    Returns {"nodes_deleted": int, "edges_deleted": int}.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()

        cur.execute("DELETE FROM graph_edges WHERE user_id = %s", (userId,))
        edges_deleted = cur.rowcount

        cur.execute("DELETE FROM graph_nodes WHERE user_id = %s", (userId,))
        nodes_deleted = cur.rowcount

        conn.commit()
        cur.close()
        logger.info(
            f"delete_user_graph userId={userId} "
            f"nodesDeleted={nodes_deleted} edgesDeleted={edges_deleted}"
        )
        return {"nodes_deleted": nodes_deleted, "edges_deleted": edges_deleted}

    except Exception as e:
        conn.rollback()
        logger.error(f"delete_user_graph failed userId={userId}: {e}")
        return {"nodes_deleted": 0, "edges_deleted": 0}
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 6: get_timeline_summary
# ================================================================

def get_timeline_summary(userId: str) -> dict:
    """
    Return all memories grouped by month → domain → ongoing/completed.

    Deliberately includes archived nodes (superseded_by IS NOT NULL) —
    unlike get_graph_context(), which filters them out because it represents
    "current state". This function represents full history, so an archived
    node's original month/status stays visible (e.g. "this task started in
    June, was consolidated into a later node in July") instead of vanishing.

    Example: {"June 2026": {"job": {"ongoing": [...], "completed": [...]}}}
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timeline_label, timeline_year, timeline_month,
                   domain, status, entity_name, text
            FROM graph_nodes
            WHERE user_id = %s
            ORDER BY timeline_year DESC, timeline_month DESC, domain ASC
            """,
            (userId,),
        )
        rows = cur.fetchall()
        cur.close()

        result: Dict[str, Any] = {}
        for row in rows:
            label = row["timeline_label"] or "Unknown"
            domain = row["domain"] or "general"
            status = row["status"] or "ongoing"
            name = row["entity_name"] or (row["text"] or "")[:60]

            if label not in result:
                result[label] = {}
            if domain not in result[label]:
                result[label][domain] = {"ongoing": [], "completed": []}
            result[label][domain].setdefault(status, []).append(name)

        return result

    except Exception as e:
        logger.error(f"get_timeline_summary failed userId={userId}: {e}")
        return {}
    finally:
        _release_connection(conn)
