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

# Fixed set of top-level life areas. `domain` must be one of these. Anything
# more specific belongs in the free-form `sub_domain` column, e.g.
# domain="hobbies", sub_domain="favorite cricket players". Previously
# `domain` itself was free-form (LLM-invented, merged after the fact by
# embedding and LLM similarity). That let related but distinct areas drift
# into one bucket, and near-synonyms ("sports", "favorite things") pile up
# as separate top-level domains. See resolve_domain() for the migration
# note on existing free-form data.
TOP_LEVEL_DOMAINS = [
    "job", "education", "hobbies", "health", "finance",
    "relationships", "travel", "food", "general",
]

# Cheap, deterministic mapping of common free-form labels onto
# TOP_LEVEL_DOMAINS, tried by resolve_domain() before it spends an LLM call.
# Also covers labels an older free-form run produced, which helps a
# backfill of existing data.
_DOMAIN_ALIASES = {
    "career": "job", "work": "job", "job_search": "job", "employment": "job",
    "study": "education", "studies": "education", "learning": "education",
    "school": "education", "college": "education", "university": "education",
    "hobby": "hobbies", "sport": "hobbies", "sports": "hobbies",
    "entertainment": "hobbies", "music": "hobbies", "games": "hobbies",
    "fitness": "health", "medical": "health", "wellness": "health",
    "money": "finance", "investment": "finance", "investments": "finance",
    "relationship": "relationships", "family": "relationships",
    "friends": "relationships", "social": "relationships",
    "trip": "travel", "trips": "travel",
    "diet": "food", "cooking": "food",
}

# Edge strength = EDGE_SIMILARITY_WEIGHT * cosine(text_a, text_b)
#               + (1 - EDGE_SIMILARITY_WEIGHT) * heuristic
# The heuristic is the old flat per-relationship value. It stands for the
# same-domain + same-month co-occurrence signal that made the two nodes
# edge candidates in the first place.
EDGE_SIMILARITY_WEIGHT = 0.5
EDGE_HEURISTIC_LED_TO = 0.7
EDGE_HEURISTIC_RELATED_TO = 0.6

# Minimum LLM-judged confidence for resolve_domain() to map a stray,
# out-of-set label onto one of TOP_LEVEL_DOMAINS (e.g. "job_search" -> "job").
# Below this it falls back to "general".
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
        # NULL means "not yet folded into a user_summaries row". Set by
        # summarization.generate_user_summary() in the same transaction as
        # the INSERT into user_summaries — see that function for why.
        "ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS summarized_at TIMESTAMP",
        # Why a node was archived: 'revision' (a fact slot got a new value —
        # the old node is history, see supersede_node()) or 'merge' (a
        # duplicate folded in by resolve_conflicting_nodes()). NULL on active
        # nodes and on nodes archived before this column existed.
        "ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS supersede_reason VARCHAR",
        # TRUE when the node was written from fallback metadata (extraction
        # failed: unsplit whole message and/or default classification).
        # Such nodes skip matching and merging, and are excluded as match/merge
        # candidates, until a backfill pass reprocesses them with a real
        # extraction. See process_memory_intent().
        "ALTER TABLE graph_nodes ADD COLUMN IF NOT EXISTS needs_reprocessing BOOLEAN DEFAULT FALSE",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_needs_reprocessing "
        "ON graph_nodes(user_id) WHERE needs_reprocessing",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_user_id ON graph_nodes(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_domain   ON graph_nodes(domain)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_superseded_by ON graph_nodes(superseded_by)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_last_accessed ON graph_nodes(last_accessed_at)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_summarized_at ON graph_nodes(summarized_at)",
        "CREATE INDEX IF NOT EXISTS idx_graph_edges_source   ON graph_edges(source_node_id)",
        # Rolling per-user digest generated periodically by
        # summarization.generate_user_summary() — see that module. Kept out
        # of graph_nodes entirely: a summary has no domain/status/timeline,
        # and giving it one of those would mean get_graph_context()'s scoring,
        # edge-building, and timeline_summary all need to special-case a
        # synthetic node type.
        """
        CREATE TABLE IF NOT EXISTS user_summaries (
            id                SERIAL PRIMARY KEY,
            user_id           VARCHAR NOT NULL,
            summary_text      TEXT NOT NULL,
            generated_at      TIMESTAMP DEFAULT NOW(),
            source_node_count INTEGER DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_user_summaries_user_id ON user_summaries(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_user_summaries_generated_at ON user_summaries(generated_at)",
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
    Fast, LLM-free way to tag text with one of the fixed TOP_LEVEL_DOMAINS,
    based on which of this user's existing domains it reads most like.

    TWO-TIER MODEL: only the top-level `domain` is classified here, and only
    into TOP_LEVEL_DOMAINS. Nodes whose stored domain is outside that set
    (legacy free-form labels) are ignored when building centroids, so this
    can never return a non-canonical domain. The finer `sub_domain` is
    free-form and not involved here.

    See the CALIBRATION STATUS note on DOMAIN_CLASSIFIER_THRESHOLD above —
    reasonably confident on write-time (atomic-fact) text, still unvalidated
    on query-time (natural-language) text. With coarser top-level buckets
    the centroids are broader, so that threshold may need re-tuning.

    For each of the user's existing top-level domains, computes the centroid (mean)
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
              AND domain = ANY(%s)
            """,
            (userId, TOP_LEVEL_DOMAINS),
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
    allowed = ", ".join(TOP_LEVEL_DOMAINS)
    prompt = (
        "Analyze this text and extract structured metadata.\n"
        "Return ONLY a JSON object with these exact fields:\n"
        "{\n"
        '  "entity_name": "the main subject/entity in this text",\n'
        f'  "domain": "EXACTLY one of: {allowed}",\n'
        '  "sub_domain": "a short, specific free-form label within the domain",\n'
        '  "status": "ongoing or completed",\n'
        '  "importance_score": 0.5,\n'
        '  "related_entities": ["list of other entities mentioned"]\n'
        "}\n\n"
        "Rules for domain and sub_domain:\n"
        f"- domain MUST be one of: {allowed}. Never invent a new domain.\n"
        "- Put anything more specific in sub_domain, not in domain. "
        'e.g. favorite cricket player -> domain "hobbies", sub_domain '
        '"favorite cricket players"; DSA prep for an interview -> domain '
        '"job", sub_domain "interview preparation"; a course or exam -> '
        'domain "education".\n'
        '- Use "general" only if none of the others fit.\n\n'
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
        parsed: Dict[str, Any] = safe_parse_json(_call_fast_llm(prompt))
    except Exception as e:
        logger.warning(
            f"extract_graph_metadata failed for userId={userId}: {e} — using defaults"
        )
        metadata = normalize_graph_metadata({}, text, userId)
        # Defaults are not a real classification — process_memory_intent()
        # must not match or supersede on them.
        metadata["extraction_failed"] = True
        return metadata
    return normalize_graph_metadata(parsed, text, userId)


def normalize_graph_metadata(parsed: Optional[Dict[str, Any]], text: str, userId: str) -> dict:
    """
    Validate raw metadata from an LLM and fill in the rest: resolve the
    domain into TOP_LEVEL_DOMAINS, clamp status/importance, default missing
    fields, stamp the timeline.

    Used by extract_graph_metadata() (its own single-fact LLM call) and by
    process_memory_intent() when the caller already has metadata from
    atomic_extractor.extract_facts_with_metadata(), which classifies all of
    a message's facts in one call. Both paths therefore get identical
    validation. Never raises; `parsed` may be None or {} (→ safe defaults).
    """
    parsed = parsed if isinstance(parsed, dict) else {}
    failed = False
    try:
        domain = str(parsed.get("domain", "general")).lower().strip() or "general"
        # The prompt forces a TOP_LEVEL_DOMAINS value, so a valid answer is
        # used as-is — no extra embedding pass or LLM call. Only when the LLM
        # strays outside the fixed set: try the cheap embedding classifier
        # (which also only returns fixed-set domains), then resolve_domain()
        # to map the stray label onto the set.
        if domain not in TOP_LEVEL_DOMAINS:
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
            f"normalize_graph_metadata failed for userId={userId}: {e} — using defaults"
        )
        failed = True
        domain = "general"
        status = "ongoing"
        importance = 0.5
        related = []
        parsed = {}

    now = datetime.now()
    result = {
        "entity_name": str(parsed.get("entity_name") or text[:60]),
        "domain": domain,
        "sub_domain": str(parsed.get("sub_domain") or ""),
        "status": status,
        "importance_score": importance,
        "related_entities": related,
        "timeline_year": now.year,
        "timeline_month": now.month,
        "timeline_label": now.strftime("%B %Y"),
    }
    if failed:
        result["extraction_failed"] = True
    return result


# ================================================================
# FUNCTION 1B: resolve_domain
# ================================================================

def resolve_domain(candidate_domain: str, userId: str) -> str:
    """
    Map a candidate domain label onto the fixed TOP_LEVEL_DOMAINS set.

    TWO-TIER MODEL: `domain` is one of TOP_LEVEL_DOMAINS (coarse, fixed);
    `sub_domain` is free-form (fine-grained). Previously this function
    merged the candidate against whatever free-form domains the user
    already had, which let the set of top-level domains grow and drift.
    It now always returns a TOP_LEVEL_DOMAINS value:
      1. candidate already in the set -> return it
      2. candidate in _DOMAIN_ALIASES -> return the mapped domain (no LLM)
      3. ask the LLM to pick the best-fitting top-level domain; accept it
         if it's in the set and confidence >= DOMAIN_MATCH_THRESHOLD
      4. otherwise -> "general"
    `userId` is kept for signature compatibility and logging only.

    MIGRATION NOTE (existing non-empty data): nodes and Chroma entries
    written before this change can carry free-form domains ("career",
    "sports", "favorite things", ...). They keep working for display
    (timeline, summaries), but they are effectively invisible to anything
    that filters on the fixed set:
      - classify_domain_by_embedding() skips them when building centroids,
      - find_matching_task_node() only compares nodes with the SAME domain,
        so a new "hobbies" fact can't match/revise a legacy "sports" node,
      - the /memory/retrieve domain filter won't return legacy-tagged
        Chroma entries once the query classifies into a fixed domain.
    To backfill, run a one-off pass in the spirit of
    backfill_legacy_domains.py: for every graph_nodes row whose domain is not
    in TOP_LEVEL_DOMAINS, compute new = resolve_domain(old_domain, user_id).
    Where the old label was more specific than the new domain and sub_domain
    is empty, move the old label into sub_domain so the finer distinction
    isn't lost. UPDATE graph_nodes.domain, then call
    memory_store.update_memory_domain(node_id, new) for the Chroma entry with
    the same id. Resolve each DISTINCT old label once and reuse the result:
    there are far fewer distinct labels than nodes. Afterwards, consider a
    full resolve_conflicting_nodes(userId, limit=None) sweep, since merging
    buckets can expose duplicates that used to sit in different domains.

    Returns the resolved domain (always one of TOP_LEVEL_DOMAINS).
    """
    candidate = candidate_domain.lower().strip() or "general"

    if candidate in TOP_LEVEL_DOMAINS:
        return candidate

    if candidate in _DOMAIN_ALIASES:
        resolved = _DOMAIN_ALIASES[candidate]
        logger.info(f"resolve_domain userId={userId} alias '{candidate}' -> '{resolved}'")
        return resolved

    prompt = (
        "A memory-tagging system must file a label under exactly one fixed "
        "top-level life-area.\n\n"
        f'CANDIDATE LABEL: "{candidate}"\n\n'
        "ALLOWED TOP-LEVEL DOMAINS:\n"
        + "\n".join(f'- "{d}"' for d in TOP_LEVEL_DOMAINS) + "\n\n"
        "Pick the one allowed domain the candidate label belongs under "
        '(e.g. "job_search" -> "job", "gym" -> "health", "cricket" -> '
        '"hobbies"). Use "general" only if nothing else fits.\n\n'
        "Return ONLY a JSON object:\n"
        '{"resolved_domain": "<one of the allowed domains>", "confidence": 0.0}\n'
        "confidence is how sure you are that resolved_domain is correct (1.0 = certain).\n"
        "No explanation. No markdown."
    )

    try:
        raw = ai_service.call_llm(message=prompt, context="", temperature=0, for_chat=False)
        parsed = safe_parse_json(raw)
        resolved = str(parsed.get("resolved_domain", "general")).lower().strip()
        confidence = float(parsed.get("confidence", 0.0))
    except Exception as e:
        logger.warning(
            f"resolve_domain LLM mapping failed userId={userId}: {e} — using 'general'"
        )
        return "general"

    if resolved in TOP_LEVEL_DOMAINS and confidence >= DOMAIN_MATCH_THRESHOLD:
        logger.info(
            f"resolve_domain userId={userId} mapped '{candidate}' -> "
            f"'{resolved}' (confidence={confidence:.2f})"
        )
        return resolved

    logger.info(
        f"resolve_domain userId={userId} could not confidently map '{candidate}' "
        f"(got {resolved!r} confidence={confidence:.2f}) — using 'general'"
    )
    return "general"


# ================================================================
# HELPER: _text_similarities (edge strength input)
# ================================================================

def _text_similarities(text: str, others: List[str]) -> List[Optional[float]]:
    """
    Cosine similarity between `text` and each string in `others`, using the
    same sentence-transformer as the rest of the codebase (embeddings.py).

    Negative cosines are clamped to 0.0 so the result stays in [0, 1] and can
    be blended with the [0, 1] heuristic edge weight.

    Returns one value per input, in order. Any element is None when it can't be
    scored (empty text, zero vector, or an embedding failure). The caller then
    falls back to the heuristic weight alone, so edge creation never fails
    because of embeddings.
    """
    if not others:
        return []
    try:
        base = np.array(embeddings.encode(text))
        base_norm = np.linalg.norm(base)
        vectors = embeddings.encode_batch([o or " " for o in others])
    except Exception as e:
        logger.warning(f"_text_similarities embedding failed: {e} — using heuristic strength only")
        return [None] * len(others)

    results: List[Optional[float]] = []
    for other, vec in zip(others, vectors):
        v = np.array(vec)
        v_norm = np.linalg.norm(v)
        if not other.strip() or base_norm == 0 or v_norm == 0:
            results.append(None)
            continue
        cosine = float(np.dot(base, v) / (base_norm * v_norm))
        results.append(max(0.0, min(1.0, cosine)))
    return results


# ================================================================
# FUNCTION 2: add_to_graph
# ================================================================

def add_to_graph(
    userId: str,
    node_id: str,
    text: str,
    metadata: dict,
    exclude_related_ids: Optional[List[str]] = None,
) -> bool:
    """
    exclude_related_ids: node ids that must not get an edge from this node,
    e.g. the node a revision is about to supersede. superseded_by already
    links the two, and a RELATED_TO edge to it would be noise.
    Insert a new node into graph_nodes and create edges to related
    existing nodes (same domain + same month).

    Relationship type: LED_TO when this node completes something that was
    ongoing, otherwise RELATED_TO. Strength is computed per edge: the text
    cosine similarity blended with the co-occurrence heuristic (see
    EDGE_SIMILARITY_WEIGHT). Every created edge's strength is logged.

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
                importance_score, text, needs_reprocessing, created_at, last_updated
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
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
                bool(metadata.get("needs_reprocessing", False)),
            ),
        )

        # Find existing nodes in the same domain + same calendar month
        cur.execute(
            """
            SELECT id, status, text
            FROM graph_nodes
            WHERE user_id = %s
              AND id <> ALL(%s)
              AND domain = %s
              AND timeline_year = %s
              AND timeline_month = %s
              AND superseded_by IS NULL
            LIMIT 10
            """,
            (
                userId,
                [node_id] + list(exclude_related_ids or []),
                metadata.get("domain", "general"),
                metadata.get("timeline_year"),
                metadata.get("timeline_month"),
            ),
        )
        related_nodes = cur.fetchall()

        similarities = _text_similarities(text, [r["text"] or "" for r in related_nodes])

        new_status = metadata.get("status", "ongoing")
        edges_created = 0
        for related, similarity in zip(related_nodes, similarities):
            # Relationship TYPE is still decided by status; only strength is computed.
            if new_status == "completed" and related["status"] == "ongoing":
                relationship, heuristic = "LED_TO", EDGE_HEURISTIC_LED_TO
            else:
                relationship, heuristic = "RELATED_TO", EDGE_HEURISTIC_RELATED_TO

            if similarity is None:
                strength = heuristic
            else:
                strength = (
                    EDGE_SIMILARITY_WEIGHT * similarity
                    + (1 - EDGE_SIMILARITY_WEIGHT) * heuristic
                )
            strength = round(strength, 4)

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
            if cur.rowcount > 0:
                edges_created += 1
                sim_label = f"{similarity:.4f}" if similarity is not None else "n/a"
                logger.info(
                    f"Edge created userId={userId} {node_id} -[{relationship}]-> {related['id']} "
                    f"strength={strength:.4f} (cosine={sim_label} heuristic={heuristic})"
                )

        conn.commit()
        cur.close()
        logger.info(
            f"Graph node stored userId={userId} nodeId={node_id} "
            f"edges_created={edges_created} candidates={len(related_nodes)}"
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
# FUNCTION 3A: revision history for history-flavoured questions
# ================================================================

HISTORY_CONTEXT_LIMIT = 5


def get_revision_history(
    userId: str, domain: Optional[str] = None, limit: int = HISTORY_CONTEXT_LIMIT
) -> List[dict]:
    """
    The user's revised fact slots, as chains of values: for each slot that
    has been revised at least once, the previous values (oldest first) and
    the current one.

    A scoped companion to get_timeline_summary(): it reads the same archived
    nodes, but follows only supersede_reason='revision' links (merges are
    duplicates, not history). It returns only the `limit` most recently
    changed slots, optionally within one domain, so a history question
    doesn't pull the user's entire history into the prompt.

    Returns [{"domain", "slot", "previous": [value, ...], "current": value,
              "last_changed": datetime}], most recently changed first.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, entity_name, text, domain, sub_domain,
                   superseded_by, supersede_reason, archived_at
            FROM graph_nodes WHERE user_id = %s
            """,
            (userId,),
        )
        rows = {r["id"]: r for r in cur.fetchall()}
        cur.close()
    except Exception as e:
        logger.error(f"get_revision_history failed userId={userId}: {e}")
        return []
    finally:
        _release_connection(conn)

    def label(r):
        return (r["entity_name"] or r["text"] or "").strip()

    def terminal(node_id):
        # Follow superseded_by (any reason) to the node that is active now.
        seen = set()
        while rows.get(node_id) and rows[node_id]["superseded_by"] and node_id not in seen:
            seen.add(node_id)
            node_id = rows[node_id]["superseded_by"]
        return node_id

    chains: Dict[str, List[dict]] = {}
    for r in rows.values():
        if r["supersede_reason"] == "revision" and r["superseded_by"]:
            head = terminal(r["id"])
            if head in rows and not rows[head]["superseded_by"]:
                chains.setdefault(head, []).append(r)

    out = []
    for head_id, previous in chains.items():
        head = rows[head_id]
        if domain and head["domain"] != domain:
            continue
        previous.sort(key=lambda r: r["archived_at"] or datetime.min)
        out.append({
            "domain": head["domain"] or "general",
            "slot": head["sub_domain"] or previous[-1]["sub_domain"] or "",
            "previous": [label(r) for r in previous],
            "current": label(head),
            "last_changed": previous[-1]["archived_at"],
        })
    out.sort(key=lambda c: c["last_changed"] or datetime.min, reverse=True)
    return out[:limit]


def format_history_context(chains: List[dict]) -> str:
    """
    Render revision chains as bullet lines, one per changed fact slot:
      "- Favorite cricket players: previously Rohit Sharma, replaced by Virat Kohli (current)"

    No heading: the caller (Spring's SessionController) labels the section,
    the same way it labels every other context section.
    """
    lines = []
    for c in chains:
        slot = (c["slot"] or c["domain"]).strip()
        slot = slot[:1].upper() + slot[1:]
        lines.append(
            f"- {slot}: previously {', then '.join(c['previous'])}, "
            f"replaced by {c['current']} (current)"
        )
    return "\n".join(lines)


def get_history_context(
    userId: str,
    query_text: Optional[str] = None,
    domain: Optional[str] = None,
    limit: int = HISTORY_CONTEXT_LIMIT,
) -> dict:
    """
    Bounded revision history for a history-flavoured question. Served by
    GET /graph/history/{userId}; Spring calls it only when its
    HistoryQuestionUtil flags the message ("who did I use to like before
    that?"). get_graph_context() stays current-state only.

    Scope, in order:
      1. `domain` if the caller gave one;
      2. else the domain of `query_text`, if classify_domain_by_embedding()
         is confident;
      3. if that finds nothing (or neither applies): the `limit` most
         recently changed slots across all domains. An anaphoric question
         ("…before that?") usually has no domain signal of its own, so
         "recent" beats returning nothing.

    Always bounded by `limit` and by revision chains only — never the full
    timeline, which grows without limit.

    Returns {"domain": <scope used or None>, "chains": [...],
             "history_context_text": <bullet lines, "" if none>}.
    """
    scope = domain
    if not scope and query_text:
        scope = classify_domain_by_embedding(query_text, userId)

    chains = get_revision_history(userId, domain=scope, limit=limit) if scope else []
    if not chains:
        scope = None
        chains = get_revision_history(userId, limit=limit)

    logger.info(
        f"get_history_context userId={userId} domain_scope={scope!r} chains={len(chains)}"
    )
    return {"domain": scope, "chains": chains, "history_context_text": format_history_context(chains)}


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


# ================================================================
# FUNCTION 4B: update_node_content
# ================================================================

def update_node_content(
    userId: str,
    node_id: str,
    new_text: str,
    new_entity_name: str,
    new_metadata: dict,
) -> bool:
    """
    Overwrite an existing node's content in place: text, entity_name,
    domain, sub_domain, importance_score, last_updated.

    Only for an ONGOING TASK mentioned again with new detail ("still
    applying to Amazon, finished the OA") — the same task, so an in-place
    edit is right. A revised FACT ("favorite player is now Virat Kohli") must
    NOT come here: the old value is history, not a mistake, so it goes
    through supersede_node() instead, which keeps both nodes.

    Status is deliberately not changed here.

    The old -> new text is logged on every call so a revision is always
    visible in the logs.

    Returns True if the node was found and updated, False otherwise.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT text FROM graph_nodes WHERE id = %s AND user_id = %s",
            (node_id, userId),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            logger.warning(
                f"update_node_content userId={userId} nodeId={node_id} not found — nothing updated"
            )
            return False
        old_text = row["text"]

        cur.execute(
            """
            UPDATE graph_nodes
            SET text = %s,
                entity_name = %s,
                domain = %s,
                sub_domain = %s,
                importance_score = %s,
                last_updated = NOW()
            WHERE id = %s AND user_id = %s
            """,
            (
                new_text,
                new_entity_name,
                new_metadata.get("domain", "general"),
                new_metadata.get("sub_domain", ""),
                new_metadata.get("importance_score", 0.5),
                node_id,
                userId,
            ),
        )
        updated = cur.rowcount > 0
        conn.commit()
        cur.close()
        logger.info(
            f"update_node_content userId={userId} nodeId={node_id} updated={updated} "
            f"old_text={old_text!r} -> new_text={new_text!r}"
        )
        return updated

    except Exception as e:
        conn.rollback()
        logger.error(f"update_node_content failed userId={userId} nodeId={node_id}: {e}")
        return False
    finally:
        _release_connection(conn)


# ================================================================
# FUNCTION 4C: supersession (shared by revisions and merges)
# ================================================================

_SUPERSEDE_REASONS = ("revision", "merge")


def _mark_superseded(cur, userId: str, old_ids: List[str], new_id: str, reason: str) -> int:
    """
    Archive `old_ids` in favour of `new_id`: superseded_by, archived_at,
    supersede_reason. Nothing is deleted — every "current state" read path
    filters `superseded_by IS NULL`, and get_timeline_summary() keeps
    archived nodes as history.

    The single place this UPDATE lives, used by supersede_node() (reason
    "revision") and resolve_conflicting_nodes() (reason "merge"). Runs on
    the caller's cursor; the caller commits. Already-archived nodes are left
    alone so an existing chain is never re-pointed. Returns rows updated.
    """
    if reason not in _SUPERSEDE_REASONS:
        raise ValueError(f"unknown supersede reason {reason!r}")
    cur.execute(
        """
        UPDATE graph_nodes
        SET superseded_by = %s, archived_at = NOW(), supersede_reason = %s
        WHERE id = ANY(%s) AND user_id = %s AND superseded_by IS NULL
        """,
        (new_id, reason, old_ids, userId),
    )
    return cur.rowcount


def supersede_node(userId: str, old_node_id: str, new_node_id: str) -> Optional[Dict[str, str]]:
    """
    Record that `new_node_id` holds the NEW value of the fact slot that
    `old_node_id` held (a "revision" match): archive the old node with
    supersede_reason='revision', keeping it as history.

    `new_node_id` must already exist (superseded_by is a foreign key), so
    the caller creates the new node first.

    Returns the old node's {"node_id", "text", "entity_name"} so the caller
    can report what was replaced, or None if the old node wasn't found or was
    already archived.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT text, entity_name FROM graph_nodes WHERE id = %s AND user_id = %s",
            (old_node_id, userId),
        )
        row = cur.fetchone()
        if not row:
            cur.close()
            logger.warning(f"supersede_node userId={userId} old nodeId={old_node_id} not found")
            return None

        updated = _mark_superseded(cur, userId, [old_node_id], new_node_id, "revision")
        conn.commit()
        cur.close()
        if not updated:
            logger.warning(
                f"supersede_node userId={userId} old nodeId={old_node_id} was already archived"
            )
            return None

        logger.info(
            f"supersede_node userId={userId} {old_node_id} -> {new_node_id} "
            f"old_text={row['text']!r}"
        )
        return {"node_id": old_node_id, "text": row["text"] or "", "entity_name": row["entity_name"] or ""}

    except Exception as e:
        conn.rollback()
        logger.error(f"supersede_node failed userId={userId} old nodeId={old_node_id}: {e}")
        return None
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

_MATCH_TYPES = ("progression", "revision")


def find_matching_task_node(userId: str, text: str, domain: str) -> Optional[Dict[str, Any]]:
    """
    Search this user's existing "ongoing" nodes in the same domain for one
    that `text` is an update to — not merely one that shares the same
    domain/life-area. Two kinds of update are recognised:

      - "progression": the same TASK reported at a later stage ("applying
        for the Amazon job" -> "got the Amazon offer"). The node's status
        may change (ongoing -> completed); its content is the task itself.
      - "revision": the same single-slot FACT with a new value ("favorite
        player is Rohit Sharma" -> "favorite player is now Virat Kohli";
        also job title, phone number, address). The old value is replaced,
        never kept side by side, and status stays "ongoing".

    The caller (process_memory_intent) uses match_type to choose between
    update_node_status() and update_node_content().

    Candidates whose entity_name is a generic pronoun/reference (e.g. "the
    user", "the speaker's task") are excluded before they ever reach the
    LLM — see _is_generic_entity_name().

    Returns {"node_id", "match_type", "confidence", "sub_domain" (the matched
    node's)}, or None if nothing
    scores above TASK_MATCH_THRESHOLD (including when there are no eligible
    candidates). A missing/unknown match_type from the LLM defaults to
    "progression", the pre-revision-support behaviour.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, text, entity_name, sub_domain
            FROM graph_nodes
            WHERE user_id = %s AND domain = %s AND status = 'ongoing'
              AND superseded_by IS NULL
              AND needs_reprocessing IS NOT TRUE
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
        "tasks/facts to detect if it's an update to the SAME specific task "
        "or the SAME specific fact-slot — not just something from the same "
        "general life-area (domain).\n\n"
        "There are TWO kinds of match:\n"
        '1. "progression" — the SAME TASK reported at a different stage. '
        "The task moves forward (ongoing -> completed).\n"
        '   e.g. "applying for the Amazon job" -> "got the Amazon offer"\n'
        '2. "revision" — the SAME single-value FACT SLOT with a NEW value. '
        "The old value is replaced, not progressed. Typical slots: favorite "
        "X, current job title, employer, phone number, address, city of "
        "residence.\n"
        '   e.g. "favorite player is Rohit Sharma" -> "favorite player is '
        'now Virat Kohli"\n'
        '   e.g. "works as a support engineer" -> "was promoted to senior '
        'developer"\n\n'
        "IMPORTANT: sharing a domain/category is NOT sufficient for a match. "
        "The slot must be the same. A NEW MESSAGE that leaves out a qualifier "
        'still refers to the existing slot ("favorite player is now X" after '
        '"favorite cricket player is Y" is a revision of the cricket slot). '
        "Only a DIFFERENT explicit qualifier (football vs cricket) makes it a "
        "different slot.\n\n"
        "Examples:\n"
        '- MATCH (progression): "applying for the AWS job" and "got the '
        'AWS offer" — same task, later stage.\n'
        '- MATCH (revision): "favorite cricket player is Rohit Sharma" and '
        '"favorite player is now Virat Kohli" — same slot, new value.\n'
        '- NO MATCH: "favorite cricket player is Virat Kohli" and "favorite '
        'football player is Messi" — different slots (cricket vs football); '
        "both values are true at the same time.\n"
        '- NO MATCH: "I\'m a software engineer at Yardi" and "update my '
        'resume before Friday" — same domain (job), but a standing fact and '
        "an unrelated task.\n"
        '- NO MATCH: "what are my pending reminders" and "remind me to '
        'deploy to AWS" — a question is never a match target.\n\n'
        f'NEW MESSAGE: "{text}"\n\n'
        "EXISTING ITEMS (id: text [sub_domain]). An item's slot is defined by "
        "its text AND its sub_domain — a revision can drop a qualifier from "
        'the text (e.g. "cricket") that the sub_domain still records:\n'
        + "\n".join(
            f'- {row["id"]}: "{row["text"] or row["entity_name"]}" '
            f'[{row["sub_domain"] or "-"}]'
            for row in candidates
        ) + "\n\n"
        "If the NEW MESSAGE is a progression of an existing task or a "
        "revision of an existing fact slot (even if worded very "
        "differently), return that item's id and the match_type. Otherwise "
        "— including when it's merely in the same domain — return "
        "matched_id as null.\n\n"
        'Return ONLY JSON: {"matched_id": "<id or null>", '
        '"match_type": "progression" | "revision" | null, "confidence": 0.0, '
        '"reason": "<one sentence explaining the decision>"}\n'
        "confidence is how sure you are of the match (1.0 = certain).\n"
        "No explanation outside the JSON. No markdown."
    )

    try:
        raw = ai_service.call_llm(message=prompt, context="", temperature=0, for_chat=False)
        parsed = safe_parse_json(raw)
        matched_id = parsed.get("matched_id")
        match_type = str(parsed.get("match_type") or "").lower().strip()
        confidence = float(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", "")).strip()
    except Exception as e:
        logger.warning(f"find_matching_task_node LLM match failed userId={userId}: {e}")
        return None

    if match_type not in _MATCH_TYPES:
        match_type = "progression"

    by_id = {row["id"]: row for row in candidates}
    if matched_id in by_id and confidence >= TASK_MATCH_THRESHOLD:
        logger.info(
            f"find_matching_task_node userId={userId} matched nodeId={matched_id} "
            f"match_type={match_type} (confidence={confidence:.2f}) reason={reason!r}"
        )
        return {
            "node_id": matched_id,
            "match_type": match_type,
            "confidence": confidence,
            "sub_domain": by_id[matched_id]["sub_domain"] or "",
        }

    logger.info(
        f"find_matching_task_node userId={userId} no match accepted "
        f"(llm matched_id={matched_id!r} match_type={match_type} confidence={confidence:.2f} "
        f"threshold={TASK_MATCH_THRESHOLD}) reason={reason!r}"
    )
    return None


# ================================================================
# FUNCTION 5B: process_memory_intent
# ================================================================

def process_memory_intent(
    userId: str,
    text: str,
    memoryId: Optional[str] = None,
    extracted_metadata: Optional[Dict[str, Any]] = None,
    extraction_fallback: bool = False,
) -> dict:
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
      - the text reports completion of an existing "ongoing" task
        (match_type="progression", status="completed") → update_node_status()
        to "completed"; the task's content is unchanged.
      - the text revises an existing fact slot (match_type="revision") →
        a NEW node is created for the new value, and the matched node is
        archived as history via supersede_node() (superseded_by -> new node,
        supersede_reason='revision'). Both nodes remain. Current-state reads
        see only the new one; get_timeline_summary() shows the chain.
      - the text re-mentions an ongoing TASK with new detail
        (match_type="progression", status="ongoing") → update_node_content()
        edits that task node in place.
      - otherwise → create a new node via add_to_graph(), as before.

    `extracted_metadata`, when provided, is this fact's raw classification
    from atomic_extractor.extract_facts_with_metadata(), which split and
    classified the whole message in one call (so sibling facts were judged
    together). It is validated by normalize_graph_metadata() and no second
    LLM call is made. When None (e.g. /graph/process, or the extractor fell
    back), extract_graph_metadata() classifies the fact on its own.

    `extraction_fallback` is True when atomic_extractor fell back to the
    whole, unsplit message. That, or a failed per-fact classification
    (extract_graph_metadata() returned defaults), is DEGRADED input:
    matching and supersession are skipped entirely and a plain node is
    created with needs_reprocessing=TRUE, so a failed extraction can never
    archive or edit an existing node. A backfill can find these via the
    needs_reprocessing index and re-run them once extraction works.

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
            "action": "skipped" | "updated" | "superseded" | "created",
            "degraded": bool,          # True → written from fallback metadata, no matching done
            "node_id": Optional[str],  # the node now holding this fact; None when skipped
            "match_type": "progression" | "revision" | None,
            "metadata": dict,          # {} when action == "skipped"
            "success": bool,           # whether the underlying DB write succeeded
            # only when action == "superseded" — enough for a caller to say
            # "Updated: you were a fan of Rohit Sharma, now it's Virat Kohli":
            "supersession": {
                "old": {"node_id", "text", "entity_name"},
                "new": {"node_id", "text", "entity_name"},
            },
        }
    """
    if not is_declarative(text):
        logger.info(
            f"process_memory_intent userId={userId} skipped — text is a "
            f"question, not a fact/task: {text[:80]!r}"
        )
        return {"action": "skipped", "node_id": None, "match_type": None, "metadata": {}, "success": True, "degraded": False}

    if extracted_metadata and not extraction_fallback:
        metadata = normalize_graph_metadata(extracted_metadata, text, userId)
    else:
        metadata = extract_graph_metadata(text, userId)
    domain = metadata.get("domain", "general")
    status = metadata.get("status", "ongoing")
    node_id = memoryId or f"{userId}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

    # FALLBACK GUARD: never match, supersede or progress on degraded input.
    # Matching an unsplit compound message or default "general" metadata
    # against real nodes archived the wrong node in testing (a football fact
    # superseded the cricket one). A plain, flagged node loses nothing.
    degraded = extraction_fallback or bool(metadata.pop("extraction_failed", False))
    if degraded:
        metadata["needs_reprocessing"] = True
        success = add_to_graph(userId=userId, node_id=node_id, text=text, metadata=metadata)
        logger.warning(
            f"process_memory_intent userId={userId} DEGRADED extraction "
            f"(extractor_fallback={extraction_fallback}) — created nodeId={node_id} "
            f"with needs_reprocessing=TRUE, no matching. text={text[:80]!r}"
        )
        return {"action": "created", "node_id": node_id, "match_type": None,
                "metadata": metadata, "success": success, "degraded": True}

    match = find_matching_task_node(userId, text, domain)

    if match and match["match_type"] == "revision":
        # A fact slot got a new value. The old value is history, not an
        # error: keep it, archived, and put the new value in its own node.
        # New node first — superseded_by is a foreign key to it.
        old_id = match["node_id"]
        # A revision is the SAME slot by definition, so the new node keeps the
        # slot identity the old one recorded. The revising message is often
        # vaguer ("my favorite player is now Kohli" drops "cricket"); without
        # this the qualifier is lost, and a later "favorite football player"
        # fact looks like a revision of the now-unqualified slot.
        if match.get("sub_domain"):
            if metadata.get("sub_domain") and metadata["sub_domain"] != match["sub_domain"]:
                logger.info(
                    f"process_memory_intent userId={userId} revision keeps slot "
                    f"sub_domain={match['sub_domain']!r} (message gave {metadata['sub_domain']!r})"
                )
            metadata["sub_domain"] = match["sub_domain"]
        created = add_to_graph(
            userId=userId, node_id=node_id, text=text, metadata=metadata,
            exclude_related_ids=[old_id],
        )
        old = supersede_node(userId, old_id, node_id) if created else None
        success = created and old is not None
        new_info = {
            "node_id": node_id,
            "text": text,
            "entity_name": metadata.get("entity_name", text[:60]),
        }
        logger.info(
            f"process_memory_intent userId={userId} revision: {old_id} superseded by "
            f"{node_id} success={success} old_text={(old or {}).get('text')!r} new_text={text!r}"
        )
        if not success:
            # The new node may exist without the old one archived; report it as
            # a plain create so callers don't claim a supersession that didn't
            # happen.
            return {"action": "created" if created else "failed", "node_id": node_id if created else None,
                    "match_type": "revision", "metadata": metadata, "success": created, "degraded": False}
        return {
            "action": "superseded",
            "node_id": node_id,
            "match_type": "revision",
            "metadata": metadata,
            "success": True,
            "degraded": False,
            "supersession": {"old": old, "new": new_info},
        }

    if match:
        matched_id = match["node_id"]
        if status == "completed":
            # Task finished — its content (the task) is unchanged, only its
            # status moves forward.
            success = update_node_status(userId, matched_id, "completed")
        else:
            # The same ongoing task mentioned again with new detail — an
            # in-place edit of the task node is right here.
            success = update_node_content(
                userId,
                matched_id,
                new_text=text,
                new_entity_name=metadata.get("entity_name", text[:60]),
                new_metadata=metadata,
            )

        logger.info(
            f"process_memory_intent userId={userId} matched nodeId={matched_id} "
            f"match_type=progression status={status} success={success}"
        )
        return {
            "action": "updated",
            "node_id": matched_id,
            "match_type": "progression",
            "metadata": metadata,
            "success": success,
            "degraded": False,
        }

    if status == "completed":
        logger.warning(
            f"process_memory_intent userId={userId} reported a completion with no "
            f"matching prior ongoing task — creating a new node. text={text[:80]!r}"
        )

    success = add_to_graph(userId=userId, node_id=node_id, text=text, metadata=metadata)
    return {"action": "created", "node_id": node_id, "match_type": None, "metadata": metadata,
            "success": success, "degraded": False}


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
        "the same group). A single-value fact slot with an old and a new "
        "value is also one group (e.g. \"favorite cricket player is Rohit "
        "Sharma\" and \"favorite cricket player is Virat Kohli\").\n\n"
        "Do NOT group nodes that are merely in the same area but describe "
        "DIFFERENT slots, tasks, or entities that can all be true at once. "
        "For example, \"favorite cricket player is Virat Kohli\", \"prefers "
        "Messi for football\" and \"prefers Federer for tennis\" are three "
        "separate facts. Grouping them would delete true information. When "
        "in doubt, leave nodes ungrouped.\n\n"
        "Nodes:\n"
        f"{listing}\n\n"
        "Return ONLY a JSON object:\n"
        '{"clusters": [["id1", "id2"], ["id3", "id4", "id5"]]}\n'
        "Only include clusters with 2 or more ids — omit any node that "
        "doesn't duplicate/conflict with anything else. "
        "No explanation. No markdown."
    )

    try:
        raw = ai_service.call_llm(message=prompt, context="", for_chat=False)
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
                  AND needs_reprocessing IS NOT TRUE
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

                _mark_superseded(cur, userId, other_ids, active["id"], "merge")
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

def get_timeline_summary(userId: str, domain: Optional[str] = None) -> dict:
    """
    Return all memories grouped by month → domain → ongoing/completed.

    Deliberately includes archived nodes (superseded_by IS NOT NULL) —
    unlike get_graph_context(), which filters them out because it represents
    "current state". This function represents full history, so an archived
    node's original month/status stays visible (e.g. "this task started in
    June, was consolidated into a later node in July") instead of vanishing.

    Archived entries are labelled with what replaced them, so a revised fact
    reads as history rather than as a duplicate next to its successor:
      - supersede_reason 'revision' → "Rohit Sharma (replaced by Virat Kohli)"
      - supersede_reason 'merge'    → "<name> (merged into <name>)"
      - archived before supersede_reason existed → "<name> (archived, see <name>)"

    `domain`, if given, limits the result to that one domain. The full
    timeline grows without limit; this is a view of complete history (for the
    /graph/timeline endpoint or a UI), never chat context — chat uses the
    bounded get_history_context().

    Example: {"June 2026": {"job": {"ongoing": [...], "completed": [...]}}}
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT g.timeline_label, g.timeline_year, g.timeline_month,
                   g.domain, g.status, g.entity_name, g.text,
                   g.superseded_by, g.supersede_reason,
                   s.entity_name AS successor_entity, s.text AS successor_text
            FROM graph_nodes g
            LEFT JOIN graph_nodes s ON s.id = g.superseded_by
            WHERE g.user_id = %s AND (%s::varchar IS NULL OR g.domain = %s)
            ORDER BY g.timeline_year DESC, g.timeline_month DESC, g.domain ASC, g.created_at ASC
            """,
            (userId, domain, domain),
        )
        rows = cur.fetchall()
        cur.close()

        result: Dict[str, Any] = {}
        for row in rows:
            label = row["timeline_label"] or "Unknown"
            domain = row["domain"] or "general"
            status = row["status"] or "ongoing"
            name = row["entity_name"] or (row["text"] or "")[:60]
            if row["superseded_by"]:
                successor = row["successor_entity"] or (row["successor_text"] or "")[:60]
                how = {"revision": "replaced by", "merge": "merged into"}.get(
                    row["supersede_reason"], "archived, see"
                )
                name = f"{name} ({how} {successor})"

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
