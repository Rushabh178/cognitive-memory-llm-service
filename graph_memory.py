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
import logging
from datetime import datetime
from typing import Dict, List, Any

import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor

import config
import ai_service

logger = logging.getLogger(__name__)

VALID_DOMAINS = {
    "sport", "movie", "study", "job", "medical",
    "finance", "relationship", "travel", "food", "general",
}


# ================================================================
# DATABASE CONNECTION
# ================================================================

def _get_connection():
    """
    Open a fresh PostgreSQL connection using config values.

    For production scale, replace with psycopg2.pool.ThreadedConnectionPool
    — opening a new connection per request adds ~5 ms overhead but is
    safe and simple for development/MVP.
    """
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )


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
            last_updated    TIMESTAMP DEFAULT NOW()
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
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_user_id ON graph_nodes(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_graph_nodes_domain   ON graph_nodes(domain)",
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
        conn.close()


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
        '  "domain": "one of: sport, movie, study, job, medical, finance, relationship, travel, food, general",\n'
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
        raw = ai_service.call_llm(message=prompt, context="")
        parsed: Dict[str, Any] = safe_parse_json(raw)

        domain = str(parsed.get("domain", "general")).lower()
        if domain not in VALID_DOMAINS:
            domain = "general"

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
        conn.close()


# ================================================================
# FUNCTION 3: get_graph_context
# ================================================================

def get_graph_context(userId: str, query_text: str, topN: int = 5) -> dict:
    """
    Score and retrieve the most relevant graph nodes for a query.

    Scoring: importance * 0.4 + recency * 0.3 + keyword_match * 0.3

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
            WHERE user_id = %s
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
            keyword_match = 1.0 if any(
                word in searchable
                for word in query_lower.split()
                if len(word) > 2
            ) else 0.0

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
                # kept for context-text building; Pydantic ignores extra fields
                "_timeline_year": node_year,
                "_timeline_month": node_month,
            }))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [n for _, n in scored[:topN]]
        top_node_ids = [n["node_id"] for n in top_nodes]

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

        # ── Build timeline summary ─────────────────────────────────
        cur.execute(
            """
            SELECT timeline_year, timeline_month, domain, status, COUNT(*) AS cnt
            FROM graph_nodes
            WHERE user_id = %s
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
        conn.close()


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
        conn.close()


# ================================================================
# FUNCTION 5: get_timeline_summary
# ================================================================

def get_timeline_summary(userId: str) -> dict:
    """
    Return all memories grouped by month → domain → ongoing/completed.

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
        conn.close()
