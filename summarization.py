# ============================================================
# FILE: summarization.py
#
# WHAT THIS FILE DOES:
# Periodically consolidates a user's raw graph_nodes into a compact,
# rolling digest (user_summaries) — separate from the per-message write
# path in graph_memory.py. Never runs inline with a chat request; it's
# only ever triggered by scheduler.py during a detected low-load window.
#
# WHERE IT FITS IN THE PIPELINE:
# graph_nodes (summarized_at IS NULL) → generate_user_summary() → LLM →
# new user_summaries row + graph_nodes.summarized_at stamped on the nodes
# that went into it
# ============================================================

import logging
from typing import List, Optional

import config
from ai_service import _client as _groq_client
from graph_memory import _get_connection, _release_connection

logger = logging.getLogger(__name__)

# Cap on how many un-summarized nodes a single run folds in — bounds prompt
# size and keeps one run fast even if a user has a large backlog. A backlog
# bigger than this just drains over multiple scheduler ticks; oldest nodes
# go first (see the ORDER BY in generate_user_summary()) so it drains in order.
MAX_NODES_PER_RUN = 50


def get_users_with_pending_nodes() -> List[str]:
    """
    Return userIds that have at least one graph_node not yet folded into a
    summary (summarized_at IS NULL) and not archived (superseded_by IS NULL).

    Called once per scheduler tick to know which users are even candidates —
    cheap enough to run every tick since it's a single indexed query.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT user_id FROM graph_nodes
            WHERE summarized_at IS NULL AND superseded_by IS NULL
            """
        )
        return [row["user_id"] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"get_users_with_pending_nodes failed: {e}")
        return []
    finally:
        _release_connection(conn)


def get_latest_summary(userId: str) -> Optional[dict]:
    """
    Return the most recently generated summary for a user, or None if
    generate_user_summary() has never produced one for them yet.

    Used by the /ai/chat context-assembly path to fold the rolling digest in
    as one of the long-term-memory layers, alongside ChromaDB retrieval and
    graph_context_text.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT summary_text, generated_at, source_node_count
            FROM user_summaries
            WHERE user_id = %s
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (userId,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"get_latest_summary failed userId={userId}: {e}")
        return None
    finally:
        _release_connection(conn)


def _build_summary_prompt(nodes: List[dict]) -> str:
    listing = "\n".join(
        f'- [{n["status"]}] {n["domain"]}: {n["entity_name"] or ""} '
        f'({n["timeline_label"] or ""}) — {(n["text"] or "")[:200]}'
        for n in nodes
    )
    return (
        "You are consolidating a user's memory graph into a compact digest "
        "for use as background context in future conversations.\n\n"
        "Below is a list of memory nodes (structured facts extracted from "
        "past messages). Produce a compact, high-level summary organized "
        "into exactly these three sections, using short bullet points:\n\n"
        "Active goals:\n"
        "Recent completions:\n"
        "Standing facts:\n\n"
        "Rules:\n"
        '- "Active goals" = ongoing tasks/aspirations (status=ongoing entries '
        "that read as a goal, not a passive fact).\n"
        '- "Recent completions" = status=completed entries.\n'
        '- "Standing facts" = durable facts about the person that aren\'t '
        "tasks (identity, role, preferences, relationships).\n"
        "- Merge duplicates/near-duplicates into one bullet. Be concise — "
        "this is a digest, not a transcript.\n"
        '- If a section has nothing to include, write "None." under it — '
        "don't omit the section header.\n\n"
        "Memory nodes:\n"
        f"{listing}\n\n"
        "Return ONLY the three-section summary as plain text. No JSON, no "
        "markdown code fences, no preamble."
    )


def generate_user_summary(userId: str) -> Optional[dict]:
    """
    Fold this user's un-summarized graph_nodes into a new user_summaries row.

    Pulls graph_nodes only, not ChromaDB. graph_nodes.text is the same
    content as the corresponding ChromaDB memory for every user-role fact,
    and graph_nodes additionally carries the domain/status/importance
    structure this summary is organized around — re-pulling ChromaDB would
    mostly re-feed the same facts a second time for no benefit.

    Idempotent: only ever selects nodes where summarized_at IS NULL, so a
    skipped or interrupted run just leaves work for the next one — nothing
    is double-processed or lost. The INSERT into user_summaries and the
    UPDATE stamping summarized_at happen in the same transaction, so a crash
    between them can't leave a summary that silently forgot which nodes it
    covered (either both land, or neither does and the nodes stay pending).

    Returns the newly created summary row as a dict, or None if there was
    nothing to summarize — the common case once a user's backlog is drained,
    not an error.
    """
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, entity_name, domain, status, timeline_label, text
            FROM graph_nodes
            WHERE user_id = %s AND summarized_at IS NULL AND superseded_by IS NULL
            ORDER BY created_at ASC
            LIMIT %s
            """,
            (userId, MAX_NODES_PER_RUN),
        )
        nodes = cur.fetchall()
        cur.close()

        if not nodes:
            logger.info(f"generate_user_summary userId={userId} — nothing to summarize")
            return None

        prompt = _build_summary_prompt(nodes)
        response = _groq_client.chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise, concise memory-summarization assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        summary_text = response.choices[0].message.content.strip()

        node_ids = [n["id"] for n in nodes]
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_summaries (user_id, summary_text, source_node_count)
            VALUES (%s, %s, %s)
            RETURNING id, user_id, summary_text, generated_at, source_node_count
            """,
            (userId, summary_text, len(node_ids)),
        )
        summary_row = dict(cur.fetchone())

        cur.execute(
            "UPDATE graph_nodes SET summarized_at = NOW() WHERE id = ANY(%s)",
            (node_ids,),
        )
        conn.commit()
        cur.close()

        logger.info(
            f"generate_user_summary userId={userId} complete — "
            f"summarized {len(node_ids)} nodes, summaryId={summary_row['id']}, "
            f"preview={summary_text[:120]!r}"
        )
        return summary_row

    except Exception as e:
        conn.rollback()
        logger.error(f"generate_user_summary failed userId={userId}: {e}")
        return None
    finally:
        _release_connection(conn)
