# ============================================================
# FILE: tests/test_fact_regressions.py
#
# Live regression tests for fact extraction and fact supersession.
# They call the real Groq API and write to the real PostgreSQL graph tables
# (under a throwaway user id that is always deleted afterwards), so they cost
# a few LLM calls and need a working .env. They skip if either is missing.
#
# Run either way (cwd = cognitive-memory-llm-service):
#   python tests/test_fact_regressions.py
#   pytest tests/test_fact_regressions.py        (if pytest is installed)
#
# REGRESSION_RUNS (default 3) repeats the extraction test, since LLM output
# varies between calls — the original bug appeared in about 1 of 5 runs.
# ============================================================

import os
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Never touch the real vector store: memory_store opens ChromaDB at import time
# (some tests import main), and ChromaDB must not be written by two processes
# while the service is running. The env var wins over .env (load_dotenv doesn't
# override), so this must run before any service module is imported.
os.environ["CHROMA_PERSIST_PATH"] = tempfile.mkdtemp(prefix="cm_test_chroma_")

COMPOUND_MESSAGE = "For football my favorite is Messi and for tennis it's Federer"


class Skip(Exception):
    pass


def _deps(require_llm: bool = True):
    """Import the service modules, or skip if config/DB (and optionally the LLM) aren't available."""
    try:
        import graph_memory
        from atomic_extractor import extract_facts_with_metadata, is_fact_worth_storing
    except Exception as e:  # config.py raises when GROQ_API_KEY / API_BEARER_TOKEN are unset
        raise Skip(f"service config not available: {e}")
    try:
        graph_memory.initialize_tables()
    except Exception as e:
        raise Skip(f"PostgreSQL not reachable: {e}")
    if require_llm:
        _require_llm_quota()
    return graph_memory, extract_facts_with_metadata, is_fact_worth_storing


def _require_llm_quota():
    """
    The extractor swallows API errors and falls back to the unsplit message,
    which would surface here as a misleading FAIL. A rate-limited or
    unreachable API isn't a regression, so skip instead.

    Groq checks its tokens-per-day limit against prompt + max_tokens
    REQUESTED, so the probe requests as much as a real extraction call
    (max_tokens=2000). It is charged only for the few tokens it uses.
    """
    import config
    from ai_service import _client
    try:
        _client.chat.completions.create(
            model=config.FAST_LLM_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        )
    except Exception as e:
        raise Skip(f"FAST_LLM_MODEL {config.FAST_LLM_MODEL!r} unavailable: {str(e)[:160]}")


def _store(gm, extract, worth, user_id, message):
    """What POST /memory/store + the background task do to the graph, per fact."""
    results = []
    for item in extract(message):
        if not worth(item["text"]):
            continue
        node_id = f"{user_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        results.append(gm.process_memory_intent(user_id, item["text"], node_id, item["metadata"]))
    return results


def _nodes(gm, user_id):
    conn = gm._get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, status, text, entity_name, superseded_by, supersede_reason "
            "FROM graph_nodes WHERE user_id = %s ORDER BY created_at",
            (user_id,),
        )
        return cur.fetchall()
    finally:
        gm._release_connection(conn)


def test_compound_preferences_all_classified_ongoing():
    """Regression: the tennis half was classified "completed" when judged alone."""
    _, extract, _ = _deps()
    runs = int(os.getenv("REGRESSION_RUNS", "3"))
    for run in range(1, runs + 1):
        items = extract(COMPOUND_MESSAGE)
        facts = [i["text"] for i in items]
        statuses = [(i["metadata"] or {}).get("status") for i in items]
        assert len(items) >= 2, f"run {run}: expected 2 facts, got {facts}"
        assert all(i["metadata"] for i in items), f"run {run}: extractor fell back: {facts}"
        assert all(s == "ongoing" for s in statuses), f"run {run}: {list(zip(facts, statuses))}"
        joined = " ".join(facts).lower()
        assert "messi" in joined and "federer" in joined, f"run {run}: {facts}"


def test_revision_supersedes_and_keeps_history():
    """Rohit -> Kohli: both nodes kept; the old one archived as a revision."""
    gm, extract, worth = _deps()
    user_id = f"regress_{uuid.uuid4().hex[:10]}"
    try:
        first = _store(gm, extract, worth, user_id, "My favorite cricket player is Rohit Sharma")
        second = _store(gm, extract, worth, user_id, "My favorite player is now Virat Kohli")
        assert [r["action"] for r in first] == ["created"], first
        assert [r["action"] for r in second] == ["superseded"], second

        sup = second[0]["supersession"]
        assert "rohit" in sup["old"]["text"].lower() and "kohli" in sup["new"]["text"].lower(), sup

        nodes = {n["id"]: n for n in _nodes(gm, user_id)}
        old, new = nodes[first[0]["node_id"]], nodes[second[0]["node_id"]]
        assert old["superseded_by"] == new["id"] and old["supersede_reason"] == "revision", old
        assert new["superseded_by"] is None, new

        timeline = str(gm.get_timeline_summary(user_id))
        assert "replaced by" in timeline, timeline
    finally:
        gm.delete_user_graph(user_id)


def test_other_slot_does_not_supersede_revised_fact():
    """
    Regression: "favorite player is now Kohli" doesn't say "cricket". The
    revision node used to lose the qualifier, so a later football fact was
    matched as a revision of it and archived the cricket preference.
    """
    gm, extract, worth = _deps()
    user_id = f"regress_{uuid.uuid4().hex[:10]}"
    try:
        _store(gm, extract, worth, user_id, "My favorite cricket player is Rohit Sharma")
        revised = _store(gm, extract, worth, user_id, "My favorite player is now Virat Kohli")
        football = _store(gm, extract, worth, user_id, "My favorite football player is Ronaldo")

        assert [r["action"] for r in revised] == ["superseded"], revised
        assert [r["action"] for r in football] == ["created"], football

        active = [n["text"].lower() for n in _nodes(gm, user_id) if not n["superseded_by"]]
        assert any("kohli" in t for t in active), active
        assert any("ronaldo" in t for t in active), active
    finally:
        gm.delete_user_graph(user_id)


def test_failed_extraction_never_archives_existing_nodes():
    """
    Regression: while Groq returned 429s, facts fell back to unsplit text with
    default "general" metadata, and matching on that data archived real
    nodes. With extraction forced to fail, the fact must become a plain,
    flagged node; no existing node may be archived; the matcher and the
    merge pass must not run. Needs no LLM quota — both LLM calls are forced
    to raise.
    """
    import asyncio
    import atomic_extractor
    import main

    gm, extract, _ = _deps(require_llm=False)
    user_id = f"regress_{uuid.uuid4().hex[:10]}"
    calls = {"matcher": 0, "merge": 0}

    def rate_limited(*_a, **_kw):
        raise RuntimeError("simulated 429: rate limit reached")

    def spy_matcher(*_a, **_kw):
        calls["matcher"] += 1
        return None

    async def spy_merge(*_a, **_kw):
        calls["merge"] += 1
        return {}

    saved = (atomic_extractor._client.chat.completions.create, gm._call_fast_llm,
             gm.find_matching_task_node, gm.resolve_conflicting_nodes)
    try:
        # Two real, active nodes in the slot the failed fact would have hit.
        for text, sub in (("User's favorite cricket player is Rohit Sharma", "favorite cricket players"),
                          ("User's favorite football player is Ronaldo", "favorite football players")):
            ok = gm.add_to_graph(user_id, f"{user_id}_{uuid.uuid4().hex[:8]}", text, {
                "entity_name": text.split(" is ")[-1], "domain": "hobbies", "sub_domain": sub,
                "status": "ongoing", "importance_score": 0.3,
                "timeline_year": 2026, "timeline_month": 9, "timeline_label": "September 2026",
            })
            assert ok, "seeding failed"

        atomic_extractor._client.chat.completions.create = rate_limited
        gm._call_fast_llm = rate_limited
        gm.find_matching_task_node = spy_matcher
        gm.resolve_conflicting_nodes = spy_merge

        message = "My favorite player is now Virat Kohli"
        items = extract(message)
        assert len(items) == 1 and items[0]["fallback"] is True and items[0]["text"] == message, items

        node_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
        asyncio.run(main.process_graph_background(
            user_id, items[0]["text"], node_id, items[0]["metadata"], items[0]["fallback"]
        ))

        nodes = {n["id"]: n for n in _nodes(gm, user_id)}
        assert node_id in nodes, "degraded fact was not stored at all"
        assert all(n["superseded_by"] is None for n in nodes.values()), \
            [(n["text"], n["superseded_by"]) for n in nodes.values()]
        assert calls == {"matcher": 0, "merge": 0}, calls

        conn = gm._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT needs_reprocessing, domain FROM graph_nodes WHERE id = %s", (node_id,))
            row = cur.fetchone()
        finally:
            gm._release_connection(conn)
        assert row["needs_reprocessing"] is True, row
    finally:
        (atomic_extractor._client.chat.completions.create, gm._call_fast_llm,
         gm.find_matching_task_node, gm.resolve_conflicting_nodes) = saved
        gm.delete_user_graph(user_id)


def test_extraction_failure_never_stores_questions():
    """
    Regression: when extraction failed, /memory/store saved the whole raw message
    to ChromaDB — including a pure question ("Who did I use to like before that?").
    It was later retrieved as a past "fact" with no answer, and the model called it
    an unanswered "past crush". With extraction forced to fail (no LLM quota
    needed): a question stores nothing; a mixed message keeps only its statement;
    a plain statement is stored unchanged. Goes through the real endpoint.
    """
    import atomic_extractor
    import config
    import main
    import memory_store
    from fastapi.testclient import TestClient

    gm, _, _ = _deps(require_llm=False)
    user_id = f"regress_{uuid.uuid4().hex[:10]}"
    auth = {"Authorization": f"Bearer {config.API_BEARER_TOKEN}"}

    def rate_limited(*_a, **_kw):
        raise RuntimeError("simulated 429: rate limit reached")

    saved = (atomic_extractor._client.chat.completions.create, gm._call_fast_llm)
    try:
        atomic_extractor._client.chat.completions.create = rate_limited
        gm._call_fast_llm = rate_limited
        client = TestClient(main.app)  # no lifespan: no scheduler, no startup work

        def store(text):
            r = client.post("/memory/store", headers=auth,
                            json={"userId": user_id, "text": text, "role": "user"})
            assert r.status_code == 200, r.text
            return r.json()

        question = store("Who did I use to like before that?")
        assert question["status"] == "filtered" and question["facts_count"] == 0, question

        mixed = store("I got the Amazon offer! What should I negotiate first?")
        statement = store("My favorite cricket player is Virat Kohli")
        assert mixed["facts_count"] == 1 and statement["facts_count"] == 1, (mixed, statement)

        texts = sorted(m["text"] for m in memory_store.get_all_memories(user_id))
        assert texts == ["I got the Amazon offer!", "My favorite cricket player is Virat Kohli"], texts
        assert not any("?" in t for t in texts), texts
    finally:
        atomic_extractor._client.chat.completions.create, gm._call_fast_llm = saved
        memory_store.delete_user_memories(user_id)
        gm.delete_user_graph(user_id)


def test_questions_produce_no_facts():
    """
    Regression: "Who did I use to like before that?" was rewritten by the
    extractor into the fact "User used to like someone before that" and
    stored. Questions must produce no facts; a mixed message keeps only its
    statements.
    """
    _, extract, _ = _deps()
    for question in ("Who did I use to like before that?",
                     "What was my old phone number?",
                     "Who is my favorite cricket player?"):
        items = extract(question)
        assert items == [], f"{question!r} produced facts: {[(i['text'], i['fallback']) for i in items]}"

    mixed = extract("I finished my AWS certification, what should I learn next?")
    assert len(mixed) == 1 and not mixed[0]["fallback"], mixed
    assert "aws" in mixed[0]["text"].lower() and mixed[0]["metadata"]["status"] == "completed", mixed


def test_history_context_is_bounded_and_scoped():
    """
    Regression: "Who did I use to like before that?" got only current-state
    context (Kohli), so the model said it had no record of a previous
    favorite. Spring's HistoryQuestionUtil now decides it's a history question
    and calls GET /graph/history -> get_history_context(). Seeds two revision
    chains directly (no LLM) and checks scoping, bounding and that the
    current-state path never leaks old values. The detector itself is tested
    in Spring (HistoryQuestionUtilTest).
    """
    gm, _, _ = _deps(require_llm=False)
    user_id = f"regress_{uuid.uuid4().hex[:10]}"
    base = {"status": "ongoing", "importance_score": 0.3, "timeline_year": 2026,
            "timeline_month": 9, "timeline_label": "September 2026"}

    def chain(domain, sub, old_text, old_name, new_text, new_name):
        old_id, new_id = f"{user_id}_{uuid.uuid4().hex[:8]}", f"{user_id}_{uuid.uuid4().hex[:8]}"
        meta = {**base, "domain": domain, "sub_domain": sub}
        assert gm.add_to_graph(user_id, old_id, old_text, {**meta, "entity_name": old_name})
        assert gm.add_to_graph(user_id, new_id, new_text, {**meta, "entity_name": new_name},
                               exclude_related_ids=[old_id])
        assert gm.supersede_node(user_id, old_id, new_id)

    try:
        chain("hobbies", "favorite cricket players",
              "User's favorite cricket player is Rohit Sharma", "Rohit Sharma",
              "User's favorite cricket player is Virat Kohli", "Virat Kohli")
        time.sleep(1.1)  # distinct archived_at, so "most recently changed" is well defined
        chain("job", "current employer",
              "User works at Yardi", "Yardi", "User works at Google", "Google")

        cricket = "- Favorite cricket players: previously Rohit Sharma, replaced by Virat Kohli (current)"
        employer = "- Current employer: previously Yardi, replaced by Google (current)"

        scoped = gm.get_history_context(user_id, domain="hobbies")
        assert scoped["domain"] == "hobbies", scoped
        assert scoped["history_context_text"] == cricket, scoped["history_context_text"]

        # A domain with no history falls back to the most recently changed slots.
        fallback = gm.get_history_context(user_id, domain="finance")
        assert fallback["domain"] is None, fallback
        assert cricket in fallback["history_context_text"] and employer in fallback["history_context_text"]

        # Bounded: limit=1 returns only the most recently changed slot.
        newest = gm.get_history_context(user_id, limit=1)
        assert newest["history_context_text"] == employer, newest["history_context_text"]

        # The current-state path never shows old values.
        current = gm.get_graph_context(user_id, "who is my favorite cricket player")
        assert "Rohit" not in current["graph_context_text"], current["graph_context_text"]

        # Timeline: optional domain filter; archived entries labelled as history.
        timeline = gm.get_timeline_summary(user_id, domain="job")
        flat = str(timeline)
        assert "Yardi (replaced by Google)" in flat and "Rohit" not in flat, timeline
        assert "hobbies" in str(gm.get_timeline_summary(user_id))
    finally:
        gm.delete_user_graph(user_id)


def test_compound_preferences_supersede_prior_slots():
    """
    With prior values for both slots, each fact of the compound message must
    supersede its OWN slot's node — a revision, never a progression, never
    status "completed".
    """
    gm, extract, worth = _deps()
    user_id = f"regress_{uuid.uuid4().hex[:10]}"
    try:
        _store(gm, extract, worth, user_id, "My favorite football player is Ronaldo")
        _store(gm, extract, worth, user_id, "My favorite tennis player is Nadal")
        results = _store(gm, extract, worth, user_id, COMPOUND_MESSAGE)

        assert len(results) == 2, results
        for r in results:
            assert r["action"] == "superseded", r
            assert r["match_type"] == "revision", r
            assert r["metadata"]["status"] == "ongoing", r

        replaced = {r["supersession"]["old"]["entity_name"].lower() for r in results}
        assert any("ronaldo" in e for e in replaced) and any("nadal" in e for e in replaced), replaced

        active = [n["text"].lower() for n in _nodes(gm, user_id) if not n["superseded_by"]]
        assert any("messi" in t for t in active) and any("federer" in t for t in active), active
        assert not any("ronaldo" in t or "nadal" in t for t in active), active
    finally:
        gm.delete_user_graph(user_id)


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    failed = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as e:
            print(f"SKIP  {name}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
