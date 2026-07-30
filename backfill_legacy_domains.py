# ============================================================
# FILE: backfill_legacy_domains.py
#
# WHAT THIS SCRIPT DOES:
# One-time backfill for ChromaDB memories stored before domain tagging
# (Task 2) existed. Those entries have no "domain" key in their metadata at
# all, so once retrieve_memories() starts filtering by domain, they'd
# silently stop being retrievable in any domain-filtered search. This script
# assigns each of them a domain so they stay searchable.
#
# For every role="user" memory missing "domain":
#   1. Try classify_domain_by_embedding() (cheap, no LLM call).
#   2. If it isn't confident, fall back to extract_graph_metadata() (LLM).
#   3. Patch the resolved domain onto the ChromaDB entry via
#      memory_store.update_memory_domain().
#
# Not wired into app startup — run manually, once, after deploying the
# domain-tagging change:
#   python backfill_legacy_domains.py
#
# Safe to re-run: only touches entries that still lack a domain, so an
# interrupted run can just be started again.
# ============================================================

import logging

import memory_store
from graph_memory import classify_domain_by_embedding, extract_graph_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    collection = memory_store._collection

    results = collection.get(
        where={"role": {"$eq": "user"}},
        include=["documents", "metadatas"],
    )
    ids = results.get("ids") or []
    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    targets = [
        (memory_id, text, meta.get("userId"))
        for memory_id, text, meta in zip(ids, documents, metadatas)
        if not meta.get("domain")
    ]

    logger.info(
        f"Found {len(targets)} user-role memories missing domain "
        f"(out of {len(ids)} total user-role memories)"
    )

    classifier_hits = 0
    llm_fallbacks = 0
    failures = 0

    for memory_id, text, userId in targets:
        if not userId or not text:
            logger.warning(f"Skipping memoryId={memory_id} — missing userId or text")
            failures += 1
            continue

        domain = classify_domain_by_embedding(text, userId)
        if domain:
            classifier_hits += 1
        else:
            metadata = extract_graph_metadata(text, userId)
            domain = metadata.get("domain", "general")
            llm_fallbacks += 1

        ok = memory_store.update_memory_domain(memory_id, domain)
        if not ok:
            failures += 1
        logger.info(
            f"memoryId={memory_id} userId={userId} domain={domain!r} "
            f"text={text[:60]!r}"
        )

    logger.info(
        f"Backfill complete: {len(targets)} processed, "
        f"{classifier_hits} via classifier, {llm_fallbacks} via LLM fallback, "
        f"{failures} failures"
    )


if __name__ == "__main__":
    main()
