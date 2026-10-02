# ============================================================
# FILE: scheduler.py
#
# WHAT THIS FILE DOES:
# Runs generate_user_summary() (summarization.py) periodically, but only
# for users the service currently isn't busy serving — a lightweight
# in-process load gate, not a job queue or worker pool. Started from
# main.py's lifespan startup; stopped on shutdown.
#
# LOAD DETECTION — WHAT THIS DOES AND DOESN'T COVER:
# Two in-memory signals, both reset to defaults on every process restart:
#   1. A global in-flight-request counter, incremented/decremented by
#      main.py's middleware around every HTTP request.
#   2. A per-user "last seen" timestamp, updated by the three endpoints
#      that take a userId (store/retrieve/chat).
# A user is eligible for summarization when concurrent requests are at or
# below a small threshold AND that specific user has been idle for a few
# minutes — so a summarization run never competes with someone mid-chat.
#
# CONSTRAINT: this only works correctly with a single uvicorn process (no
# `workers=N`). main.py's __main__ block runs exactly one process today, so
# that holds — but if this service is ever scaled to multiple worker
# processes, each process would track its own counter/timestamps
# independently and this gate would stop reflecting true global load.
# Fixing that would need a shared store (e.g. Redis); deliberately not
# built now since it isn't needed at this scale — flagged here so it isn't
# a silent trap later.
#
# WHERE IT FITS IN THE PIPELINE:
# main.py middleware/endpoints → mark_request_start/end(), mark_user_active()
# → scheduler tick (every SUMMARY_TICK_INTERVAL_MINUTES) → is_low_load(userId)
# → summarization.generate_user_summary(userId)
# ============================================================

import logging
import time
from typing import Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler

import summarization

logger = logging.getLogger(__name__)

# How often the scheduler checks for summarization work.
SUMMARY_TICK_INTERVAL_MINUTES = 10

# A user is only summarized if the service has this many or fewer HTTP
# requests in flight right now, across all users.
MAX_CONCURRENT_REQUESTS_FOR_SUMMARY = 2

# A user is only summarized if they haven't hit store/retrieve/chat in at
# least this many seconds — avoids competing with someone mid-conversation.
USER_IDLE_SECONDS_FOR_SUMMARY = 300  # 5 minutes

# Cap on how many users get summarized in a single tick, so one tick can't
# run long even if many users have a backlog — the rest just get picked up
# on the next tick.
MAX_USERS_PER_TICK = 5

_in_flight_requests = 0
_last_seen: Dict[str, float] = {}

_scheduler: Optional[BackgroundScheduler] = None


def mark_request_start() -> None:
    """Call at the start of every HTTP request (main.py middleware)."""
    global _in_flight_requests
    _in_flight_requests += 1


def mark_request_end() -> None:
    """Call at the end of every HTTP request (main.py middleware)."""
    global _in_flight_requests
    _in_flight_requests = max(0, _in_flight_requests - 1)


def mark_user_active(userId: str) -> None:
    """
    Record that this user just made a request. Called from the
    store/retrieve/chat endpoint handlers specifically (not generic
    middleware) since the userId lives in the request body, not the URL.
    """
    _last_seen[userId] = time.time()


def is_low_load(userId: str) -> bool:
    """
    True if it's currently safe to run a summarization job for this user:
    the service isn't busy overall, and this specific user isn't mid-
    conversation.
    """
    if _in_flight_requests > MAX_CONCURRENT_REQUESTS_FOR_SUMMARY:
        return False
    idle_for = time.time() - _last_seen.get(userId, 0)
    return idle_for >= USER_IDLE_SECONDS_FOR_SUMMARY


def _run_summarization_tick() -> None:
    """
    The APScheduler job body: find users with pending nodes, summarize
    whichever ones are currently low-load, leave the rest for next tick.
    Never raises — an uncaught exception here would kill the scheduler's
    job silently; APScheduler logs it but the job wouldn't reschedule.
    """
    try:
        pending_users = summarization.get_users_with_pending_nodes()
    except Exception as e:
        logger.error(f"Summarization tick: failed to list pending users: {e}")
        return

    if not pending_users:
        logger.info("Summarization tick: no users with pending nodes")
        return

    logger.info(
        f"Summarization tick started — {len(pending_users)} user(s) with "
        f"pending nodes, in_flight_requests={_in_flight_requests}"
    )

    processed = 0
    skipped_busy = 0
    for userId in pending_users:
        if processed >= MAX_USERS_PER_TICK:
            logger.info(
                f"Summarization tick: hit MAX_USERS_PER_TICK={MAX_USERS_PER_TICK}, "
                f"remaining {len(pending_users) - processed - skipped_busy} user(s) "
                f"deferred to next tick"
            )
            break
        if not is_low_load(userId):
            skipped_busy += 1
            continue

        try:
            result = summarization.generate_user_summary(userId)
        except Exception as e:
            logger.error(f"Summarization tick: generate_user_summary failed userId={userId}: {e}")
            continue
        if result:
            processed += 1

    logger.info(
        f"Summarization tick complete — {processed} user(s) summarized, "
        f"{skipped_busy} skipped (not low-load)"
    )


def start() -> None:
    """Start the background scheduler. Called once from main.py's lifespan startup."""
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        _run_summarization_tick,
        "interval",
        minutes=SUMMARY_TICK_INTERVAL_MINUTES,
        id="user_summarization_tick",
    )
    _scheduler.start()
    logger.info(
        f"Summarization scheduler started — tick every "
        f"{SUMMARY_TICK_INTERVAL_MINUTES} min, "
        f"max_concurrent_requests={MAX_CONCURRENT_REQUESTS_FOR_SUMMARY}, "
        f"user_idle_seconds={USER_IDLE_SECONDS_FOR_SUMMARY}"
    )


def shutdown() -> None:
    """Stop the background scheduler. Called from main.py's lifespan shutdown."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Summarization scheduler stopped")
