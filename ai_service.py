# ============================================================
# FILE: ai_service.py
#
# WHAT THIS FILE DOES:
# Handles all communication with the Groq LLM API.
# Builds the system prompt with or without memory context,
# then calls the Groq API and returns the response text.
#
# KEY CONCEPT FOR LEARNING:
# Context injection — we stuff retrieved memories into the
# system prompt so the LLM "knows" about the user's past.
# The LLM itself is stateless; the memory system around it
# creates the illusion of continuity.
#
# WHERE IT FITS IN THE PIPELINE:
# retrieved memories → build_system_prompt() →
# Groq API → response text → back to Spring Boot
# ============================================================

import logging
from typing import List, Optional

from groq import Groq
import config

logger = logging.getLogger(__name__)

# Initialise the Groq client once at module level.
# Creating it here means it is built once when the server
# starts, not on every request — much more efficient.
# Swap back to another provider: replace this client and
# the call inside call_llm(). Signatures stay the same.
_client = Groq(api_key=config.GROQ_API_KEY)

# Appended to EVERY chat system prompt. The frontend renders Markdown with
# remark-gfm and deliberately does NOT render raw HTML (XSS safety), so any
# <br> the model writes shows up as literal text. GFM tables have no way to
# break a line inside a cell, which is exactly where the model used <br>.
_FORMATTING_RULE = (
    "Format replies in Markdown only — never use HTML tags such as <br>. "
    "Keep every table cell on a single line; if a cell would need several "
    "items, separate them with commas, or use a list instead of a table."
)

# Appended whenever long-term context is present. The context can contain an
# old question or a fact without its resolution; the model once turned a
# sports-preference question into "a past crush" that was "still unanswered".
_CONTEXT_FIDELITY_RULE = (
    "Treat the background context as what the user actually said, in the "
    "sense they said it: never reinterpret a stored fact or question into an "
    "unrelated category (a favorite sports player is not a romantic "
    "interest). Never describe anything as unanswered, pending or unknown "
    "if the conversation or the background context already resolves it, and "
    "don't treat a past question the user asked as an open request."
)


def build_system_prompt(
    long_term_context: str, has_session_history: bool, for_chat: bool = True
) -> str:
    """
    Build the system prompt that tells the LLM how to behave.

    long_term_context is background/supporting material only — ChromaDB
    retrieval, graph_context_text, and the rolling summary, joined into one
    string by the caller. It is deliberately kept OUT of the conversation
    turns themselves: the actual current-session messages are passed as
    real prior `messages` entries by call_llm() (see below), not as text in
    here, so the model sees them as what they are — the live conversation —
    rather than as a paraphrase competing with stored memory for trust.

    has_session_history controls the wording: if there IS a current session
    (has_session_history=True), the instruction explicitly subordinates
    long_term_context to it. If this is the very first message of a session
    (has_session_history=False), there's nothing to subordinate it to, so
    that instruction would be meaningless — the memory context is simply
    background for a fresh conversation.

    for_chat=False is for graph_memory's internal judgement calls (matching,
    domain resolution, clustering), which reuse call_llm() with no context and
    must return strict JSON: they get the plain prompt, without the chat-only
    formatting rule ("Markdown only…"), exactly as before that rule existed.
    """
    if not long_term_context.strip():
        return (
            "You are a helpful AI assistant. "
            "This appears to be the beginning of a new "
            "conversation. "
            "Respond helpfully and naturally."
            + (f"\n\n{_FORMATTING_RULE}" if for_chat else "")
        )

    priority_instruction = (
        (
            "The conversation above (if any) is the CURRENT session — it is "
            "always more authoritative than the background context below. "
            "If the current conversation contradicts something below, trust "
            "the current conversation; it's more recent. Use the background "
            "context below only to fill in gaps the current conversation "
            "doesn't cover — never let it override or silently replace "
            "something the user has already said in this conversation.\n\n"
        )
        if has_session_history
        else ""
    )

    return (
        "You are a helpful AI assistant with memory of "
        "past conversations. "
        "You remember things this user has shared with "
        "you before.\n\n"
        f"{priority_instruction}"
        "Background context from long-term memory (past conversations, "
        "stored facts, and a periodic summary):\n"
        f"{long_term_context}\n\n"
        "Use this context naturally when relevant. "
        "Never say phrases like 'based on my memory' or "
        "'according to context'. "
        "Respond the way a knowledgeable, caring friend "
        "would respond.\n\n"
        f"{_CONTEXT_FIDELITY_RULE}\n\n"
        f"{_FORMATTING_RULE}"
    )


_VALID_ROLES = {"user", "assistant"}


def call_llm(
    message: str,
    context: str,
    session_history: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    for_chat: bool = True,
) -> str:
    """
    Call the Groq API and return the response text.

    Args:
        message: the user's current message
        context: long-term memory context (ChromaDB + graph_context_text +
                 rolling summary), already joined into a single string
                 (empty string if none found) — background only, see
                 build_system_prompt()
        session_history: prior turns of THIS session, oldest first, each
                 {"role": "user"|"assistant", "content": "..."}. Placed in
                 the `messages` array as real prior turns BEFORE the current
                 message — this is what gives the current session real
                 priority over long_term_context, not just wording: the
                 model is shown the actual conversation, not a summary of it
                 competing with retrieved memory for trust. Entries with any
                 other role are dropped defensively, since this list crosses
                 a service boundary (Spring Boot) this function doesn't
                 control the shape of.
        temperature: overrides config.LLM_TEMPERATURE for this call. Chat
                 leaves it unset (conversational). Yes/no judgement calls in
                 graph_memory (find_matching_task_node, resolve_domain) pass
                 0, because a matching decision should be deterministic.
        for_chat: False for internal JSON judgement calls — omits the chat-only
                 formatting rule from the system prompt (see build_system_prompt).

    Returns:
        The LLM's response as a plain string.

    Raises:
        Exception: re-raised so main.py can return HTTP 503.
    """
    session_history = session_history or []
    history_turns = [
        {"role": turn["role"], "content": turn["content"]}
        for turn in session_history
        if turn.get("role") in _VALID_ROLES and turn.get("content")
    ]

    system_prompt = build_system_prompt(
        context, has_session_history=bool(history_turns), for_chat=for_chat
    )

    logger.info("========== SYSTEM PROMPT ==========")
    logger.info(system_prompt)
    logger.info(f"========== SESSION HISTORY ({len(history_turns)} turns) ==========")
    logger.info("========== USER MESSAGE ==========")
    logger.info(message)
    logger.info("==================================")

    # temperature controls creativity: 0.7 is a good balance
    # for conversational AI — not too robotic, not too random.
    response = _client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_turns,
            {"role": "user", "content": message},
        ],
    )

    answer = response.choices[0].message.content

    logger.info(f"call_llm → response length={len(answer)} chars")
    return answer
