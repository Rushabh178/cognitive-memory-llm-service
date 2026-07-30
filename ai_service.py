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
from groq import Groq
import config

logger = logging.getLogger(__name__)

# Initialise the Groq client once at module level.
# Creating it here means it is built once when the server
# starts, not on every request — much more efficient.
# Swap back to another provider: replace this client and
# the call inside call_llm(). Signatures stay the same.
_client = Groq(api_key=config.GROQ_API_KEY)


def build_system_prompt(context: str) -> str:
    """
    Build the system prompt that tells the LLM how to behave.

    If context (retrieved memories) is present, we inject it
    so the LLM can reference the user's past conversations.
    If context is empty, we tell the LLM this is a fresh start.
    """
    if context.strip():
        return (
            "You are a helpful AI assistant with memory of "
            "past conversations. "
            "You remember things this user has shared with "
            "you before.\n\n"
            "Relevant context from past conversations:\n"
            f"{context}\n\n"
            "Use this context naturally when relevant. "
            "Never say phrases like 'based on my memory' or "
            "'according to context'. "
            "Respond the way a knowledgeable, caring friend "
            "would respond."
        )
    else:
        return (
            "You are a helpful AI assistant. "
            "This appears to be the beginning of a new "
            "conversation. "
            "Respond helpfully and naturally."
        )


def call_llm(message: str, context: str) -> str:
    """
    Call the Groq API and return the response text.

    Args:
        message: the user's current message
        context: retrieved memories as a newline-joined string
                 (empty string if no memories found)

    Returns:
        The LLM's response as a plain string.

    Raises:
        Exception: re-raised so main.py can return HTTP 503.
    """
    system_prompt = build_system_prompt(context)

    logger.info("========== SYSTEM PROMPT ==========")
    logger.info(system_prompt)
    logger.info("========== USER MESSAGE ==========")
    logger.info(message)
    logger.info("==================================")

    # temperature controls creativity: 0.7 is a good balance
    # for conversational AI — not too robotic, not too random.
    response = _client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": message},
        ],
    )

    answer = response.choices[0].message.content

    logger.info(f"call_llm → response length={len(answer)} chars")
    return answer
