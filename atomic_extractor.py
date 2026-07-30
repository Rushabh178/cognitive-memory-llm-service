# ============================================================
# FILE: atomic_extractor.py
#
# WHAT THIS FILE DOES:
# Splits a compound user message into individual atomic facts using the
# LLM, so "My name is Rushabh and I work at Yardi" is stored as two
# separate memories instead of one blob mixing two unrelated facts.
# Splitting at storage time means retrieval can later match ONE fact
# precisely instead of pulling in a whole sentence to get part of it.
#
# WHERE IT FITS IN THE PIPELINE:
# raw user text → extract_atomic_facts() → list of atomic fact strings
#              → main.py stores each one separately in ChromaDB
# ============================================================

import json
import logging
from typing import List

import config
from ai_service import _client

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = (
    "Split this text into individual atomic facts.\n\n"
    "Rules:\n"
    "- Always start each fact with 'User' as subject\n"
    "- Never use he/she/they/it/his/her pronouns\n"
    "- Never use the person's name as subject\n"
    "- Each fact must be self-contained\n"
    "- Maximum 12 words per fact\n"
    "- Split on: and, but, also, additionally\n\n"
    "Examples:\n"
    "Input: \"My name is Rushabh and I work at Yardi\"\n"
    "Output: [\"User name is Rushabh\", \"User works at Yardi\"]\n\n"
    "Input: \"I work in CSD at Yardi and want to switch to MNC tech role\"\n"
    "Output: [\"User works in CSD department at Yardi\", \"User goal is to switch to MNC tech role\"]\n\n"
    "Input: \"I want to deploy on AWS\"\n"
    "Output: [\"User wants to deploy project on AWS\"]\n\n"
    "Input: \"I have fever\"\n"
    "Output: [\"User has fever\"]\n\n"
    "Text: \"{text}\"\n"
    "Return ONLY JSON array: [\"fact1\", \"fact2\"]"
)


def extract_atomic_facts(text: str) -> List[str]:
    """
    Split a compound statement into a list of atomic facts via the LLM.

    Falls back to [text] (the original, unsplit) on any failure — a
    malformed LLM response or API error should never block memory storage.
    """
    if not text or not text.strip():
        return [text]

    prompt = _EXTRACTION_PROMPT.format(text=text)

    try:
        # temperature=0.1 keeps the split deterministic and literal — this
        # is a structured extraction task, not a creative one.
        response = _client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            temperature=0.1,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()

        # The LLM sometimes wraps JSON in a markdown code fence despite the
        # "Return ONLY JSON array" instruction — strip it before parsing.
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        facts = json.loads(raw)

        if not isinstance(facts, list) or not facts or not all(
            isinstance(f, str) and f.strip() for f in facts
        ):
            raise ValueError(f"LLM returned an unexpected shape: {raw!r}")

        logger.info(f"extract_atomic_facts split into {len(facts)} facts: {facts}")
        return facts

    except Exception as e:
        logger.warning(
            f"extract_atomic_facts failed, falling back to original text: {e}"
        )
        return [text]


def is_fact_worth_storing(fact: str) -> bool:
    """
    Filters out low-quality atomic facts that survived extraction but aren't
    worth storing on their own — fragments too short to carry meaning, or
    meta-cognitive statements about the conversation itself (a question about
    a fact isn't the fact).
    """
    fact_lower = fact.lower().strip()

    # Too short — not meaningful alone
    if len(fact.split()) < 4:
        return False

    # Meta-cognitive fragments — not real facts
    skip_patterns = [
        "user thinks", "user might",
        "user will forget", "user asks about",
        "user needs to know", "user wants to know",
        "user is asking", "user asked",
        "user wonders"
    ]
    if any(p in fact_lower for p in skip_patterns):
        return False

    return True
