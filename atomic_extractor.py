# ============================================================
# FILE: atomic_extractor.py
#
# WHAT THIS FILE DOES:
# Splits a user message into atomic facts AND classifies each one (domain,
# sub_domain, status, importance, entity) in a SINGLE LLM call, with the
# whole message as shared context. "My name is Rushabh and I work at Yardi"
# becomes two facts, each with its own graph metadata.
#
# Why one call instead of split-then-classify: classifying each fact alone
# lost the sibling context. "For football my favorite is Messi and for
# tennis it's Federer" was split correctly, but the tennis half — judged in
# isolation — was sometimes classified status="completed" while its football
# sibling was "ongoing". Here the model sees the sentence once and classifies
# its facts together and consistently.
#
# WHERE IT FITS IN THE PIPELINE:
# raw user text → extract_facts_with_metadata() → [{text, metadata}, ...]
#   → main.py stores each fact in ChromaDB, then hands its metadata to the
#     background graph task (graph_memory.process_memory_intent), which
#     normalizes it instead of making a second LLM call per fact.
# ============================================================

import json
import logging
import re
from typing import Any, Dict, List, Optional

import config
from ai_service import _client
from graph_memory import TOP_LEVEL_DOMAINS, is_declarative, safe_parse_json

logger = logging.getLogger(__name__)

_TEXT_PLACEHOLDER = "<<TEXT>>"

# Built with a placeholder rather than str.format(): the JSON examples below
# contain literal { } braces.
_EXTRACTION_PROMPT = (
    "Split the MESSAGE into individual atomic facts, and classify each fact.\n\n"
    "All the facts come from the SAME message. Classify them together and "
    "consistently: facts that make the same kind of statement get the same "
    "status. If one fact is a restated preference, sibling facts from the "
    "same sentence describing similar preferences are also preferences — "
    "never completed tasks.\n\n"
    "SPLITTING RULES:\n"
    "- Start each fact with 'User' or 'User's' and write a grammatical sentence:\n"
    "  - actions, states, goals: 'User <verb> ...' (User works at Yardi)\n"
    "  - attributes and preferences: 'User's <attribute> is <value>' "
    "(User's favorite movie is Inception)\n"
    "- Never make the user equal to a person or thing: write "
    "'User's favorite football player is Messi', never 'User is Messi'\n"
    "- Spell out the attribute an elliptical phrase implies: 'for football "
    "it's Messi' means User's favorite football player is Messi\n"
    "- Keep a comparison in ONE fact: 'User prefers A over B'; never split on "
    "than, over, rather\n"
    "- Never use he/she/they/it/his/her pronouns; never use the person's name as subject\n"
    "- Each fact must be self-contained, maximum 12 words\n"
    "- Split on: and, but, also, additionally\n"
    "- Questions and requests are NOT facts. Never rewrite a question as a "
    "statement (\"What was my old phone number?\" is NOT \"User had an old "
    "phone number\"). A message that is only questions or requests produces "
    "an empty array []. In a mixed message, extract only the statements.\n\n"
    "CLASSIFICATION RULES (per fact):\n"
    "- entity_name: the main subject/entity of the fact\n"
    f"- domain: EXACTLY one of: {', '.join(TOP_LEVEL_DOMAINS)}. Never invent one. "
    "Favorites and preferences about sports, players, teams, music, films, "
    "shows or games are \"hobbies\" — also when the sport isn't named "
    "(\"my favorite player is X\"). Use \"general\" only if no other domain fits.\n"
    "- sub_domain: a short, specific free-form label inside the domain "
    "(e.g. \"favorite cricket players\", \"interview preparation\")\n"
    "- status: \"completed\" ONLY for a task, goal or event the message says "
    "has finished (finished, done, passed, failed, got, received, quit). "
    "Standing facts, attributes and preferences are ALWAYS \"ongoing\" — "
    "including a changed one (\"my favorite is now X\" is ongoing, not completed).\n"
    "- importance_score: 0.8-1.0 major life events (job offer, diagnosis, "
    "graduation); 0.5-0.7 significant events (interview, exam, new course); "
    "0.2-0.4 regular activities and ordinary preferences; 0.1-0.2 casual mentions\n"
    "- related_entities: other entities the fact mentions (may be empty)\n\n"
    "EXAMPLES:\n"
    "Message: \"My name is Rushabh and I work at Yardi\"\n"
    "Output: ["
    "{\"fact\": \"User's name is Rushabh\", \"entity_name\": \"Rushabh\", \"domain\": \"general\", "
    "\"sub_domain\": \"identity\", \"status\": \"ongoing\", \"importance_score\": 0.5, \"related_entities\": []}, "
    "{\"fact\": \"User works at Yardi\", \"entity_name\": \"Yardi\", \"domain\": \"job\", "
    "\"sub_domain\": \"current employer\", \"status\": \"ongoing\", \"importance_score\": 0.6, \"related_entities\": []}]\n\n"
    "Message: \"My favourite movie is Inception and for TV it's Breaking Bad\"\n"
    "Output: ["
    "{\"fact\": \"User's favorite movie is Inception\", \"entity_name\": \"Inception\", \"domain\": \"hobbies\", "
    "\"sub_domain\": \"favorite movies\", \"status\": \"ongoing\", \"importance_score\": 0.3, \"related_entities\": []}, "
    "{\"fact\": \"User's favorite TV show is Breaking Bad\", \"entity_name\": \"Breaking Bad\", \"domain\": \"hobbies\", "
    "\"sub_domain\": \"favorite TV shows\", \"status\": \"ongoing\", \"importance_score\": 0.3, \"related_entities\": []}]\n"
    "(two sibling preferences from one sentence: both ongoing)\n\n"
    "Message: \"I finished my AWS certification and my go-to editor is VS Code\"\n"
    "Output: ["
    "{\"fact\": \"User finished the AWS certification\", \"entity_name\": \"AWS certification\", \"domain\": \"education\", "
    "\"sub_domain\": \"certifications\", \"status\": \"completed\", \"importance_score\": 0.7, \"related_entities\": [\"AWS\"]}, "
    "{\"fact\": \"User's go-to editor is VS Code\", \"entity_name\": \"VS Code\", \"domain\": \"hobbies\", "
    "\"sub_domain\": \"developer tools\", \"status\": \"ongoing\", \"importance_score\": 0.2, \"related_entities\": []}]\n"
    "(a finished task and a preference in one sentence: consistency means "
    "the SAME KIND of fact gets the same status, not that every fact does)\n\n"
    "Message: \"Which laptop did I say I wanted to buy?\"\n"
    "Output: []\n"
    "(only a question — nothing to store)\n\n"
    "Message: \"I got the Amazon offer! What should I negotiate first?\"\n"
    "Output: [{\"fact\": \"User got the Amazon job offer\", \"entity_name\": \"Amazon offer\", \"domain\": \"job\", "
    "\"sub_domain\": \"job offers\", \"status\": \"completed\", \"importance_score\": 0.9, \"related_entities\": [\"Amazon\"]}]\n"
    "(the question part is dropped; the statement is kept)\n\n"
    "Message: \"For lunch it's usually dal rice\"\n"
    "Output: [{\"fact\": \"User's usual lunch is dal rice\", \"entity_name\": \"dal rice\", \"domain\": \"food\", "
    "\"sub_domain\": \"usual meals\", \"status\": \"ongoing\", \"importance_score\": 0.2, \"related_entities\": []}]\n\n"
    "Message: \"I'd rather work from home than go to the office\"\n"
    "Output: [{\"fact\": \"User prefers working from home over going to the office\", \"entity_name\": \"remote work\", "
    "\"domain\": \"job\", \"sub_domain\": \"work preferences\", \"status\": \"ongoing\", \"importance_score\": 0.3, "
    "\"related_entities\": []}]\n\n"
    f"MESSAGE: \"{_TEXT_PLACEHOLDER}\"\n\n"
    "Return ONLY a JSON array of objects with exactly these keys: fact, entity_name, "
    "domain, sub_domain, status, importance_score, related_entities. No markdown."
)


def extract_facts_with_metadata(text: str) -> List[Dict[str, Any]]:
    """
    Split `text` into atomic facts and classify each, in one LLM call.

    Returns [{"text": <fact>, "metadata": <raw classification dict>,
              "fallback": False}, ...].
    The metadata is the model's raw output (entity_name, domain, sub_domain,
    status, importance_score, related_entities). graph_memory's
    normalize_graph_metadata() validates and clamps it later, in the
    background task, so this synchronous step does no DB work.

    Returns [] when the message contains no statements (only questions or
    requests) — a valid result: nothing is stored. Questions are never
    rewritten into facts ("Who did I use to like before that?" once became
    "User used to like someone before that", which the per-fact
    is_declarative() gate can't catch).

    Falls back to [{"text": text, "metadata": None, "fallback": True}] on
    any failure — a malformed response or API error (e.g. a 429 rate limit)
    must never block memory storage. "fallback": True marks the fact as the
    unsplit whole message; graph_memory.process_memory_intent() then stores
    it as a plain, needs_reprocessing node and never matches it against
    existing nodes.
    """
    fallback: List[Dict[str, Any]] = [{"text": text, "metadata": None, "fallback": True}]
    if not text or not text.strip():
        return fallback

    prompt = _EXTRACTION_PROMPT.replace(_TEXT_PLACEHOLDER, text)

    try:
        # temperature=0.1: structured extraction, not creative writing.
        # max_tokens covers hidden reasoning + the JSON answer: gpt-oss models
        # reason first, and this answer is larger than the old string-only
        # split (which needed ~250 tokens for 2 facts and failed at 200).
        response = _client.chat.completions.create(
            model=config.FAST_LLM_MODEL,
            temperature=0.1,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            # gpt-oss returns empty content when it runs out of tokens mid-reasoning.
            raise ValueError("empty response")
        parsed = _parse_json_array(raw)

        if not isinstance(parsed, list):
            raise ValueError(f"expected a JSON array, got: {raw[:200]!r}")
        if not parsed:
            # A valid answer, not a failure: the message had no statements
            # (only questions/requests). Store nothing — never fall back to
            # storing the question itself as a fact.
            logger.info(f"extract_facts_with_metadata: no facts in message {text[:80]!r}")
            return []

        items: List[Dict[str, Any]] = []
        for obj in parsed:
            if not isinstance(obj, dict) or not str(obj.get("fact") or "").strip():
                raise ValueError(f"item without a 'fact': {obj!r}")
            fact = str(obj["fact"]).strip()
            metadata = {k: v for k, v in obj.items() if k != "fact"}
            items.append({"text": fact, "metadata": metadata, "fallback": False})

        logger.info(
            "extract_facts_with_metadata split into %d facts: %s",
            len(items),
            [(i["text"], i["metadata"].get("status"), i["metadata"].get("domain")) for i in items],
        )
        return items

    except Exception as e:
        logger.warning(f"extract_facts_with_metadata failed, falling back to original text: {e}")
        return fallback


def _parse_json_array(raw: str) -> Optional[Any]:
    """Parse a JSON array, tolerating code fences and surrounding prose."""
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
        # A single object instead of an array — accept it as one fact.
        return [safe_parse_json(cleaned)]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def drop_question_sentences(text: str) -> str:
    """
    Remove question sentences from a message, keeping its statements:
      "Who did I use to like before that?"                      -> ""
      "I got the Amazon offer! What should I negotiate first?"  -> "I got the Amazon offer!"
      "My favorite player is Virat Kohli"                       -> unchanged

    Used on the extractor's FALLBACK item (the whole raw message, stored when
    extraction fails). A successful extraction already drops questions, but
    the fallback used to store a raw question in ChromaDB as a "fact": it was
    later retrieved as a past memory with no answer attached, and the model
    described it as an unanswered "past crush". The per-fact is_declarative()
    gate in graph_memory only protects the graph, not ChromaDB.
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split((text or "").strip()) if s.strip()]
    return " ".join(s for s in sentences if is_declarative(s))


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
