"""
core/guardrails.py

Three-layer guardrail system that sits in front of the RAG engine:

  Layer 1 — Input Sanitization
  Layer 2 — Intent Classification (keyword pre-check + LLM classifier)
  Layer 3 — Query Rewriting
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from openai import OpenAI
from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class IntentClass(str, Enum):
    RELEVANT   = "RELEVANT"
    IRRELEVANT = "IRRELEVANT"
    UNCLEAR    = "UNCLEAR"


@dataclass
class GuardrailResult:
    allowed: bool
    intent: IntentClass
    reason: str
    sanitized_query: str
    rewritten_query: str
    preset_answer: Optional[str] = None  # if set, RAG skips retrieval


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500
MIN_QUERY_LENGTH = 3

_GREETING_NORMALIZED = frozenset({
    "hi", "hello", "hey", "howdy", "greetings", "hiya", "yo", "sup",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there",
})

_GREETING_REPLY = (
    "Hello! I'm your earnings-call assistant. Ask me anything about revenue, margins, "
    "guidance, deals, or other topics covered in the transcripts you've ingested."
)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"forget\s+(everything|all|your)",
    r"act\s+as\s+(if\s+you\s+are|a)",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[\s*INST\s*\]",
    r"jailbreak",
    r"DAN\s+mode",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_FINANCE_KEYWORDS = {
    "revenue", "earnings", "profit", "margin", "ebitda", "pat", "quarter",
    "fy", "financial", "growth", "guidance", "deal", "tcv", "order", "book",
    "client", "customer", "segment", "vertical", "business", "company",
    "ceo", "cfo", "management", "analyst", "investor", "stock", "share",
    "dividend", "cash", "flow", "balance", "sheet", "debt", "headcount",
    "hiring", "attrition", "utilization", "offshore", "onsite", "erp",
    "infrastructure", "digital", "bfsi", "manufacturing", "healthcare",
    "q1", "q2", "q3", "q4", "birlasoft", "infosys", "tcs", "wipro",
    "performance", "outlook", "forecast", "pipeline", "signing", "ramp",
    "salary", "hike", "cost", "expense", "operating", "inr", "usd",
    "crore", "million", "billion", "basis", "point", "yoy", "qoq",
}

_OFFTOPIC_KEYWORDS = {
    "recipe", "cooking", "weather", "sports", "cricket", "football",
    "movie", "song", "lyrics", "celebrity", "actor", "politician",
    "joke", "poem", "story", "write me", "translate", "code for",
    "what is the capital", "who invented", "how to cook",
}

_INTENT_SYSTEM_PROMPT = """You are a query classifier for a financial earnings call analysis system.
Classify if the user's query is relevant to financial earnings calls, company performance, business metrics, or related topics.

Respond with exactly one word:
- RELEVANT   → query is about earnings, financials, business performance, companies, deals, margins, revenue, etc.
- IRRELEVANT → query is clearly off-topic (cooking, sports, general knowledge, coding help, etc.)
- UNCLEAR    → cannot determine intent

Examples:
"What was Birlasoft's revenue in Q1?" → RELEVANT
"How did BFSI margins change?" → RELEVANT
"What is the weather today?" → IRRELEVANT
"Write me a Python function" → IRRELEVANT
"Tell me about the company" → UNCLEAR"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_greeting_text(q: str) -> str:
    s = re.sub(r"[^\w\s]", "", q.lower())
    return re.sub(r"\s+", " ", s).strip()


def _is_greeting_only(q: str) -> bool:
    if not q:
        return False
    return _normalize_greeting_text(q) in _GREETING_NORMALIZED


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class Guardrails:

    def __init__(self) -> None:
        settings.validate()
        self._client = OpenAI(api_key=settings.openai_api_key)

    def check(self, query: str) -> GuardrailResult:
        stripped = query.strip()

        # Greeting shortcut — no RAG needed
        if _is_greeting_only(stripped):
            return GuardrailResult(
                allowed=True,
                intent=IntentClass.UNCLEAR,
                reason="",
                sanitized_query=stripped,
                rewritten_query=stripped,
                preset_answer=_GREETING_REPLY,
            )

        # Layer 1: Sanitize
        sanitized, sanitize_reason = self._sanitize(query)
        if sanitize_reason:
            return GuardrailResult(
                allowed=False,
                intent=IntentClass.IRRELEVANT,
                reason=sanitize_reason,
                sanitized_query=sanitized,
                rewritten_query=sanitized,
            )

        # Layer 2a: Fast keyword check
        fast_result = self._fast_keyword_check(sanitized)
        if fast_result is not None:
            intent, allowed = fast_result
            if not allowed:
                return GuardrailResult(
                    allowed=False,
                    intent=intent,
                    reason=(
                        "Your question doesn't appear to be related to financial earnings calls. "
                        "I can only answer questions about company performance, revenue, margins, "
                        "deals, and other earnings call topics."
                    ),
                    sanitized_query=sanitized,
                    rewritten_query=sanitized,
                )

        # Layer 2b: LLM intent classifier
        intent = self._classify_intent(sanitized)
        if intent == IntentClass.IRRELEVANT:
            return GuardrailResult(
                allowed=False,
                intent=intent,
                reason=(
                    "Your question doesn't appear to be related to financial earnings calls. "
                    "I can only answer questions about company performance, revenue, margins, "
                    "deals, and other earnings call topics."
                ),
                sanitized_query=sanitized,
                rewritten_query=sanitized,
            )

        # Layer 3: Query rewriting
        rewritten = self._rewrite_query(sanitized)

        return GuardrailResult(
            allowed=True,
            intent=intent,
            reason="",
            sanitized_query=sanitized,
            rewritten_query=rewritten,
        )

    def _sanitize(self, query: str) -> tuple[str, str]:
        cleaned = query.strip()
        if len(cleaned) < MIN_QUERY_LENGTH:
            return cleaned, "Query is too short. Please ask a complete question."
        if len(cleaned) > MAX_QUERY_LENGTH:
            return cleaned[:MAX_QUERY_LENGTH], f"Query was truncated to {MAX_QUERY_LENGTH} characters."
        if _INJECTION_RE.search(cleaned):
            logger.warning(f"Prompt injection attempt detected: {cleaned[:100]}")
            return cleaned, (
                "I detected an attempt to override my instructions. "
                "Please ask a genuine question about financial earnings calls."
            )
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned, ""

    def _fast_keyword_check(self, query: str) -> Optional[tuple[IntentClass, bool]]:
        q_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', q_lower))
        if words & _FINANCE_KEYWORDS:
            return (IntentClass.RELEVANT, True)
        for phrase in _OFFTOPIC_KEYWORDS:
            if phrase in q_lower:
                return (IntentClass.IRRELEVANT, False)
        return None

    def _classify_intent(self, query: str) -> IntentClass:
        try:
            response = self._client.chat.completions.create(
                model=settings.model,
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                max_tokens=5,
                temperature=0,
            )
            result = response.choices[0].message.content.strip().upper()
            logger.debug(f"Intent classification: '{query[:50]}' → {result}")
            if "IRRELEVANT" in result:
                return IntentClass.IRRELEVANT
            elif "UNCLEAR" in result:
                return IntentClass.UNCLEAR
            else:
                return IntentClass.RELEVANT
        except Exception as e:
            logger.warning(f"Intent classifier error (defaulting to RELEVANT): {e}")
            return IntentClass.RELEVANT

    def _rewrite_query(self, query: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=settings.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a search query optimizer for financial earnings call transcripts. "
                            "Rewrite the user's question as a concise search query using financial terminology. "
                            "Include relevant keywords like metric names, company names, quarter identifiers. "
                            "Return ONLY the rewritten query, nothing else. Keep it under 20 words."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=40,
                temperature=0,
            )
            rewritten = response.choices[0].message.content.strip()
            if len(rewritten) > 3 and len(rewritten) < len(query) * 3:
                logger.debug(f"Query rewrite: '{query[:50]}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed (using original): {e}")
        return query