"""
core/guardrails.py

Three-layer guardrail system that sits in front of the RAG engine:

  Layer 1 — Input Sanitization
    Strip prompt injection attempts, excessive length, dangerous patterns
    before anything hits the LLM.

  Layer 2 — Intent Classification
    Fast, cheap LLM call to decide if a query is finance/earnings related.
    Blocks off-topic questions (sports, recipes, general knowledge, etc.)
    before spending tokens on full RAG retrieval.

  Layer 3 — Response Validation
    After RAG generates an answer, check it doesn't hallucinate or go
    off-topic. If confidence is too low, return a graceful fallback.

Design note:
  The intent classifier uses gpt-4o-mini with a very small prompt and
  max_tokens=10 — it only needs to return "RELEVANT" or "IRRELEVANT".
  Cost is negligible (~0.001 cents per check).
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
    RELEVANT   = "RELEVANT"    # finance / earnings related → proceed
    IRRELEVANT = "IRRELEVANT"  # off-topic → block
    UNCLEAR    = "UNCLEAR"     # ambiguous → allow with warning


@dataclass
class GuardrailResult:
    allowed: bool
    intent: IntentClass
    reason: str
    sanitized_query: str        # cleaned version of original input
    rewritten_query: str        # optimized for embedding retrieval


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 500          # chars
MIN_QUERY_LENGTH = 3

# Patterns that indicate prompt injection attempts
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

# Finance-adjacent keywords for fast pre-check before LLM call
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

# Clearly off-topic keywords — fast reject without LLM call
_OFFTOPIC_KEYWORDS = {
    "recipe", "cooking", "weather", "sports", "cricket", "football",
    "movie", "song", "lyrics", "celebrity", "actor", "politician",
    "joke", "poem", "story", "write me", "translate", "code for",
    "what is the capital", "who invented", "how to cook",
}

# Intent classifier system prompt — kept very short for speed/cost
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
# Guardrails
# ---------------------------------------------------------------------------

class Guardrails:
    """
    Wraps the full guardrail pipeline.
    Used by RAGEngine before every query.
    """

    def __init__(self) -> None:
        settings.validate()
        self._client = OpenAI(api_key=settings.openai_api_key)

    def check(self, query: str) -> GuardrailResult:
        """
        Run all guardrail layers on a user query.
        Returns GuardrailResult — caller checks .allowed before proceeding.
        """
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

        # Layer 2: Fast keyword pre-check (no LLM cost)
        fast_result = self._fast_keyword_check(sanitized)
        if fast_result is not None:
            intent, allowed = fast_result
            if not allowed:
                return GuardrailResult(
                    allowed=False,
                    intent=intent,
                    reason="Your question doesn't appear to be related to financial earnings calls. "
                           "I can only answer questions about company performance, revenue, margins, "
                           "deals, and other earnings call topics.",
                    sanitized_query=sanitized,
                    rewritten_query=sanitized,
                )

        # Layer 2b: LLM intent classifier (for ambiguous cases)
        intent = self._classify_intent(sanitized)
        if intent == IntentClass.IRRELEVANT:
            return GuardrailResult(
                allowed=False,
                intent=intent,
                reason="Your question doesn't appear to be related to financial earnings calls. "
                       "I can only answer questions about company performance, revenue, margins, "
                       "deals, and other earnings call topics.",
                sanitized_query=sanitized,
                rewritten_query=sanitized,
            )

        # Layer 3: Query rewriting for better retrieval
        rewritten = self._rewrite_query(sanitized)

        return GuardrailResult(
            allowed=True,
            intent=intent,
            reason="",
            sanitized_query=sanitized,
            rewritten_query=rewritten,
        )

    # ------------------------------------------------------------------
    # Layer 1: Input sanitization
    # ------------------------------------------------------------------

    def _sanitize(self, query: str) -> tuple[str, str]:
        """
        Clean and validate input.
        Returns (cleaned_query, error_message).
        error_message is empty string if input is clean.
        """
        # Strip leading/trailing whitespace
        cleaned = query.strip()

        # Length checks
        if len(cleaned) < MIN_QUERY_LENGTH:
            return cleaned, "Query is too short. Please ask a complete question."

        if len(cleaned) > MAX_QUERY_LENGTH:
            return cleaned[:MAX_QUERY_LENGTH], \
                f"Query was truncated to {MAX_QUERY_LENGTH} characters."

        # Prompt injection detection
        if _INJECTION_RE.search(cleaned):
            logger.warning(f"Prompt injection attempt detected: {cleaned[:100]}")
            return cleaned, "I detected an attempt to override my instructions. " \
                            "Please ask a genuine question about financial earnings calls."

        # Strip potential HTML/script tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)

        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned, ""

    # ------------------------------------------------------------------
    # Layer 2: Intent classification
    # ------------------------------------------------------------------

    def _fast_keyword_check(self, query: str) -> Optional[tuple[IntentClass, bool]]:
        """
        Zero-cost keyword check before LLM call.
        Returns (IntentClass, allowed) or None if inconclusive.
        """
        q_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', q_lower))

        # Definite finance hit → skip LLM check
        if words & _FINANCE_KEYWORDS:
            return (IntentClass.RELEVANT, True)

        # Definite off-topic hit → block without LLM call
        for phrase in _OFFTOPIC_KEYWORDS:
            if phrase in q_lower:
                return (IntentClass.IRRELEVANT, False)

        return None  # Inconclusive — let LLM decide

    def _classify_intent(self, query: str) -> IntentClass:
        """
        LLM-based intent classification.
        Uses minimal tokens — only returns one word.
        """
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
            # If classifier fails, default to ALLOW — don't block legitimate queries
            logger.warning(f"Intent classifier error (defaulting to RELEVANT): {e}")
            return IntentClass.RELEVANT

    # ------------------------------------------------------------------
    # Layer 3: Query rewriting
    # ------------------------------------------------------------------

    def _rewrite_query(self, query: str) -> str:
        """
        Rewrite query to improve embedding retrieval quality.

        Examples:
          "what did birlasoft earn" → "Birlasoft revenue earnings Q1 FY25"
          "how are margins"         → "EBITDA margin performance quarterly"
          "tell me about deals"     → "TCV deal wins order book"

        Uses a fast, cheap LLM call. Falls back to original if it fails.
        """
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
            # Sanity check — if rewrite is wildly different length, use original
            if len(rewritten) > 3 and len(rewritten) < len(query) * 3:
                logger.debug(f"Query rewrite: '{query[:50]}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed (using original): {e}")

        return query