"""
core/rag.py

Retrieval-Augmented Generation engine with guardrails.

Pipeline per query:
  1. Guardrails (sanitize → intent classify → rewrite)
  2. Embed rewritten query
  3. Retrieve top-k chunks from VectorStore
  4. Filter by similarity threshold
  5. Build context prompt
  6. LLM generation
  7. Return answer + sources + metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.embedder import Embedder
from core.vectorstore import VectorStore, SearchResult, SearchFilters
from core.llm import LLMClient, ConversationHistory
from core.guardrails import Guardrails, extract_quarter_filter
from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    used_rag: bool
    query: str
    rewritten_query: str = ""
    blocked: bool = False
    block_reason: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a financial analyst assistant specializing in earnings call analysis.

Answer the user's question using ONLY the context provided below from earnings call transcripts.
Be precise and cite specific numbers, metrics, and speaker names where relevant.

Rules:
- Only use information from the provided context. Do not use external knowledge.
- If the context doesn't contain enough information, say so explicitly.
- Do not speculate beyond what is stated in the transcripts.
- When citing figures, mention the company, quarter, and speaker when known.
- Format numbers clearly (e.g. "14.7% EBITDA margin", "$160M TCV").
"""

RAG_CONTEXT_TEMPLATE = """### Transcript Excerpts

{context_blocks}

---
Based only on the excerpts above, answer this question:
**{question}**"""

NO_CONTEXT_RESPONSE = """I couldn't find relevant information."""


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Stateless RAG engine with guardrail integration.
    Conversation history is managed externally and passed in per call.
    """

    def __init__(self) -> None:
        self.embedder   = Embedder()
        self.store      = VectorStore()
        self.llm        = LLMClient()
        self.guardrails = Guardrails()

    def query(
        self,
        question: str,
        history: ConversationHistory,
        filters: Optional[SearchFilters] = None,
    ) -> RAGResponse:
        """Full RAG pipeline with guardrails."""

        # ── Step 1: Guardrails ─────────────────────────────────────────
        guard = self.guardrails.check(question)
        if not guard.allowed:
            logger.info(f"Query blocked by guardrails: {guard.reason}")
            return RAGResponse(
                answer=guard.reason,
                sources=[],
                used_rag=False,
                query=question,
                rewritten_query=question,
                blocked=True,
                block_reason="intent_filter",
            )

        # ── Preset answer (e.g. greetings) — skip RAG entirely ────────
        if guard.preset_answer:
            history.add_user(guard.sanitized_query)
            history.add_assistant(guard.preset_answer)
            return RAGResponse(
                answer=guard.preset_answer,
                sources=[],
                used_rag=False,
                query=question,
                rewritten_query=question,
            )

        retrieval_query = guard.rewritten_query
        logger.debug(f"Retrieval query: '{retrieval_query}'")

        # ── Auto quarter filter ────────────────────────────────────────
        # If the caller didn't pass explicit filters, detect a quarter in
        # the question and lock ChromaDB to that quarter/FY so all top-k
        # slots come from the right period instead of spreading across quarters.
        if filters is None:
            qf = extract_quarter_filter(question)
            if qf:
                quarter, fiscal_year = qf
                filters = SearchFilters(quarter=quarter, fiscal_year=fiscal_year)
                logger.debug(f"Auto quarter filter: {quarter} {fiscal_year}")

        # ── Step 2: Embed ──────────────────────────────────────────────
        query_embedding = self.embedder.embed_single(retrieval_query)

        # ── Step 3: Retrieve ───────────────────────────────────────────
        results = self.store.search(
            query_embedding=query_embedding,
            filters=filters,
            top_k=settings.top_k_results,
        )

        # ── Step 4: Confidence filter ──────────────────────────────────
        relevant = [r for r in results if r.score <= settings.similarity_threshold]

        if not relevant:
            best_score = f"{results[0].score:.3f}" if results else "N/A"
            logger.info(
                f"No chunks above threshold "
                f"(best: {best_score}, threshold: {settings.similarity_threshold})"
            )
            history.add_user(guard.sanitized_query)
            history.add_assistant(NO_CONTEXT_RESPONSE)
            return RAGResponse(
                answer=NO_CONTEXT_RESPONSE,
                sources=[],
                used_rag=False,
                query=question,
                rewritten_query=retrieval_query,
            )

        logger.info(f"Retrieved {len(relevant)} chunks (best: {relevant[0].score:.3f})")

        # ── Step 5: Build context prompt ───────────────────────────────
        context_prompt = self._build_context_prompt(guard.sanitized_query, relevant)

        # ── Step 6: Generate ───────────────────────────────────────────
        history.add_user(context_prompt)
        response = self.llm.chat(history, system_prompt=RAG_SYSTEM_PROMPT)
        answer = response.content

        # Store clean question/answer in history (not the padded context)
        history.messages[-1].content = guard.sanitized_query
        history.add_assistant(answer)

        return RAGResponse(
            answer=answer,
            sources=relevant,
            used_rag=True,
            query=question,
            rewritten_query=retrieval_query,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context_prompt(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        blocks = []
        for i, r in enumerate(results, 1):
            m = r.metadata
            header = (
                f"[Source {i}] {m.get('company', '?')} | "
                f"{m.get('quarter', '?')} {m.get('fiscal_year', '?')} | "
                f"Section: {m.get('section', '?')} | "
                f"Date: {m.get('date', '?')}"
            )
            blocks.append(f"{header}\n{r.text}")

        return RAG_CONTEXT_TEMPLATE.format(
            context_blocks="\n\n---\n\n".join(blocks),
            question=question,
        )