"""
core/rag.py

Retrieval-Augmented Generation engine.

Flow:
  user query
    → embed query
    → retrieve top-k chunks from VectorStore (with optional filters)
    → filter by similarity threshold
    → build prompt with context
    → call LLM
    → return answer + source attribution
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from core.embedder import Embedder
from core.vectorstore import VectorStore, SearchResult, SearchFilters
from core.llm import LLMClient, ConversationHistory
from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RAGResponse:
    answer: str
    sources: list[SearchResult]
    used_rag: bool          # False if no relevant chunks found
    query: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a financial analyst assistant specializing in earnings call analysis.

Answer the user's question using ONLY the context provided below from earnings call transcripts.
Be precise, cite specific numbers and quotes where relevant.
If the context does not contain enough information to answer, say: "I don't have sufficient information in the provided transcripts to answer this question."
Do NOT speculate or use knowledge outside the provided context.
"""

RAG_CONTEXT_TEMPLATE = """
### Relevant Transcript Excerpts

{context_blocks}

---
Answer the following question based on the above excerpts:
"""

OUT_OF_SCOPE_RESPONSE = (
    "I can only answer questions about the earnings call transcripts that have been loaded. "
    "Your question doesn't seem to be related to the available financial data. "
    "Try asking about revenue, margins, guidance, deal wins, or other topics from the earnings calls."
)

NO_CONTEXT_RESPONSE = (
    "I couldn't find relevant information in the loaded transcripts to answer your question. "
    "This might be because:\n"
    "• The topic isn't covered in the available transcripts\n"
    "• Try rephrasing your question or specifying a company/quarter\n"
    "• Use `list` to see which transcripts are available"
)


# ---------------------------------------------------------------------------
# RAG Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Combines retrieval and generation into a single query interface.

    The engine is stateless — conversation history is managed externally
    (in the CLI session) and passed in on each call.
    """

    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = VectorStore()
        self.llm = LLMClient()

    def query(
        self,
        question: str,
        history: ConversationHistory,
        filters: Optional[SearchFilters] = None,
    ) -> RAGResponse:
        """
        Full RAG pipeline: retrieve → build context → generate answer.
        """
        # 1. Embed query
        query_embedding = self.embedder.embed_single(question)

        # 2. Retrieve relevant chunks
        results = self.store.search(
            query_embedding=query_embedding,
            filters=filters,
            top_k=settings.top_k_results,
        )

        # 3. Filter by similarity threshold
        relevant = [
            r for r in results
            if r.score <= settings.similarity_threshold
        ]

        # 4. If no relevant chunks → no-context response
        if not relevant:
            logger.info(f"No relevant chunks found (threshold={settings.similarity_threshold})")
            history.add_user(question)
            history.add_assistant(NO_CONTEXT_RESPONSE)
            return RAGResponse(
                answer=NO_CONTEXT_RESPONSE,
                sources=[],
                used_rag=False,
                query=question,
            )

        logger.info(f"Retrieved {len(relevant)} relevant chunks (best score: {relevant[0].score:.3f})")

        # 5. Build context-augmented prompt
        context_prompt = self._build_context_prompt(question, relevant)

        # 6. Add to history and call LLM
        history.add_user(context_prompt)
        response = self.llm.chat(history, system_prompt=RAG_SYSTEM_PROMPT)
        answer = response.content

        # Store clean question/answer in history (not the full context blob)
        history.messages[-1].content = question   # replace context-padded msg
        history.add_assistant(answer)

        return RAGResponse(
            answer=answer,
            sources=relevant,
            used_rag=True,
            query=question,
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
        for i, result in enumerate(results, 1):
            meta = result.metadata
            header = (
                f"[{i}] {meta.get('company', '?')} | "
                f"{meta.get('quarter', '?')} {meta.get('fiscal_year', '?')} | "
                f"Section: {meta.get('section', '?')}"
            )
            blocks.append(f"{header}\n{result.text}")

        context = RAG_CONTEXT_TEMPLATE.format(
            context_blocks="\n\n---\n\n".join(blocks)
        )
        return context + f"\n**Question:** {question}"