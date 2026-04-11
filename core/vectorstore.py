"""
core/vectorstore.py

ChromaDB abstraction layer.

Why abstracted:
  - Swapping ChromaDB → Pinecone/pgvector for production requires changing
    only this file, not the RAG engine or ingestion pipeline.
  - Encapsulates ChromaDB's quirky filter syntax in one place.

Metadata filter support:
  - company     (exact match)
  - quarter     (exact match: "Q1", "Q2" ...)
  - fiscal_year (exact match: "FY25")
  - section     ("management_remarks" | "qa")
  - date        (exact match)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

import logging
import chromadb

# Silence noisy ChromaDB telemetry errors
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
from chromadb.config import Settings as ChromaSettings

from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float        # cosine distance (lower = more similar)
    chunk_id: str


@dataclass
class SearchFilters:
    company: Optional[str] = None
    quarter: Optional[str] = None
    fiscal_year: Optional[str] = None
    section: Optional[str] = None


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Manages the ChromaDB collection for earnings transcript chunks.

    Persistence: vectors are written to disk at settings.chroma_persist_dir
    so they survive restarts — no re-embedding needed after first ingest.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            f"VectorStore ready: '{settings.collection_name}' "
            f"({self._collection.count()} chunks)"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """
        Add pre-embedded chunks to the collection.
        ChromaDB upserts — safe to re-run on same data.
        """
        if not texts:
            return

        # ChromaDB metadata values must be str | int | float | bool
        clean_metadatas = [self._sanitize_metadata(m) for m in metadatas]

        self._collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_metadatas,
            ids=ids,
        )
        logger.info(f"Upserted {len(texts)} chunks into collection.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        filters: Optional[SearchFilters] = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Retrieve the top-k most similar chunks.
        Optionally filter by company / quarter / fiscal_year / section.
        """
        where = self._build_where_clause(filters)

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        return self._parse_results(results)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[dict]:
        """Return unique (company, quarter, fiscal_year) combos in the store."""
        if self.count() == 0:
            return []

        results = self._collection.get(include=["metadatas"])
        seen = set()
        docs = []
        for meta in results["metadatas"]:
            key = (meta.get("company"), meta.get("quarter"), meta.get("fiscal_year"))
            if key not in seen:
                seen.add(key)
                docs.append({
                    "company": meta.get("company"),
                    "quarter": meta.get("quarter"),
                    "fiscal_year": meta.get("fiscal_year"),
                    "date": meta.get("date"),
                    "source_file": meta.get("source_file"),
                })
        return sorted(docs, key=lambda x: (x["company"], x["fiscal_year"], x["quarter"]))

    def delete_document(self, company: str, quarter: str, fiscal_year: str) -> int:
        """Delete all chunks for a specific transcript."""
        results = self._collection.get(
            where={"$and": [
                {"company": company},
                {"quarter": quarter},
                {"fiscal_year": fiscal_year},
            ]},
            include=[],
        )
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunks for {company} {quarter} {fiscal_year}")
        return len(ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_where_clause(self, filters: Optional[SearchFilters]) -> Optional[dict]:
        if not filters:
            return None

        conditions = []
        if filters.company:
            conditions.append({"company": {"$eq": filters.company}})
        if filters.quarter:
            conditions.append({"quarter": {"$eq": filters.quarter.upper()}})
        if filters.fiscal_year:
            conditions.append({"fiscal_year": {"$eq": filters.fiscal_year.upper()}})
        if filters.section:
            conditions.append({"section": {"$eq": filters.section}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _parse_results(self, raw: dict) -> list[SearchResult]:
        results = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        for doc, meta, dist, chunk_id in zip(docs, metas, distances, ids):
            results.append(SearchResult(
                text=doc,
                metadata=meta,
                score=dist,
                chunk_id=chunk_id,
            ))
        return results

    def _sanitize_metadata(self, meta: dict) -> dict:
        """ChromaDB only accepts scalar metadata values."""
        clean = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                clean[k] = ", ".join(str(i) for i in v)
            else:
                clean[k] = str(v)
        return clean