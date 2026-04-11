"""
ingestion/pipeline.py

Orchestrates the full ingestion flow:
  PDF → parse → chunk → embed → store in ChromaDB

Folder structure supported:
  Flat:    data/transcripts/*.pdf
  Nested:  data/transcripts/<scrip_code>/<quarter>/<file>.pdf
           e.g. 532400/Q1/transcript.pdf

Quarter is read from folder name (authoritative).
Company name is always extracted from PDF text — never from folder name.
Scrip code from folder is used only if ticker wasn't found in the PDF.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ingestion.parser import TranscriptParser
from ingestion.chunker import TranscriptChunker, Chunk
from core.embedder import Embedder
from core.vectorstore import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    source_file: str
    company: str
    quarter: str
    fiscal_year: str
    chunks_added: int
    success: bool
    error: str = ""


@dataclass
class BatchIngestResult:
    results: list[IngestResult] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total_files - self.successful

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks_added for r in self.results if r.success)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Single-responsibility pipeline for ingesting transcript PDFs.

    Idempotent: re-ingesting the same file overwrites existing chunks
    (ChromaDB upsert) rather than creating duplicates. Chunk IDs are
    deterministic hashes of (source_file + chunk_index).
    """

    def __init__(self) -> None:
        self.parser = TranscriptParser()
        self.chunker = TranscriptChunker()
        self.embedder = Embedder()
        self.store = VectorStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        pdf_path: str | Path,
        folder_hints: Optional[dict] = None,
    ) -> IngestResult:
        """
        Ingest a single PDF transcript.

        folder_hints: optional dict from _extract_folder_hints().
          - 'quarter'    → overrides PDF-parsed quarter (folder name is authoritative)
          - 'scrip_code' → used as ticker only if PDF didn't yield one
          Company name is ALWAYS taken from PDF text, never from folder.
        """
        path = Path(pdf_path)

        try:
            # 1. Parse PDF → structured transcript
            logger.info(f"[1/3] Parsing {path.name}...")
            transcript = self.parser.parse(path)
            meta = transcript.metadata

            # Apply folder hints
            if folder_hints:
                q = folder_hints.get("quarter", "").upper()
                if q in ("Q1", "Q2", "Q3", "Q4"):
                    logger.info(f"      → Quarter overridden by folder: {q}")
                    meta.quarter = q
                # Only use scrip code as ticker if PDF didn't find one
                scrip = folder_hints.get("scrip_code", "")
                if scrip and meta.ticker in ("UNKNOWN", ""):
                    meta.ticker = scrip

            logger.info(
                f"      → {meta.company} {meta.quarter} {meta.fiscal_year} "
                f"(ticker: {meta.ticker}, {len(transcript.sections)} sections)"
            )

            # 2. Chunk into embeddable units
            logger.info(f"[2/3] Chunking...")
            chunks = self.chunker.chunk(transcript)
            logger.info(f"      → {len(chunks)} chunks")

            if not chunks:
                return IngestResult(
                    source_file=path.name,
                    company=meta.company,
                    quarter=meta.quarter,
                    fiscal_year=meta.fiscal_year,
                    chunks_added=0,
                    success=False,
                    error="No chunks produced — check parser/chunker",
                )

            # 3. Embed + store
            logger.info(f"[3/3] Embedding and storing...")
            self._embed_and_store(chunks, path.name)
            logger.info(f"      → Done. Total in store: {self.store.count()}")

            return IngestResult(
                source_file=path.name,
                company=meta.company,
                quarter=meta.quarter,
                fiscal_year=meta.fiscal_year,
                chunks_added=len(chunks),
                success=True,
            )

        except FileNotFoundError as e:
            return IngestResult(
                source_file=path.name,
                company="UNKNOWN", quarter="UNKNOWN", fiscal_year="UNKNOWN",
                chunks_added=0, success=False, error=str(e),
            )
        except Exception as e:
            logger.exception(f"Ingestion failed for {path.name}")
            return IngestResult(
                source_file=path.name,
                company="UNKNOWN", quarter="UNKNOWN", fiscal_year="UNKNOWN",
                chunks_added=0, success=False, error=str(e),
            )

    def ingest_directory(self, dir_path: str | Path) -> BatchIngestResult:
        """
        Ingest all PDFs found in a directory, including nested subdirectories.

        Supports two layouts:
          Flat:    data/transcripts/*.pdf
          Nested:  data/transcripts/<scrip_code>/<quarter>/<file>.pdf
                   e.g. 532400/Q1/transcript.pdf

        Quarter from folder name is used to override PDF-parsed quarter.
        Company name is always taken from PDF text.
        """
        path = Path(dir_path)
        pdfs = sorted(path.rglob("*.pdf"))

        if not pdfs:
            logger.warning(f"No PDF files found (recursively) in {dir_path}")
            return BatchIngestResult()

        logger.info(f"Found {len(pdfs)} PDF(s) under {dir_path}")
        batch = BatchIngestResult()

        for pdf in pdfs:
            logger.info(f"\n{'─' * 50}")
            logger.info(f"Ingesting: {pdf.relative_to(path)}")
            hints = self._extract_folder_hints(pdf, path)
            result = self.ingest_file(pdf, folder_hints=hints)
            batch.results.append(result)

        return batch

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_folder_hints(self, pdf_path: Path, base_dir: Path) -> dict:
        """
        Infer quarter and scrip code from folder structure.

        Expected layout: <base>/<scrip_code>/<quarter>/<file>.pdf
        e.g.             data/532400/Q2/20241107_532400_Transcript.pdf
          → { scrip_code: "532400", quarter: "Q2" }

        Company name is intentionally NOT inferred here — always from PDF.
        """
        hints = {}
        try:
            parts = pdf_path.relative_to(base_dir).parts
            # parts = ("532400", "Q2", "filename.pdf")
            if len(parts) >= 3:
                hints["scrip_code"] = parts[-3]
                q = parts[-2].upper()
                if q in ("Q1", "Q2", "Q3", "Q4"):
                    hints["quarter"] = q
            elif len(parts) == 2:
                q = parts[-2].upper()
                if q in ("Q1", "Q2", "Q3", "Q4"):
                    hints["quarter"] = q
        except Exception:
            pass
        return hints

    def _embed_and_store(self, chunks: list[Chunk], source_file: str) -> None:
        texts     = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids       = [self._chunk_id(source_file, i) for i in range(len(chunks))]
        embeddings = self.embedder.embed(texts)
        self.store.add_chunks(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def _chunk_id(self, source_file: str, index: int) -> str:
        """
        Deterministic chunk ID = md5(filename + index).
        Ensures re-ingestion upserts rather than duplicates.
        """
        return hashlib.md5(f"{source_file}::{index}".encode()).hexdigest()