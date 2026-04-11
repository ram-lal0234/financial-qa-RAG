"""
ingestion/chunker.py

Converts parsed transcript sections into embeddable chunks.

Strategy:
  - management_remarks  → paragraph-based chunks (~400 tokens each)
  - qa                  → Q+A pair chunks (question + full answer together)

Each chunk carries full metadata for precise retrieval filtering.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from ingestion.parser import ParsedTranscript, RawSection, TranscriptMetadata

logger = logging.getLogger(__name__)

# Target chunk size in approximate tokens (1 token ≈ 4 chars)
TARGET_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS = 600
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single embeddable unit of text with full provenance metadata."""
    text: str
    metadata: dict

    @property
    def token_estimate(self) -> int:
        return len(self.text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class TranscriptChunker:
    """
    Splits a ParsedTranscript into Chunks ready for embedding.

    Q&A section:  Each question+answer pair becomes one chunk.
                  This preserves the analytical context — knowing what was
                  ASKED is as important as the answer for retrieval.

    Remarks:      Split by paragraph, merged until ~TARGET_CHUNK_TOKENS,
                  with one paragraph of overlap to preserve context.
    """

    # Pattern to detect speaker turns in transcript text
    # Matches: "Angan Guha:  Some text..." or "MODERATOR:  ..."
    _SPEAKER_PATTERN = re.compile(
        r'^([A-Z][a-zA-Z\s\.\-]+?):\s{2,}(.+?)(?=^[A-Z][a-zA-Z\s\.\-]+?:\s{2,}|\Z)',
        re.MULTILINE | re.DOTALL,
    )

    def chunk(self, transcript: ParsedTranscript) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        base_meta = self._base_metadata(transcript.metadata)

        for section in transcript.sections:
            if section.section_type == "management_remarks":
                chunks = self._chunk_remarks(section, base_meta)
            elif section.section_type == "qa":
                chunks = self._chunk_qa(section, base_meta)
            else:
                continue
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {transcript.metadata.company} "
            f"{transcript.metadata.quarter} {transcript.metadata.fiscal_year} "
            f"→ {len(all_chunks)} chunks"
        )
        return all_chunks

    # ------------------------------------------------------------------
    # Management remarks chunker
    # ------------------------------------------------------------------

    def _chunk_remarks(self, section: RawSection, base_meta: dict) -> list[Chunk]:
        """Split remarks into speaker-turn aware paragraphs."""
        turns = self._extract_speaker_turns(section.text)

        if not turns:
            # Fallback: simple paragraph split
            return self._paragraph_chunks(section.text, base_meta, "management_remarks")

        chunks = []
        buffer_speaker = ""
        buffer_text = ""

        for speaker, content in turns:
            content = content.strip()
            # Skip moderator boilerplate
            if speaker.lower() in ("moderator", "operator"):
                continue

            candidate = f"{speaker}: {content}"
            combined = f"{buffer_text}\n\n{candidate}".strip()

            if len(combined) // CHARS_PER_TOKEN > TARGET_CHUNK_TOKENS and buffer_text:
                # Flush current buffer
                chunks.append(self._make_chunk(
                    buffer_text,
                    {**base_meta, "section": "management_remarks", "speaker": buffer_speaker},
                ))
                buffer_text = candidate
                buffer_speaker = speaker
            else:
                buffer_text = combined
                buffer_speaker = speaker  # last speaker in buffer

        if buffer_text.strip():
            chunks.append(self._make_chunk(
                buffer_text,
                {**base_meta, "section": "management_remarks", "speaker": buffer_speaker},
            ))

        return chunks

    # ------------------------------------------------------------------
    # Q&A chunker
    # ------------------------------------------------------------------

    def _chunk_qa(self, section: RawSection, base_meta: dict) -> list[Chunk]:
        """
        Pair each analyst question with the management answer(s) that follow.
        Each pair becomes one chunk.
        """
        turns = self._extract_speaker_turns(section.text)
        if not turns:
            return self._paragraph_chunks(section.text, base_meta, "qa")

        chunks = []
        i = 0

        while i < len(turns):
            speaker, content = turns[i]

            # Detect analyst question (not management, not moderator)
            if self._is_analyst(speaker, base_meta.get("speakers", [])):
                # Collect all consecutive management responses
                analyst_text = f"[Question by {speaker}]\n{content.strip()}"
                answer_parts = []
                j = i + 1

                while j < len(turns):
                    resp_speaker, resp_content = turns[j]
                    if self._is_analyst(resp_speaker, base_meta.get("speakers", [])):
                        break  # next analyst question — end this pair
                    if resp_speaker.lower() not in ("moderator", "operator"):
                        answer_parts.append(f"[{resp_speaker}]\n{resp_content.strip()}")
                    j += 1

                if answer_parts:
                    full_text = analyst_text + "\n\n" + "\n\n".join(answer_parts)
                    chunks.append(self._make_chunk(
                        full_text,
                        {
                            **base_meta,
                            "section": "qa",
                            "analyst": speaker,
                        },
                    ))
                i = j
            else:
                i += 1

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_speaker_turns(self, text: str) -> list[tuple[str, str]]:
        """
        Extract (speaker, content) pairs from transcript text.
        Handles multi-line content per speaker.
        """
        # Split on lines that start with a name followed by ":"
        # e.g. "Angan Guha:   Some text"
        pattern = re.compile(
            r'^([A-Z][a-zA-Z\s\.\-]{2,40}):\s{1,}',
            re.MULTILINE,
        )

        matches = list(pattern.finditer(text))
        if not matches:
            return []

        turns = []
        for idx, match in enumerate(matches):
            speaker = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                turns.append((speaker, content))

        return turns

    def _is_analyst(self, speaker: str, mgmt_speakers: list[str]) -> bool:
        """
        An analyst is anyone who isn't management and isn't the moderator.
        """
        speaker_lower = speaker.lower()
        if speaker_lower in ("moderator", "operator"):
            return False

        mgmt_lower = [s.lower() for s in mgmt_speakers]
        # Check if any management name is contained in the speaker string
        for mgmt in mgmt_lower:
            mgmt_parts = mgmt.split()
            if any(part in speaker_lower for part in mgmt_parts if len(part) > 3):
                return False

        return True  # Not management → analyst

    def _paragraph_chunks(
        self,
        text: str,
        base_meta: dict,
        section: str,
    ) -> list[Chunk]:
        """Simple paragraph-based chunking as fallback."""
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        chunks = []
        buffer = ""

        for para in paragraphs:
            candidate = f"{buffer}\n\n{para}".strip()
            if len(candidate) // CHARS_PER_TOKEN > TARGET_CHUNK_TOKENS and buffer:
                chunks.append(self._make_chunk(buffer, {**base_meta, "section": section}))
                buffer = para
            else:
                buffer = candidate

        if buffer:
            chunks.append(self._make_chunk(buffer, {**base_meta, "section": section}))

        return chunks

    def _make_chunk(self, text: str, metadata: dict) -> Chunk:
        return Chunk(text=text.strip(), metadata=metadata)

    def _base_metadata(self, meta: TranscriptMetadata) -> dict:
        return {
            "company": meta.company,
            "ticker": meta.ticker,
            "quarter": meta.quarter,
            "fiscal_year": meta.fiscal_year,
            "calendar_year": str(meta.calendar_year),
            "date": meta.date,
            "source_file": meta.source_file,
            "speakers": meta.speakers,  # kept for analyst detection
        }