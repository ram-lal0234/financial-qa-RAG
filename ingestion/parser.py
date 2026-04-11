"""
ingestion/parser.py

Extracts raw text and structured metadata from earnings call transcript PDFs.

Handles:
  - Text extraction via pdfplumber (better than pypdf for formatted docs)
  - Metadata inference: company, quarter, fiscal year, date
  - Section detection: cover page / management remarks / Q&A
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TranscriptMetadata:
    company: str
    ticker: str
    quarter: str           # "Q1", "Q2", "Q3", "Q4"
    fiscal_year: str       # "FY25", "FY24" etc.
    calendar_year: int     # 2024
    date: str              # "2024-07-31"
    source_file: str       # original filename
    speakers: list[str] = field(default_factory=list)


@dataclass
class RawSection:
    section_type: str      # "management_remarks" | "qa" | "cover"
    text: str


@dataclass
class ParsedTranscript:
    metadata: TranscriptMetadata
    sections: list[RawSection]
    full_text: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TranscriptParser:
    """
    Parses a single earnings call transcript PDF into structured sections.

    Design notes:
    - pdfplumber is used over pypdf because it preserves text layout better
      for multi-column formatted PDFs like these BSE filings.
    - Metadata is inferred from text patterns rather than requiring a
      structured filename convention, making it robust to varied sources.
    """

    # Patterns for metadata extraction
    _QUARTER_PATTERN = re.compile(r'\b(Q[1-4])\b', re.IGNORECASE)
    _FY_PATTERN = re.compile(r'\bFY\s*\'?(\d{2,4})\b', re.IGNORECASE)
    _DATE_PATTERN = re.compile(r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)[,\s]+(\d{4})', re.IGNORECASE)
    _TICKER_PATTERN = re.compile(r'(?:Scrip\s+(?:ID|Code|Symbol)\s*[:\-]?\s*)([A-Z0-9]+)', re.IGNORECASE)
    _COMPANY_PATTERN = re.compile(r'^(.+?)\s+(?:Limited|Ltd\.?|Inc\.?|Corp\.?)\s+(?:Q[1-4]|FY)', re.IGNORECASE | re.MULTILINE)

    # Section boundary markers
    _QA_MARKERS = [
        "question and answer", "q&a session", "question-and-answer",
        "we will now begin", "open for questions", "first question",
        "moderator:", "operator:",
    ]
    _CLOSING_MARKERS = [
        "that concludes", "end of question", "no further questions",
        "thank you for joining", "you may now disconnect",
    ]

    def parse(self, pdf_path: str | Path) -> ParsedTranscript:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing: {path.name}")

        full_text = self._extract_text(path)
        metadata = self._extract_metadata(full_text, path.name)
        sections = self._split_sections(full_text)

        return ParsedTranscript(
            metadata=metadata,
            sections=sections,
            full_text=full_text,
        )

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path) -> str:
        """Extract all text from PDF, page by page."""
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
        return "\n\n".join(pages)

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _extract_metadata(self, text: str, filename: str) -> TranscriptMetadata:
        first_2000 = text[:2000]  # metadata almost always in first pages

        quarter = self._extract_quarter(first_2000)
        fiscal_year = self._extract_fiscal_year(first_2000)
        date_str, calendar_year = self._extract_date(first_2000)
        company, ticker = self._extract_company_ticker(first_2000, filename)
        speakers = self._extract_speakers(text)

        return TranscriptMetadata(
            company=company,
            ticker=ticker,
            quarter=quarter,
            fiscal_year=fiscal_year,
            calendar_year=calendar_year,
            date=date_str,
            source_file=filename,
            speakers=speakers,
        )

    def _extract_quarter(self, text: str) -> str:
        m = self._QUARTER_PATTERN.search(text)
        return m.group(1).upper() if m else "UNKNOWN"

    def _extract_fiscal_year(self, text: str) -> str:
        m = self._FY_PATTERN.search(text)
        if m:
            yr = m.group(1)
            return f"FY{yr}" if not yr.startswith("FY") else yr
        return "UNKNOWN"

    def _extract_date(self, text: str) -> tuple[str, int]:
        month_map = {
            "january": "01", "february": "02", "march": "03",
            "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09",
            "october": "10", "november": "11", "december": "12",
        }
        m = self._DATE_PATTERN.search(text)
        if m:
            day, month_name, year = m.group(1), m.group(2).lower(), m.group(3)
            month = month_map.get(month_name, "01")
            return f"{year}-{month}-{int(day):02d}", int(year)
        return "UNKNOWN", 0

    def _extract_company_ticker(self, text: str, filename: str) -> tuple[str, str]:
        # Ticker: look for "Scrip ID/Code/Symbol: BSOFT"
        ticker_m = self._TICKER_PATTERN.search(text)
        ticker = ticker_m.group(1).strip() if ticker_m else "UNKNOWN"

        # Company name: "XYZ Limited Q1/FY..." pattern (title page)
        company_m = self._COMPANY_PATTERN.search(text)
        if company_m:
            company = company_m.group(1).strip().split("\n")[-1].strip()
            return company, ticker

        # Fallback: "For XYZ Limited" (signature block)
        sig_m = re.search(
            r'For\s+([A-Z][a-zA-Z\s]+?)\s+(?:Limited|Ltd\.?)',
            text[:3000], re.IGNORECASE,
        )
        if sig_m:
            return sig_m.group(1).strip(), ticker

        # Never use filename/folder — scrip codes like "532400" are not company names
        return "UNKNOWN", ticker

    def _extract_speakers(self, text: str) -> list[str]:
        """Extract named management speakers (not analysts) from transcript."""
        # Look for MANAGEMENT: section listing speakers
        mgmt_section = re.search(
            r'MANAGEMENT\s*:\s*(.*?)(?=\n\n|\Z)',
            text[:3000],
            re.IGNORECASE | re.DOTALL,
        )
        if not mgmt_section:
            return []

        # Each speaker line: "MR. ANGAN GUHA, CHIEF EXECUTIVE OFFICER"
        names = re.findall(
            r'(?:MR\.|MS\.|MRS\.)\s+([A-Z][A-Z\s]+?)(?:,|\n)',
            mgmt_section.group(1),
            re.IGNORECASE,
        )
        return [n.strip().title() for n in names if n.strip()]

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    def _split_sections(self, text: str) -> list[RawSection]:
        """
        Split transcript into logical sections:
          1. Cover / boilerplate (skip for chunking)
          2. Management remarks
          3. Q&A session
        """
        text_lower = text.lower()
        sections = []

        # Find Q&A start
        qa_start = None
        for marker in self._QA_MARKERS:
            idx = text_lower.find(marker)
            if idx != -1:
                # Walk back to start of that paragraph
                qa_start = text.rfind("\n", 0, idx) + 1
                break

        if qa_start is None:
            # No clear Q&A boundary — treat whole thing as remarks
            sections.append(RawSection(section_type="management_remarks", text=text))
            return sections

        # Find closing
        closing_start = len(text)
        for marker in self._CLOSING_MARKERS:
            idx = text_lower.rfind(marker)
            if idx != -1 and idx > qa_start:
                closing_start = min(closing_start, idx)

        # Split
        remarks_text = text[:qa_start].strip()
        qa_text = text[qa_start:closing_start].strip()

        if remarks_text:
            sections.append(RawSection(section_type="management_remarks", text=remarks_text))
        if qa_text:
            sections.append(RawSection(section_type="qa", text=qa_text))

        return sections