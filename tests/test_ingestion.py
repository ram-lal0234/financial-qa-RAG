"""
Tests for ingestion pipeline components.
No API keys or PDFs required — uses synthetic text fixtures.
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestion.parser import TranscriptParser, TranscriptMetadata
from ingestion.chunker import TranscriptChunker, Chunk
from ingestion.parser import ParsedTranscript, RawSection
from core.vectorstore import SearchFilters, VectorStore


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

SAMPLE_TRANSCRIPT = """
Birlasoft Limited Q1 FY25 Earnings Conference Call
5.00pm IST, 31 July 2024

MANAGEMENT:
MR. ANGAN GUHA, CHIEF EXECUTIVE OFFICER & MANAGING DIRECTOR
MS. KAMINI SHAH, CHIEF FINANCIAL OFFICER

Scrip ID: BSOFT

Moderator:  Ladies and gentlemen, welcome to the earnings call.
            I now hand over to Mr. Abhinandan Singh.

Abhinandan Singh:  Thank you. Joining me are Angan Guha and Kamini Shah.

Angan Guha:  Good evening everyone. Our Q1 performance reflects a challenging
             operating environment. Revenue grew 3.8% year-on-year in constant
             currency. EBITDA margin was 14.7%.

Kamini Shah:  Thank you Angan. Revenue for Q1 was INR12,274 million.
              Cash and bank balances stood at $230 million.

Moderator:  We will now begin the question and answer session.

Krish Beriwal:  My question is about revenue decline. What caused the Q-o-Q fall?

Angan Guha:  The decline was primarily due to ERP project deferrals in manufacturing.
             Some projects that were expected to start in Q1 were pushed out.
             We expect a recovery in Q2.

Ravi Menon:  How is the BFSI segment performing?

Angan Guha:  BFSI has been growing strongly for the last 7 quarters.
             It grew 8.4% quarter-on-quarter this period.

Moderator:  That concludes our Q&A session. Thank you for joining.
"""


class TestTranscriptParser:
    def setup_method(self):
        self.parser = TranscriptParser()

    def test_extract_quarter(self):
        result = self.parser._extract_quarter("Birlasoft Q1 FY25 Earnings")
        assert result == "Q1"

    def test_extract_fiscal_year(self):
        result = self.parser._extract_fiscal_year("Q1 FY25 Earnings Call")
        assert "FY25" in result or "25" in result

    def test_extract_date(self):
        date_str, year = self.parser._extract_date("31 July 2024")
        assert year == 2024
        assert "2024" in date_str
        assert "07" in date_str

    def test_extract_speakers(self):
        speakers = self.parser._extract_speakers(SAMPLE_TRANSCRIPT)
        assert len(speakers) >= 1
        assert any("Guha" in s or "Shah" in s for s in speakers)

    def test_split_sections_finds_qa(self):
        sections = self.parser._split_sections(SAMPLE_TRANSCRIPT)
        section_types = [s.section_type for s in sections]
        assert "qa" in section_types

    def test_split_sections_finds_remarks(self):
        sections = self.parser._split_sections(SAMPLE_TRANSCRIPT)
        section_types = [s.section_type for s in sections]
        assert "management_remarks" in section_types


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

def make_transcript(remarks_text: str, qa_text: str) -> ParsedTranscript:
    from ingestion.parser import TranscriptMetadata
    meta = TranscriptMetadata(
        company="Birlasoft",
        ticker="BSOFT",
        quarter="Q1",
        fiscal_year="FY25",
        calendar_year=2024,
        date="2024-07-31",
        source_file="test.pdf",
        speakers=["Angan Guha", "Kamini Shah"],
    )
    sections = []
    if remarks_text:
        sections.append(RawSection(section_type="management_remarks", text=remarks_text))
    if qa_text:
        sections.append(RawSection(section_type="qa", text=qa_text))
    return ParsedTranscript(metadata=meta, sections=sections, full_text=remarks_text + qa_text)


class TestTranscriptChunker:
    def setup_method(self):
        self.chunker = TranscriptChunker()

    def test_chunks_have_required_metadata(self):
        transcript = make_transcript("Angan Guha:  Revenue grew this quarter.", "")
        chunks = self.chunker.chunk(transcript)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "company" in chunk.metadata
            assert chunk.metadata["company"] == "Birlasoft"
            assert "quarter" in chunk.metadata
            assert "fiscal_year" in chunk.metadata

    def test_qa_chunks_include_question(self):
        qa_text = (
            "Krish Beriwal:  What was the revenue growth?\n\n"
            "Angan Guha:  Revenue grew 3.8% in constant currency terms.\n\n"
            "Ravi Menon:  How about margins?\n\n"
            "Angan Guha:  EBITDA margin was 14.7%.\n\n"
        )
        transcript = make_transcript("", qa_text)
        chunks = self.chunker.chunk(transcript)
        qa_chunks = [c for c in chunks if c.metadata.get("section") == "qa"]
        assert len(qa_chunks) >= 1
        # Question context should be in chunk
        combined = " ".join(c.text for c in qa_chunks)
        assert "revenue" in combined.lower() or "margin" in combined.lower()

    def test_speaker_turn_extraction(self):
        text = "Angan Guha:  Revenue grew this quarter significantly.\n\nKamini Shah:  EBITDA was 14.7%.\n\n"
        turns = self.chunker._extract_speaker_turns(text)
        assert len(turns) == 2
        assert turns[0][0] == "Angan Guha"
        assert turns[1][0] == "Kamini Shah"

    def test_analyst_detection(self):
        mgmt = ["Angan Guha", "Kamini Shah"]
        assert self.chunker._is_analyst("Krish Beriwal", mgmt) is True
        assert self.chunker._is_analyst("Angan Guha", mgmt) is False
        assert self.chunker._is_analyst("Moderator", mgmt) is False

    def test_no_empty_chunks(self):
        transcript = make_transcript(
            "Angan Guha:  Some management remarks here about performance.\n\n",
            "Krish Beriwal:  A question.\n\nAngan Guha:  An answer to the question.\n\n",
        )
        chunks = self.chunker.chunk(transcript)
        for chunk in chunks:
            assert chunk.text.strip() != ""


# ---------------------------------------------------------------------------
# SearchFilters tests
# ---------------------------------------------------------------------------

class TestSearchFilters:
    def test_no_filters_returns_none_where(self):
        store = MagicMock(spec=VectorStore)
        store._build_where_clause = VectorStore._build_where_clause.__get__(store)
        result = store._build_where_clause(None)
        assert result is None

    def test_single_filter(self):
        store = MagicMock(spec=VectorStore)
        store._build_where_clause = VectorStore._build_where_clause.__get__(store)
        f = SearchFilters(company="Birlasoft")
        result = store._build_where_clause(f)
        assert result == {"company": {"$eq": "Birlasoft"}}

    def test_multiple_filters_uses_and(self):
        store = MagicMock(spec=VectorStore)
        store._build_where_clause = VectorStore._build_where_clause.__get__(store)
        f = SearchFilters(company="Birlasoft", quarter="Q1")
        result = store._build_where_clause(f)
        assert "$and" in result
        assert len(result["$and"]) == 2