"""
Central configuration management.
All settings come from environment variables (via .env file).
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

    # System behaviour
    system_prompt: str = field(default_factory=lambda: os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a financial analyst assistant specializing in earnings call analysis. "
            "You answer questions clearly, concisely, and only based on the context provided. "
            "If you do not have enough information to answer, say so explicitly. "
            "Do not speculate beyond the data given."
        ),
    ))

    # ChromaDB (used in Part 2)
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_DIR", "./data/chroma"))
    collection_name: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "earnings_transcripts"))

    # RAG (used in Part 2)
    top_k_results: int = int(os.getenv("TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

    def validate(self) -> None:
        """Raise early if critical config is missing."""
        if not self.openai_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )


# Singleton — import this everywhere
settings = Settings()