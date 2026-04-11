"""
LLM abstraction layer.

Wraps the OpenAI client so the rest of the codebase never imports openai directly.
Supports:
  - Single-turn completions
  - Multi-turn chat with history
  - Streaming responses
  - Structured error handling with retries
"""

from __future__ import annotations

import time
import logging
from typing import Generator, Iterator
from dataclasses import dataclass, field

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from openai.types.chat import ChatCompletionMessageParam

from core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation."""
    role: str   # "system" | "user" | "assistant"
    content: str

    def to_api_dict(self) -> ChatCompletionMessageParam:
        return {"role": self.role, "content": self.content}  # type: ignore


@dataclass
class LLMResponse:
    """Wrapper around a completed LLM response."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ConversationHistory:
    """
    Manages multi-turn chat history.
    Automatically prepends the system prompt on first use.
    """
    messages: list[Message] = field(default_factory=list)
    _system_injected: bool = field(default=False, repr=False)

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def to_api_messages(self, system_prompt: str) -> list[ChatCompletionMessageParam]:
        system = [Message(role="system", content=system_prompt).to_api_dict()]
        return system + [m.to_api_dict() for m in self.messages]

    def clear(self) -> None:
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Thin wrapper around OpenAI's chat completions API.

    Responsibilities:
      - Authentication via settings
      - Retry logic for transient errors
      - Uniform error surface for callers
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds (doubles on each retry)

    def __init__(self) -> None:
        settings.validate()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        history: ConversationHistory,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """
        Send a full conversation history and get a complete response.
        Blocks until the model finishes.
        """
        prompt = system_prompt or settings.system_prompt
        messages = history.to_api_messages(prompt)
        return self._complete(messages)

    def chat_stream(
        self,
        history: ConversationHistory,
        system_prompt: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream a response token-by-token.
        Yields string chunks as they arrive.
        """
        prompt = system_prompt or settings.system_prompt
        messages = history.to_api_messages(prompt)
        yield from self._stream(messages)

    def simple(self, user_message: str, system_prompt: str | None = None) -> LLMResponse:
        """Single-turn convenience method — no history required."""
        history = ConversationHistory()
        history.add_user(user_message)
        return self.chat(history, system_prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complete(self, messages: list[ChatCompletionMessageParam]) -> LLMResponse:
        """Non-streaming completion with retry logic."""
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                choice = response.choices[0]
                return LLMResponse(
                    content=choice.message.content or "",
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    finish_reason=choice.finish_reason or "unknown",
                )

            except RateLimitError as e:
                wait = self.RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited. Retrying in {wait}s... (attempt {attempt + 1})")
                time.sleep(wait)
                last_error = e

            except APITimeoutError as e:
                logger.warning(f"Request timed out (attempt {attempt + 1})")
                last_error = e

            except APIConnectionError as e:
                logger.error("Connection error — check your network.")
                raise RuntimeError("Could not connect to OpenAI API.") from e

            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise RuntimeError(f"LLM API error: {e}") from e

        raise RuntimeError(
            f"LLM request failed after {self.MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def _stream(self, messages: list[ChatCompletionMessageParam]) -> Iterator[str]:
        """Streaming completion. Yields text chunks."""
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        except RateLimitError as e:
            raise RuntimeError("Rate limit exceeded. Please wait and try again.") from e
        except APIConnectionError as e:
            raise RuntimeError("Could not connect to OpenAI API.") from e
        except APIError as e:
            raise RuntimeError(f"LLM API error: {e}") from e