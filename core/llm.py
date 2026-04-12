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

import re
import time
import logging
from typing import Generator, Iterator
from dataclasses import dataclass, field

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError
from openai.types.chat import ChatCompletionMessageParam

from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model compatibility helpers
# ---------------------------------------------------------------------------

# Models in these families require max_completion_tokens instead of max_tokens
# and do not support a custom temperature (only the default of 1 is allowed).
_NEW_MODEL_RE = re.compile(
    r"^o\d|gpt-4\.1|gpt-4\.5|gpt-5", re.IGNORECASE
)


def _is_new_model(model: str) -> bool:
    return bool(_NEW_MODEL_RE.search(model))


def _tokens_kwarg(model: str, n: int) -> dict[str, int]:
    """Return the correct token-limit kwarg for the given model.

    Older models (gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-*) use ``max_tokens``.
    Newer models (o1/o3/o4 reasoning series, gpt-4.1-*, gpt-4.5-*, gpt-5-*)
    require ``max_completion_tokens``.
    """
    if _is_new_model(model):
        return {"max_completion_tokens": n}
    return {"max_tokens": n}


def _temperature_kwarg(model: str, temperature: float) -> dict[str, float]:
    """Return temperature kwarg only for models that support it.

    Newer models (o-series, gpt-4.1+, gpt-4.5+, gpt-5+) only accept the
    default temperature of 1 and raise an error if a different value is passed.
    For those models we omit the parameter entirely.
    """
    if _is_new_model(model):
        return {}
    return {"temperature": temperature}


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
                    **_tokens_kwarg(self.model, self.max_tokens),
                    **_temperature_kwarg(self.model, self.temperature),
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
                **_tokens_kwarg(self.model, self.max_tokens),
                **_temperature_kwarg(self.model, self.temperature),
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