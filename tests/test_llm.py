"""
Tests for core/llm.py

Run with: pytest tests/test_llm.py -v
These tests mock the OpenAI client — no API key required.
"""

import pytest
from unittest.mock import MagicMock, patch
from core.llm import ConversationHistory, Message, LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# ConversationHistory tests
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def test_add_user_message(self):
        h = ConversationHistory()
        h.add_user("Hello")
        assert len(h) == 1
        assert h.messages[0].role == "user"
        assert h.messages[0].content == "Hello"

    def test_add_assistant_message(self):
        h = ConversationHistory()
        h.add_assistant("Hi there")
        assert h.messages[0].role == "assistant"

    def test_to_api_messages_injects_system(self):
        h = ConversationHistory()
        h.add_user("test")
        messages = h.to_api_messages("You are a helpful assistant.")
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_clear_resets_history(self):
        h = ConversationHistory()
        h.add_user("Q1")
        h.add_assistant("A1")
        h.clear()
        assert len(h) == 0

    def test_multi_turn_ordering(self):
        h = ConversationHistory()
        h.add_user("first question")
        h.add_assistant("first answer")
        h.add_user("second question")
        messages = h.to_api_messages("system")
        assert messages[1]["content"] == "first question"
        assert messages[2]["content"] == "first answer"
        assert messages[3]["content"] == "second question"


# ---------------------------------------------------------------------------
# LLMClient tests (mocked)
# ---------------------------------------------------------------------------

def make_mock_response(content: str = "mocked response") -> MagicMock:
    """Build a mock that looks like an OpenAI ChatCompletion response."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "gpt-4o-mini"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    return mock_response


@patch("core.llm.OpenAI")
@patch("core.llm.settings")
class TestLLMClient:
    def test_simple_completion(self, mock_settings, mock_openai_class):
        mock_settings.openai_api_key = "sk-test"
        mock_settings.model = "gpt-4o-mini"
        mock_settings.max_tokens = 100
        mock_settings.temperature = 0.2
        mock_settings.system_prompt = "You are an assistant."
        mock_settings.validate = MagicMock()

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_response("Revenue grew 5%.")

        llm = LLMClient()
        response = llm.simple("What was the revenue growth?")

        assert response.content == "Revenue grew 5%."
        assert response.total_tokens == 30
        assert response.finish_reason == "stop"

    def test_chat_with_history(self, mock_settings, mock_openai_class):
        mock_settings.openai_api_key = "sk-test"
        mock_settings.model = "gpt-4o-mini"
        mock_settings.max_tokens = 100
        mock_settings.temperature = 0.2
        mock_settings.system_prompt = "You are a financial assistant."
        mock_settings.validate = MagicMock()

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_response("EBITDA was 14.7%.")

        history = ConversationHistory()
        history.add_user("What was the EBITDA margin?")

        llm = LLMClient()
        response = llm.chat(history)
        assert "14.7" in response.content