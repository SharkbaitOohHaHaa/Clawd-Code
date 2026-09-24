"""Tests for LLM providers."""

from __future__ import annotations

import unittest
from unittest.mock import ANY, MagicMock, patch

from src.providers import PROVIDER_INFO, get_provider_class
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.deepseek_provider import DeepSeekProvider
from src.providers.qwen_provider import QwenProvider
from src.providers.glm_provider import GLMProvider
from src.providers.minimax_provider import MinimaxProvider
from src.providers.openai_provider import OpenAIProvider
from src.providers.base import ChatMessage, ChatResponse

# Real no-redirect HTTP clients are covered in test_provider_sdk_policy.py. Each one builds an
# SSL context, which these constructor and parsing tests do not need.
_HTTP_CLIENT_PATCHES = (
    patch("src.providers.anthropic_provider.anthropic.DefaultHttpxClient"),
    patch("src.providers.openai_provider.DefaultHttpxClient"),
)


def setUpModule():
    for patcher in _HTTP_CLIENT_PATCHES:
        patcher.start()


def tearDownModule():
    for patcher in _HTTP_CLIENT_PATCHES:
        patcher.stop()


class TestChatMessage(unittest.TestCase):
    """Test ChatMessage dataclass."""

    def test_create_message(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello")

    def test_to_dict(self):
        """Test converting message to dict."""
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_dict()
        self.assertEqual(result, {"role": "user", "content": "Hello"})


class TestChatResponse(unittest.TestCase):
    """Test ChatResponse dataclass."""

    def test_create_response(self):
        """Test creating a chat response."""
        response = ChatResponse(
            content="Hello!",
            model="gpt-4",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
        )
        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "gpt-4")
        self.assertIsNone(response.reasoning_content)

    def test_response_with_reasoning(self):
        """Test response with reasoning content."""
        response = ChatResponse(
            content="Answer",
            model="glm-4.5",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="stop",
            reasoning_content="Reasoning process...",
        )
        self.assertEqual(response.reasoning_content, "Reasoning process...")


class TestAnthropicProvider(unittest.TestCase):
    """Test Anthropic provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = AnthropicProvider(api_key="test_key")
        self.assertEqual(provider.model, "claude-sonnet-4-6")
        self.assertEqual(provider.api_key, "test_key")

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = AnthropicProvider(api_key="test_key", model="claude-3-opus-20240229")
        self.assertEqual(provider.model, "claude-3-opus-20240229")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = AnthropicProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("claude-sonnet-4-20250514", models)
        self.assertIn("claude-3-5-sonnet-20241022", models)

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat(self, mock_anthropic):
        """Test synchronous chat."""
        # Setup mock
        mock_client = MagicMock(max_retries=0)
        mock_response = MagicMock()
        # Mock text block with type and text attributes
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello!"
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Test
        provider = AnthropicProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "claude-sonnet-4-20250514")
        self.assertEqual(response.finish_reason, "end_turn")

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_accepts_dict_messages(self, mock_anthropic):
        """Test synchronous chat with dict messages."""
        mock_client = MagicMock(max_retries=0)
        mock_response = MagicMock()
        # Mock text block with type and text attributes
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Hello!"
        mock_response.content = [mock_text_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="test_key")
        messages = [{"role": "user", "content": "Hi"}]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        mock_client.messages.create.assert_called_once()
        self.assertEqual(
            mock_client.messages.create.call_args.kwargs["messages"], messages
        )

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_stream_response_with_tool_use(self, mock_anthropic):
        """Structured streaming returns final text and tool uses."""
        mock_client = MagicMock(max_retries=0)
        mock_stream = MagicMock()
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = False
        mock_stream.text_stream = iter(["Hello", " world"])

        final_response = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "toolu_1"
        tool_block.name = "Read"
        tool_block.input = {"file_path": "README.md"}
        final_response.content = [text_block, tool_block]
        final_response.model = "claude-sonnet-4-20250514"
        final_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        final_response.stop_reason = "tool_use"
        mock_stream.get_final_message.return_value = final_response

        mock_client.messages.stream.return_value = mock_stream
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="test_key")
        chunks: list[str] = []
        response = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")],
            tools=[{"name": "Read", "description": "", "input_schema": {"type": "object"}}],
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello world")
        self.assertEqual(response.content, "Hello world")
        self.assertEqual(response.finish_reason, "tool_use")
        self.assertEqual(response.tool_uses[0]["name"], "Read")


class TestOpenAIProvider(unittest.TestCase):
    """Test OpenAI provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = OpenAIProvider(api_key="test_key")
        self.assertEqual(provider.model, "gpt-5.4")

    def test_custom_model(self):
        """Test provider with custom model."""
        provider = OpenAIProvider(api_key="test_key", model="gpt-4-turbo")
        self.assertEqual(provider.model, "gpt-4-turbo")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = OpenAIProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertIn("gpt-4", models)
        self.assertIn("gpt-4o", models)

    @patch("src.providers.openai_provider.OpenAI")
    def test_chat(self, mock_openai):
        """Test synchronous chat."""
        # Setup mock
        mock_client = MagicMock(max_retries=0)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Test
        provider = OpenAIProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.usage["total_tokens"], 15)

    @patch("src.providers.openai_provider.OpenAI")
    def test_chat_accepts_dict_messages(self, mock_openai):
        """Test synchronous chat with dict messages."""
        mock_client = MagicMock(max_retries=0)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test_key")
        messages = [{"role": "user", "content": "Hi"}]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["messages"], messages
        )

    @patch("src.providers.openai_provider.OpenAI")
    def test_chat_stream_response_rebuilds_tool_calls(self, mock_openai):
        """Streaming chunks are rebuilt into a final response with tool calls."""
        mock_client = MagicMock(max_retries=0)

        chunk1 = MagicMock()
        chunk1.model = "gpt-4"
        chunk1.usage = None
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].finish_reason = None
        chunk1.choices[0].delta.content = "Hello"
        chunk1.choices[0].delta.reasoning_content = None
        chunk1.choices[0].delta.tool_calls = []

        tool_call_delta = MagicMock()
        tool_call_delta.index = 0
        tool_call_delta.id = "call_1"
        tool_call_delta.function = MagicMock(name="function")
        tool_call_delta.function.name = "Read"
        tool_call_delta.function.arguments = '{"file_path":"README.md"}'

        chunk2 = MagicMock()
        chunk2.model = "gpt-4"
        chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].finish_reason = "tool_calls"
        chunk2.choices[0].delta.content = None
        chunk2.choices[0].delta.reasoning_content = None
        chunk2.choices[0].delta.tool_calls = [tool_call_delta]

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test_key")
        chunks: list[str] = []
        response = provider.chat_stream_response(
            [ChatMessage(role="user", content="Hi")],
            tools=[{"name": "Read", "description": "", "input_schema": {"type": "object"}}],
            on_text_chunk=chunks.append,
        )

        self.assertEqual("".join(chunks), "Hello")
        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(response.tool_uses[0]["name"], "Read")
        self.assertEqual(response.usage["total_tokens"], 15)


class TestDeepSeekProvider(unittest.TestCase):
    """Test DeepSeek provider wiring without live API calls."""

    def test_initialization(self):
        provider = DeepSeekProvider(api_key="test_key")
        self.assertEqual(provider.model, "deepseek-flash")
        self.assertEqual(provider.base_url, "https://api.deepseek.com")

    def test_custom_model_and_base_url(self):
        provider = DeepSeekProvider(
            api_key="test_key",
            base_url="https://example.test",
            model="deepseek-v4-pro",
        )
        self.assertEqual(provider.model, "deepseek-v4-pro")
        self.assertEqual(provider.base_url, "https://example.test")

    def test_get_available_models(self):
        provider = DeepSeekProvider(api_key="test_key")
        self.assertEqual(
            provider.get_available_models(),
            ["deepseek-flash", "deepseek-v4-pro"],
        )

    @patch("src.providers.openai_provider.OpenAI", return_value=MagicMock(max_retries=0))
    def test_client_uses_deepseek_endpoint(self, mock_openai):
        provider = DeepSeekProvider(api_key="test_key")
        _ = provider.client
        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.deepseek.com",
            max_retries=0,
            http_client=ANY,
        )

    def test_usage_includes_cache_and_reasoning_details(self):
        provider = DeepSeekProvider(api_key="test_key")
        usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130,
            prompt_cache_hit_tokens=40,
        )
        usage.prompt_tokens_details = MagicMock(cached_tokens=40)
        usage.completion_tokens_details = MagicMock(reasoning_tokens=12)

        result = provider._build_usage_dict(usage)

        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 30)
        self.assertEqual(result["cached_tokens"], 40)
        self.assertEqual(result["thought_tokens"], 12)
        self.assertEqual(result["total_tokens"], 130)


class TestQwenProvider(unittest.TestCase):
    """Test Qwen provider wiring without live API calls."""

    def test_initialization(self):
        provider = QwenProvider(api_key="test_key")
        self.assertEqual(provider.model, "qwen3.8-max")
        self.assertEqual(
            provider.base_url,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def test_custom_model_and_base_url(self):
        provider = QwenProvider(
            api_key="test_key",
            base_url="https://example.test",
            model="qwen3-coder-plus",
        )
        self.assertEqual(provider.model, "qwen3-coder-plus")
        self.assertEqual(provider.base_url, "https://example.test")

    def test_get_available_models(self):
        provider = QwenProvider(api_key="test_key")
        self.assertEqual(
            provider.get_available_models(),
            [
                "qwen3.8-max",
                "qwen3.8-flash",
                "qwen3.7-plus",
                "qwen3.7-flash",
                "qwen3-coder-plus",
                "qwen3-coder-flash",
            ],
        )

    @patch("src.providers.openai_provider.OpenAI", return_value=MagicMock(max_retries=0))
    def test_client_uses_singapore_endpoint(self, mock_openai):
        provider = QwenProvider(api_key="test_key")
        _ = provider.client
        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            max_retries=0,
            http_client=ANY,
        )

    @patch("src.providers.openai_provider.OpenAI", return_value=MagicMock(max_retries=0))
    def test_client_can_disable_sdk_retries_for_media_path(self, mock_openai):
        provider = QwenProvider(api_key="test_key", max_retries=0)
        _ = provider.client
        mock_openai.assert_called_once_with(
            api_key="test_key",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            max_retries=0,
            http_client=ANY,
        )


class TestGLMProvider(unittest.TestCase):
    """Test GLM provider."""

    def test_initialization(self):
        """Test provider initialization."""
        provider = GLMProvider(api_key="test_key")
        self.assertEqual(provider.model, "glm-5-turbo")
        self.assertEqual(
            provider.base_url,
            "https://open.bigmodel.cn/api/paas/v4",
        )

    def test_custom_model_and_base_url(self):
        """Test provider with custom model and endpoint."""
        provider = GLMProvider(
            api_key="test_key",
            base_url="https://example.test/v4",
            model="glm-4.6",
        )
        self.assertEqual(provider.model, "glm-4.6")
        self.assertEqual(provider.base_url, "https://example.test/v4")

    def test_get_available_models(self):
        """Test getting available models."""
        provider = GLMProvider(api_key="test_key")
        self.assertEqual(
            provider.get_available_models(),
            ["glm-5.2", "glm-5-turbo", "glm-5", "glm-4.7", "glm-4.6"],
        )

    @patch("src.providers.glm_provider.ZhipuAI", return_value=MagicMock(max_retries=0))
    def test_client_honors_configured_endpoint(self, mock_zhipu):
        provider = GLMProvider(
            api_key="test_key",
            base_url="https://example.test/v4",
        )
        _ = provider.client
        mock_zhipu.assert_called_once_with(
            api_key="test_key",
            base_url="https://example.test/v4",
            max_retries=0,
        )

    @patch("src.providers.glm_provider.ZhipuAI")
    def test_chat(self, mock_zhipu):
        """Test synchronous chat."""
        # Setup mock
        mock_client = MagicMock(max_retries=0)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.reasoning_content = None
        mock_response.model = "glm-4.5"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_zhipu.return_value = mock_client

        # Test
        provider = GLMProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Hi")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Hello!")
        self.assertEqual(response.model, "glm-4.5")
        self.assertIsNone(response.reasoning_content)

    @patch("src.providers.glm_provider.ZhipuAI")
    def test_chat_with_reasoning(self, mock_zhipu):
        """Test chat with reasoning content."""
        # Setup mock
        mock_client = MagicMock(max_retries=0)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_response.choices[0].message.reasoning_content = "Thinking..."
        mock_response.model = "glm-4.5"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response
        mock_zhipu.return_value = mock_client

        # Test
        provider = GLMProvider(api_key="test_key")
        messages = [ChatMessage(role="user", content="Complex question")]
        response = provider.chat(messages)

        self.assertEqual(response.content, "Answer")
        self.assertEqual(response.reasoning_content, "Thinking...")


class TestMinimaxProvider(unittest.TestCase):
    """Test MiniMax provider wiring without live API calls."""

    def test_initialization(self):
        provider = MinimaxProvider(api_key="test_key")
        self.assertEqual(provider.model, "MiniMax-M3")
        self.assertEqual(
            provider.base_url,
            "https://api.minimaxi.com/anthropic",
        )

    def test_custom_model_and_base_url(self):
        provider = MinimaxProvider(
            api_key="test_key",
            base_url="https://example.test/anthropic",
            model="MiniMax-M2.7",
        )
        self.assertEqual(provider.model, "MiniMax-M2.7")
        self.assertEqual(provider.base_url, "https://example.test/anthropic")

    def test_get_available_models(self):
        provider = MinimaxProvider(api_key="test_key")
        models = provider.get_available_models()
        self.assertEqual(models[0], "MiniMax-M3")
        self.assertIn("MiniMax-M2.7", models)
        self.assertIn("MiniMax-M2.7-highspeed", models)
        self.assertIn("MiniMax-M2.5", models)

    @patch("src.providers.minimax_provider.anthropic.Anthropic", return_value=MagicMock(max_retries=0))
    def test_client_uses_minimax_endpoint(self, mock_anthropic):
        provider = MinimaxProvider(api_key="test_key")
        _ = provider._ensure_client()
        mock_anthropic.assert_called_once_with(
            api_key="test_key",
            base_url="https://api.minimaxi.com/anthropic",
            max_retries=0,
            http_client=ANY,
        )


class TestChineseProviderParity(unittest.TestCase):
    """Built-in Chinese providers expose consistent registry/runtime metadata."""

    def test_registry_defaults_match_provider_instances(self):
        providers = {
            "deepseek": DeepSeekProvider,
            "qwen": QwenProvider,
            "glm": GLMProvider,
            "minimax": MinimaxProvider,
        }
        for name, provider_class in providers.items():
            with self.subTest(provider=name):
                info = PROVIDER_INFO[name]
                provider = provider_class(api_key="test_key")
                self.assertEqual(provider.model, info["default_model"])
                self.assertEqual(provider.base_url, info["default_base_url"])
                self.assertEqual(
                    provider.get_available_models(),
                    info["available_models"],
                )


class TestGetProviderClass(unittest.TestCase):
    """Test get_provider_class function."""

    def test_get_anthropic_provider(self):
        """Test getting Anthropic provider class."""
        cls = get_provider_class("anthropic")
        self.assertEqual(cls, AnthropicProvider)

    def test_get_openai_provider(self):
        """Test getting OpenAI provider class."""
        cls = get_provider_class("openai")
        self.assertEqual(cls, OpenAIProvider)

    def test_get_deepseek_provider(self):
        """Test getting DeepSeek provider class."""
        cls = get_provider_class("deepseek")
        self.assertEqual(cls, DeepSeekProvider)

    def test_get_qwen_provider(self):
        """Test getting Qwen provider class."""
        cls = get_provider_class("qwen")
        self.assertEqual(cls, QwenProvider)

    def test_get_glm_provider(self):
        """Test getting GLM provider class."""
        cls = get_provider_class("glm")
        self.assertEqual(cls, GLMProvider)

    def test_get_minimax_provider(self):
        """Test getting MiniMax provider class."""
        cls = get_provider_class("minimax")
        self.assertEqual(cls, MinimaxProvider)

    def test_get_unknown_provider(self):
        """Test getting unknown provider."""
        with self.assertRaises(ValueError) as context:
            get_provider_class("unknown")

        self.assertIn("Unknown provider", str(context.exception))


if __name__ == "__main__":
    unittest.main()
