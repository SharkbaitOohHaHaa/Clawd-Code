from __future__ import annotations

import unittest
from unittest.mock import patch

from src.cli import handle_login
from src.providers import clear_plugin_providers, register_plugin_provider
from src.providers.base import BaseProvider, ChatResponse


class LocalProvider(BaseProvider):
    def chat(self, messages, tools=None, **kwargs):
        return ChatResponse(
            content="local",
            model=self.model,
            usage={},
            finish_reason="stop",
        )

    def chat_stream(self, messages, tools=None, **kwargs):
        if False:
            yield ""

    def get_available_models(self):
        return ["local-model"]


class ProviderExtensionCLITests(unittest.TestCase):
    def tearDown(self) -> None:
        clear_plugin_providers()

    def test_local_provider_login_skips_api_key_prompt(self) -> None:
        register_plugin_provider(
            "local-demo",
            LocalProvider,
            {
                "label": "Local Demo",
                "default_base_url": "http://127.0.0.1:11434/v1",
                "default_model": "local-model",
                "available_models": ["local-model"],
                "requires_api_key": False,
                "local_only": True,
            },
            plugin_name="test-plugin",
            artifact_sha256="a" * 64,
        )

        with patch("src.cli._show_provider_defaults_table"), patch(
            "src.cli.Prompt.ask",
            side_effect=[
                "local-demo",
                "http://127.0.0.1:11434/v1",
                "local-model",
            ],
        ) as prompt, patch("src.config.set_api_key") as set_api_key, patch(
            "src.config.set_default_provider"
        ) as set_default_provider:
            result = handle_login()

        self.assertEqual(result, 0)
        self.assertEqual(prompt.call_count, 3)
        set_api_key.assert_called_once_with(
            "local-demo",
            api_key="",
            base_url="http://127.0.0.1:11434/v1",
            default_model="local-model",
        )
        set_default_provider.assert_called_once_with("local-demo")


if __name__ == "__main__":
    unittest.main()
