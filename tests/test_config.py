"""Tests for configuration management."""

from __future__ import annotations

import unittest
from unittest.mock import patch, Mock
from pathlib import Path
import tempfile
import json
import base64
import os

from src.config import (
    get_config_path,
    get_default_config,
    load_config,
    save_config,
    get_provider_config,
    set_api_key,
    set_default_provider,
    get_default_provider,
    _encode_api_key,
    _decode_api_key,
    _provider_api_key_env,
    load_secrets_env,
)


class TestConfigPath(unittest.TestCase):
    """Test configuration path functions."""

    def test_get_config_path(self):
        """Test getting config path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.config.Path.home', return_value=Path(temp_dir)):
                path = get_config_path()
                expected = Path(temp_dir) / ".clawd" / "config.json"
                self.assertEqual(path, expected)

    def test_config_dir_created(self):
        """Test that config directory is created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            config_dir = home / ".clawd"
            self.assertFalse(config_dir.exists())

            with patch('src.config.Path.home', return_value=home):
                path = get_config_path()
                self.assertTrue(config_dir.exists())


class TestDefaultConfig(unittest.TestCase):
    """Test default configuration."""

    def test_get_default_config(self):
        """Test getting default config."""
        config = get_default_config()

        self.assertIn("default_provider", config)
        self.assertIn("providers", config)
        self.assertIn("anthropic", config["providers"])
        self.assertIn("openai", config["providers"])
        self.assertIn("deepseek", config["providers"])
        self.assertIn("qwen", config["providers"])
        self.assertIn("glm", config["providers"])
        self.assertIn("minimax", config["providers"])

    def test_default_provider_is_anthropic(self):
        """Test that default provider is Anthropic."""
        config = get_default_config()
        self.assertEqual(config["default_provider"], "anthropic")

    def test_default_models(self):
        """Test default models for providers."""
        config = get_default_config()
        self.assertEqual(
            config["providers"]["anthropic"]["default_model"],
            "claude-sonnet-4-6"
        )
        self.assertEqual(
            config["providers"]["openai"]["default_model"],
            "gpt-5.4"
        )
        self.assertEqual(
            config["providers"]["deepseek"]["default_model"],
            "deepseek-flash"
        )
        self.assertEqual(
            config["providers"]["qwen"]["default_model"],
            "qwen3.8-max"
        )
        self.assertEqual(
            config["providers"]["glm"]["default_model"],
            "glm-5-turbo"
        )
        self.assertEqual(
            config["providers"]["minimax"]["default_model"],
            "MiniMax-M3"
        )


class TestAPIKeyEncoding(unittest.TestCase):
    """Test API key encoding/decoding."""

    def test_encode_api_key(self):
        """Test API key encoding."""
        api_key = "test_api_key_123"
        encoded = _encode_api_key(api_key)
        expected = base64.b64encode(api_key.encode()).decode()
        self.assertEqual(encoded, expected)

    def test_decode_api_key(self):
        """Test API key decoding."""
        api_key = "test_api_key_123"
        encoded = base64.b64encode(api_key.encode()).decode()
        decoded = _decode_api_key(encoded)
        self.assertEqual(decoded, api_key)

    def test_decode_plain_text(self):
        """Test decoding plain text (not encoded)."""
        plain_key = "plain_api_key"
        decoded = _decode_api_key(plain_key)
        self.assertEqual(decoded, plain_key)


class TestLoadSaveConfig(unittest.TestCase):
    """Test loading and saving configuration."""

    def test_save_and_load_config(self):
        """Test save and load roundtrip."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                config = {
                    "default_provider": "glm",
                    "providers": {
                        "glm": {
                            "api_key": "test_key",
                            "base_url": "https://example.com",
                            "default_model": "glm-4"
                        }
                    }
                }

                save_config(config)
                loaded = load_config()

                self.assertEqual(loaded["default_provider"], "glm")
                self.assertEqual(
                    loaded["providers"]["glm"]["api_key"],
                    "test_key"
                )

    def test_load_config_creates_default(self):
        """Test that loading non-existent config creates default."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                config = load_config()

                self.assertIn("default_provider", config)
                self.assertIn("providers", config)
                self.assertTrue(config_path.exists())

    def test_api_keys_encoded_on_save(self):
        """Test that API keys are encoded when saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                config = {
                    "default_provider": "glm",
                    "providers": {
                        "glm": {
                            "api_key": "plain_text_key",
                            "base_url": "https://example.com",
                            "default_model": "glm-4"
                        }
                    }
                }

                save_config(config)

                # Read raw file to check encoding
                with open(config_path, 'r') as f:
                    raw_data = json.load(f)

                encoded_key = raw_data["providers"]["glm"]["api_key"]
                self.assertNotEqual(encoded_key, "plain_text_key")

                # Verify it can be decoded
                decoded_key = _decode_api_key(encoded_key)
                self.assertEqual(decoded_key, "plain_text_key")

    def test_api_keys_decoded_on_load(self):
        """Test that API keys are decoded when loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            # Create config with encoded key
            encoded_key = _encode_api_key("secret_key")
            raw_config = {
                "default_provider": "glm",
                "providers": {
                    "glm": {
                        "api_key": encoded_key,
                        "base_url": "https://example.com",
                        "default_model": "glm-4"
                    }
                }
            }

            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(raw_config, f)

            with patch('src.config.get_config_path', return_value=config_path):
                config = load_config()

                # API key should be decoded
                self.assertEqual(
                    config["providers"]["glm"]["api_key"],
                    "secret_key"
                )

    @unittest.skipIf(os.name == "nt", "POSIX file permission semantics differ on Windows")
    def test_config_file_permissions_restricted_on_save(self):
        """Test that saved config uses owner-only permissions on POSIX systems."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                save_config(get_default_config())
                mode = config_path.stat().st_mode & 0o777
                self.assertEqual(mode, 0o600)


class TestProviderConfig(unittest.TestCase):
    """Test provider-specific configuration."""

    def test_builtin_chinese_provider_env_keys(self):
        self.assertEqual(_provider_api_key_env("deepseek"), "DEEPSEEK_API_KEY")
        self.assertEqual(_provider_api_key_env("qwen"), "DASHSCOPE_API_KEY")
        self.assertEqual(_provider_api_key_env("glm"), "GLM_API_KEY")
        self.assertEqual(_provider_api_key_env("minimax"), "MINIMAX_API_KEY")

    def test_plugin_provider_env_key_is_shell_safe(self):
        self.assertEqual(_provider_api_key_env("remote-demo"), "REMOTE_DEMO_API_KEY")
        self.assertEqual(_provider_api_key_env("vendor.model"), "VENDOR_MODEL_API_KEY")

    def test_get_provider_config(self):
        """Test getting provider config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                glm_config = get_provider_config("glm")

                self.assertIn("api_key", glm_config)
                self.assertIn("base_url", glm_config)
                self.assertIn("default_model", glm_config)

    def test_environment_api_key_overrides_stored_key(self):
        """Test provider API keys can be supplied from the environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_api_key("anthropic", "stored_key")
                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "environment_key"}):
                    anthropic_config = get_provider_config("anthropic")

                self.assertEqual("environment_key", anthropic_config["api_key"])

    def test_new_provider_can_use_environment_key_with_legacy_config(self):
        """New registered providers work without rewriting an older config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps({
                    "default_provider": "anthropic",
                    "providers": {
                        "anthropic": {
                            "api_key": "",
                            "base_url": "https://api.anthropic.com",
                            "default_model": "claude-sonnet-4-6",
                        }
                    },
                }),
                encoding="utf-8",
            )

            with patch('src.config.get_config_path', return_value=config_path), patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "environment_deepseek_key"},
                clear=False,
            ):
                deepseek_config = get_provider_config("deepseek")

            self.assertEqual(deepseek_config["api_key"], "environment_deepseek_key")
            self.assertEqual(deepseek_config["base_url"], "https://api.deepseek.com")
            self.assertEqual(deepseek_config["default_model"], "deepseek-flash")

    def test_qwen_uses_dashscope_key_with_legacy_config(self):
        """Qwen uses the Model Studio key without rewriting an older config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps({
                    "default_provider": "anthropic",
                    "providers": {
                        "anthropic": {
                            "api_key": "",
                            "base_url": "https://api.anthropic.com",
                            "default_model": "claude-sonnet-4-6",
                        }
                    },
                }),
                encoding="utf-8",
            )

            with patch('src.config.get_config_path', return_value=config_path), patch.dict(
                os.environ,
                {"DASHSCOPE_API_KEY": "environment_qwen_key"},
                clear=False,
            ):
                qwen_config = get_provider_config("qwen")

            self.assertEqual(qwen_config["api_key"], "environment_qwen_key")
            self.assertEqual(
                qwen_config["base_url"],
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            )
            self.assertEqual(qwen_config["default_model"], "qwen3.8-max")

    def test_minimax_uses_env_key_with_legacy_config(self):
        """MiniMax is first-class even when config predates the provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps({
                    "default_provider": "anthropic",
                    "providers": {
                        "anthropic": {
                            "api_key": "",
                            "base_url": "https://api.anthropic.com",
                            "default_model": "claude-sonnet-4-6",
                        }
                    },
                }),
                encoding="utf-8",
            )

            with patch('src.config.get_config_path', return_value=config_path), patch.dict(
                os.environ,
                {"MINIMAX_API_KEY": "environment_minimax_key"},
                clear=False,
            ):
                minimax_config = get_provider_config("minimax")

            self.assertEqual(minimax_config["api_key"], "environment_minimax_key")
            self.assertEqual(
                minimax_config["base_url"],
                "https://api.minimaxi.com/anthropic",
            )
            self.assertEqual(minimax_config["default_model"], "MiniMax-M3")

    def test_get_unknown_provider(self):
        """Test getting unknown provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                with self.assertRaises(ValueError) as context:
                    get_provider_config("unknown")

                self.assertIn("Unknown provider", str(context.exception))


class TestSetAPIKey(unittest.TestCase):
    """Test setting API keys."""

    def test_set_api_key(self):
        """Test setting API key for provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_api_key("glm", "new_api_key")

                config = load_config()
                self.assertEqual(
                    config["providers"]["glm"]["api_key"],
                    "new_api_key"
                )

    def test_set_api_key_with_options(self):
        """Test setting API key with base URL and model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_api_key(
                    "glm",
                    "new_api_key",
                    base_url="https://custom.url",
                    default_model="custom-model"
                )

                config = load_config()
                self.assertEqual(
                    config["providers"]["glm"]["api_key"],
                    "new_api_key"
                )
                self.assertEqual(
                    config["providers"]["glm"]["base_url"],
                    "https://custom.url"
                )
                self.assertEqual(
                    config["providers"]["glm"]["default_model"],
                    "custom-model"
                )


class TestDefaultProvider(unittest.TestCase):
    """Test default provider management."""

    def test_set_default_provider(self):
        """Test setting default provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                set_default_provider("openai")

                provider = get_default_provider()
                self.assertEqual(provider, "openai")

    def test_get_default_provider(self):
        """Test getting default provider."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"

            with patch('src.config.get_config_path', return_value=config_path):
                provider = get_default_provider()
                self.assertEqual(provider, "anthropic")


class TestLoadSecretsEnv(unittest.TestCase):
    """Test loading the external secrets file (fake values only)."""

    def _load(self, content: bytes, preset: dict[str, str] | None = None) -> dict[str, str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_bytes(content)
            environ = {"CLAWD_SECRETS_FILE": str(env_file), **(preset or {})}
            with patch.dict(os.environ, environ, clear=True):
                self.assertEqual(load_secrets_env(), env_file)
                return dict(os.environ)

    def test_utf8_bom_does_not_corrupt_first_name(self):
        loaded = self._load(b"\xef\xbb\xbfFIRST_KEY=fake-one\r\nSECOND_KEY=fake-two\r\n")
        self.assertEqual(loaded.get("FIRST_KEY"), "fake-one")
        self.assertEqual(loaded.get("SECOND_KEY"), "fake-two")
        self.assertFalse(any(name.startswith("﻿") for name in loaded))

    def test_comments_and_invalid_names_are_skipped(self):
        loaded = self._load(
            b"#HASH_KEY=nope\n"
            b"  #INDENTED_KEY=nope\n"
            b";SEMI_KEY=nope\n"
            b"9DIGIT_KEY=nope\n"
            b"BAD NAME=nope\n"
            b"=nope\n"
            b"VALID_KEY=fake\n"
        )
        extra = set(loaded) - {"CLAWD_SECRETS_FILE"}
        self.assertEqual(extra, {"VALID_KEY"})

    def test_export_quotes_equals_and_spacing(self):
        loaded = self._load(
            b"export EXPORTED_KEY=fake-exported\n"
            b'QUOTED_KEY="fake quoted"\n'
            b"EQUALS_KEY=a=b=c\n"
            b"SPACED_KEY = fake-spaced \n"
        )
        self.assertEqual(loaded.get("EXPORTED_KEY"), "fake-exported")
        self.assertEqual(loaded.get("QUOTED_KEY"), "fake quoted")
        self.assertEqual(loaded.get("EQUALS_KEY"), "a=b=c")
        self.assertEqual(loaded.get("SPACED_KEY"), "fake-spaced")

    def test_existing_environment_values_are_not_overridden(self):
        loaded = self._load(b"PRESET_KEY=from-file\n", preset={"PRESET_KEY": "from-env"})
        self.assertEqual(loaded.get("PRESET_KEY"), "from-env")


if __name__ == '__main__':
    unittest.main()
