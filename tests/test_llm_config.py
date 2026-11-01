import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rag.rag_chain import LLMConfigurationError, get_llm


class LLMConfigurationTests(unittest.TestCase):
    def test_ollama_is_the_default_provider(self):
        chat_ollama = Mock()
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(
                sys.modules,
                {"langchain_ollama": SimpleNamespace(ChatOllama=chat_ollama)},
            ):
                get_llm()

        chat_ollama.assert_called_once_with(model="llama3")

    def test_openai_requires_api_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            with self.assertRaisesRegex(
                LLMConfigurationError, "OPENAI_API_KEY is required"
            ):
                get_llm()

    def test_anthropic_requires_api_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic"}, clear=True):
            with self.assertRaisesRegex(
                LLMConfigurationError, "ANTHROPIC_API_KEY is required"
            ):
                get_llm()

    def test_gemini_requires_api_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True):
            with self.assertRaisesRegex(
                LLMConfigurationError, "GEMINI_API_KEY is required"
            ):
                get_llm()

    def test_gemini_uses_current_default_model(self):
        chat_gemini = Mock()
        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"},
            clear=True,
        ):
            with patch.dict(
                sys.modules,
                {
                    "langchain_google_genai": SimpleNamespace(
                        ChatGoogleGenerativeAI=chat_gemini
                    )
                },
            ):
                get_llm()

        chat_gemini.assert_called_once_with(
            model="gemini-3.8-flash",
            google_api_key="test-key",
        )

    def test_gemini_uses_configured_model_and_key(self):
        chat_gemini = Mock()
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "GEMINI_API_KEY": "test-key",
                "LLM_MODEL": "gemini-test",
            },
            clear=True,
        ):
            with patch.dict(
                sys.modules,
                {
                    "langchain_google_genai": SimpleNamespace(
                        ChatGoogleGenerativeAI=chat_gemini
                    )
                },
            ):
                get_llm()

        chat_gemini.assert_called_once_with(
            model="gemini-test",
            google_api_key="test-key",
        )

    def test_request_settings_override_environment(self):
        chat_gemini = Mock()
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "environment-key",
            },
            clear=True,
        ):
            with patch.dict(
                sys.modules,
                {
                    "langchain_google_genai": SimpleNamespace(
                        ChatGoogleGenerativeAI=chat_gemini
                    )
                },
            ):
                get_llm(
                    provider="gemini",
                    model="gemini-request",
                    api_key="request-key",
                )

        chat_gemini.assert_called_once_with(
            model="gemini-request",
            google_api_key="request-key",
        )

    def test_unknown_provider_is_rejected(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "local-api"}, clear=True):
            with self.assertRaisesRegex(
                LLMConfigurationError, "Unsupported LLM_PROVIDER"
            ):
                get_llm()


if __name__ == "__main__":
    unittest.main()
