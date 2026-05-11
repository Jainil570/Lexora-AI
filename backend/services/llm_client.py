"""LLM Provider abstraction layer.

Supports Ollama (free, local) and OpenAI (optional, paid).
Configured via environment variables.
"""
import os
import logging
from typing import Protocol

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        ...


class OllamaProvider:
    """Local Ollama LLM provider (free, no API key needed)."""

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5")

    async def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        retries = 2
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"Ollama attempt {attempt + 1}/{retries + 1} failed: {e}"
                )
                if attempt == retries:
                    raise RuntimeError(
                        f"Ollama generation failed after {retries + 1} attempts: {e}"
                    ) from e
            except Exception as e:
                logger.error(f"Unexpected Ollama error: {e}")
                raise RuntimeError(f"Ollama generation error: {e}") from e

        raise RuntimeError("Ollama generation failed unexpectedly")


class OpenAIProvider:
    """OpenAI LLM provider (optional, requires API key)."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            logger.warning("OpenAI API key not set. OpenAI provider will fail.")

    async def generate(self, system: str, user: str, temperature: float = 0.2) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }

        retries = 2
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        url, json=payload, headers=headers
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                logger.warning(
                    f"OpenAI attempt {attempt + 1}/{retries + 1} failed: {e}"
                )
                if attempt == retries:
                    raise RuntimeError(
                        f"OpenAI generation failed after {retries + 1} attempts: {e}"
                    ) from e
            except Exception as e:
                logger.error(f"Unexpected OpenAI error: {e}")
                raise RuntimeError(f"OpenAI generation error: {e}") from e

        raise RuntimeError("OpenAI generation failed unexpectedly")


def get_llm_provider() -> LLMProvider:
    """Factory function to get the configured LLM provider."""
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider_name == "openai":
        logger.info("Using OpenAI LLM provider")
        return OpenAIProvider()
    else:
        logger.info("Using Ollama LLM provider")
        return OllamaProvider()
