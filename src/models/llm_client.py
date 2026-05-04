"""LLM client implementation for Anthropic Claude."""

from abc import ABC, abstractmethod
from typing import Optional

from anthropic import Anthropic

from src.utils.config import Config


class LLMClient(ABC):
    """Base abstract class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 3000,
    ) -> str:
        """Generate a completion for the provided prompt."""
        pass


class AnthropicClient(LLMClient):
    """Thin client wrapper for Anthropic Claude."""

    def __init__(self):
        self.model = Config.ANTHROPIC_MODEL_NAME
        self.client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 3000,
    ) -> str:
        """Generate a completion using Claude."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            if not response.content:
                return ""

            return response.content[0].text

        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            return f"Error generating response: {str(e)}"


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Return the configured LLM client.

    Kept compatible with existing eval/core code that calls get_llm_client(args.provider).
    """

    if provider is None:
        provider = "anthropic"

    if provider not in {"anthropic", "claude"}:
        raise ValueError(
            f"Unsupported provider: {provider}. This project currently only supports Anthropic Claude."
        )

    return AnthropicClient()
