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
        max_tokens: int = 25000,
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
        max_tokens: int = 25000,
    ) -> str:
        """Generate a completion using Claude with streaming."""
        try:
            chunks = []

            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            ) as stream:
                for text in stream.text_stream:
                    chunks.append(text)

                final_message = stream.get_final_message()

            print("\n========== LLM DEBUG ==========")
            print(f"MODEL: {self.model}")
            print(f"STOP REASON: {final_message.stop_reason}")
            print(f"USAGE: {final_message.usage}")
            print("================================\n")

            return "".join(chunks)

        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            return f"Error generating response: {e}"


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
