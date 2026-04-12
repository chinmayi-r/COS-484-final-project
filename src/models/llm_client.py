"""LLM client implementations for different providers."""
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI, AzureOpenAI
from src.utils.config import Config


class LLMClient(ABC):
    """Base abstract class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion for the provided prompt."""
        pass


SYSTEM_PROMPT = (
    "You are a planning assistant that solves problems step by step. "
    "Follow these guidelines:\n"
    "1. Clearly prefix each action with 'ACTION:'\n"
    "2. After significant steps, show the current state with 'STATE x.y:' notation\n"
    "3. Include exploration and backtracking when appropriate\n"
    "4. End with a 'Plan summary:' section that lists all actions in order\n"
    "5. Be precise and detailed in your reasoning."
)


class AzureOpenAIClient(LLMClient):
    """Client for Azure OpenAI API."""

    def __init__(self):
        self.deployment_name = Config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure OpenAI API: {e}")
            return f"Error generating response: {e}"


class OpenAIClient(LLMClient):
    """Client for OpenAI API (supports gpt-4o-mini and other modern models)."""

    def __init__(self):
        self.model = Config.OPENAI_MODEL_NAME or "gpt-4o-mini"
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {e}"


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self):
        self.model = Config.ANTHROPIC_MODEL_NAME

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        return "Anthropic client not implemented yet."


class GoogleClient(LLMClient):
    """Client for Google Gemini API."""

    def __init__(self):
        self.model = Config.GOOGLE_MODEL_NAME

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        return "Google Gemini client not implemented yet."


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Factory function to get the appropriate LLM client."""
    if provider is None:
        provider = Config.get_default_provider()
    if provider is None:
        raise ValueError("No LLM provider configured. Check your .env file.")

    if provider == "azure_openai":
        return AzureOpenAIClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    elif provider == "google":
        return GoogleClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
