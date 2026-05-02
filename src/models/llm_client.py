"""LLM client implementations for different providers."""
from abc import ABC, abstractmethod
from typing import Optional

from src.utils.config import Config


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI API (v1.x SDK)."""

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        self.model = model or Config.OPENAI_MODEL_NAME or "gpt-4o-mini"
        self._client = OpenAI(api_key=Config.OPENAI_API_KEY)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AzureOpenAIClient(LLMClient):
    """Client for Azure OpenAI API (v1.x SDK)."""

    def __init__(self):
        from openai import AzureOpenAI
        self._client = AzureOpenAI(
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version=Config.AZURE_OPENAI_API_VERSION,
        )
        self.deployment = Config.AZURE_OPENAI_DEPLOYMENT_NAME

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": "You are a planning assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self):
        import anthropic
        self.model = Config.ANTHROPIC_MODEL_NAME or "claude-opus-4-7"
        self._client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


class GoogleClient(LLMClient):
    """Client for Google Gemini API."""

    def __init__(self):
        import google.generativeai as genai
        self.model_name = Config.GOOGLE_MODEL_NAME or "gemini-1.5-flash"
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048) -> str:
        config = {"temperature": temperature, "max_output_tokens": max_tokens}
        response = self._model.generate_content(prompt, generation_config=config)
        return response.text


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
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
