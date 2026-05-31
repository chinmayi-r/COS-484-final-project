"""LLM client implementations for different providers."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from openai import OpenAI, AzureOpenAI
from src.utils.config import Config


class LLMClient(ABC):
    """Base abstract class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion for the provided prompt."""
        pass

    def generate_messages(self, messages: List[Dict[str, str]],
                          temperature: float = 0.0, max_tokens: int = 3000) -> str:
        """Generate a completion from a full messages list (system/user/assistant turns).
        Default implementation formats messages into a single prompt string."""
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt += content + "\n\n"
            elif role == "user":
                prompt += f"User:\n{content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant:\n{content}\n\n"
        return self.generate(prompt, temperature, max_tokens)


SYSTEM_PROMPT = (
    "You are a planning assistant that solves problems step by step. "
    "Follow these guidelines:\n"
    "1. Clearly prefix each action with 'ACTION:'\n"
    "2. Follow the output format demonstrated in the examples exactly "
    "(including STATE_JSON blocks immediately after every ACTION in JSON mode)\n"
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
        return self.generate_messages(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature, max_tokens,
        )

    def generate_messages(self, messages: List[Dict[str, str]],
                          temperature: float = 0.0, max_tokens: int = 3000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
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
        self.last_usage: Optional[dict] = None

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        return self.generate_messages(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature, max_tokens,
        )

    def generate_messages(self, messages: List[Dict[str, str]],
                          temperature: float = 0.0, max_tokens: int = 3000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.last_usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {e}"


class SandboxClient(LLMClient):
    """Client for the Princeton AI Sandbox.

    The sandbox exposes an Azure OpenAI-compatible endpoint:
      POST {base_url}/openai/deployments/{model}/chat/completions?api-version=...
    We use AzureOpenAI so the SDK constructs that path automatically.
    """

    def __init__(self):
        self.model = Config.SANDBOX_MODEL_NAME
        self.client = AzureOpenAI(
            azure_endpoint=Config.SANDBOX_BASE_URL,
            api_key=Config.SANDBOX_API_KEY,
            api_version=Config.SANDBOX_API_VERSION,
        )
        self.last_usage: Optional[dict] = None

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        return self.generate_messages(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature, max_tokens,
        )

    def generate_messages(self, messages: List[Dict[str, str]],
                          temperature: float = 0.0, max_tokens: int = 3000) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self.last_usage = {
                "prompt_tokens":     response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens":      response.usage.total_tokens,
            }
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Princeton AI Sandbox: {e}")
            return f"Error generating response: {e}"


class AnthropicClient(LLMClient):
    """Client for Anthropic Claude API."""

    def __init__(self):
        import anthropic as _anthropic
        self.model = Config.ANTHROPIC_MODEL_NAME
        self.client = _anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.last_usage: Dict = {}

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        return self.generate_messages(
            [{"role": "user", "content": prompt}],
            temperature, max_tokens,
        )

    def generate_messages(self, messages: List[Dict[str, str]],
                          temperature: float = 0.0, max_tokens: int = 3000) -> str:
        # Anthropic API separates system prompt from the messages list
        system_content = None
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                api_messages.append({"role": m["role"], "content": m["content"]})

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=api_messages,
        )
        if system_content:
            kwargs["system"] = system_content

        response = self.client.messages.create(**kwargs)
        self.last_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return response.content[0].text


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

    if provider == "sandbox":
        return SandboxClient()
    elif provider == "azure_openai":
        return AzureOpenAIClient()
    elif provider == "openai":
        return OpenAIClient()
    elif provider == "anthropic":
        return AnthropicClient()
    elif provider == "google":
        return GoogleClient()
    else:
        raise ValueError(f"Unknown provider: {provider}")
