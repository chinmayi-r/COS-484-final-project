import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Anthropic-only config."""

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-6")

    @classmethod
    def get_available_providers(cls):
        """Return available LLM providers."""
        providers = []

        if cls.ANTHROPIC_API_KEY and cls.ANTHROPIC_MODEL_NAME:
            providers.append("anthropic")

        return providers

    @classmethod
    def get_default_provider(cls):
        """Return the default LLM provider."""
        if cls.ANTHROPIC_API_KEY:
            return "anthropic"

        return None
