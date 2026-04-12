import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
    
    # Anthropic (Claude)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL_NAME = os.getenv("ANTHROPIC_MODEL_NAME")
    
    # Google (Gemini)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME")

    # Princeton AI Sandbox (Portkey gateway — OpenAI-compatible)
    SANDBOX_API_KEY   = os.getenv("SANDBOX_API_KEY")
    SANDBOX_BASE_URL  = os.getenv("SANDBOX_BASE_URL", "https://api.portkey.ai/v1")
    SANDBOX_MODEL_NAME = os.getenv("SANDBOX_MODEL_NAME", "gpt-4o-mini")

    @classmethod
    def get_available_providers(cls):
        """Returns a list of available LLM providers based on configurations."""
        providers = []

        if cls.SANDBOX_API_KEY:
            providers.append("sandbox")

        if cls.AZURE_OPENAI_ENDPOINT and cls.AZURE_OPENAI_API_KEY and cls.AZURE_OPENAI_DEPLOYMENT_NAME:
            providers.append("azure_openai")

        if cls.OPENAI_API_KEY and cls.OPENAI_MODEL_NAME:
            providers.append("openai")
            
        if cls.ANTHROPIC_API_KEY and cls.ANTHROPIC_MODEL_NAME:
            providers.append("anthropic")
            
        if cls.GOOGLE_API_KEY and cls.GOOGLE_MODEL_NAME:
            providers.append("google")
            
        return providers
    
    @classmethod
    def get_default_provider(cls):
        """Returns the default provider to use based on available configurations."""
        providers = cls.get_available_providers()
        if providers:
            return providers[0]
        return None 