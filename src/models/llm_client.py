"""LLM client implementations for different providers."""
from abc import ABC, abstractmethod
import os
import random
from typing import Dict, List, Optional, Any, Union
import openai
# Replace the direct import with the correct way to get the OpenAI client
# from openai import AzureOpenAI 
from src.utils.config import Config

class LLMClient(ABC):
    """Base abstract class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion for the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            The generated text
        """
        pass


class AzureOpenAIClient(LLMClient):
    """Client for Azure OpenAI API."""
    
    def __init__(self):
        """Initialize the Azure OpenAI client."""
        # Configure base URL and API key
        openai.api_type = "azure"
        openai.api_base = Config.AZURE_OPENAI_ENDPOINT
        openai.api_version = Config.AZURE_OPENAI_API_VERSION
        openai.api_key = Config.AZURE_OPENAI_API_KEY
        self.deployment_name = Config.AZURE_OPENAI_DEPLOYMENT_NAME
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion using Azure OpenAI."""
        try:
            response = openai.ChatCompletion.create(
                engine=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a planning assistant that solves problems step by step. Follow these guidelines:\n1. Clearly prefix each action with 'ACTION:'\n2. After significant steps, show the current state with 'STATE x.y:' notation\n3. Include exploration and backtracking when appropriate\n4. End with a 'Plan summary:' section that lists all actions in order\n5. Be precise and detailed in your reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Azure OpenAI API: {str(e)}")
            return f"Error generating response: {str(e)}"


class OpenAIClient(LLMClient):
    """Client for OpenAI API."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        self.model = Config.OPENAI_MODEL_NAME
        openai.api_key = Config.OPENAI_API_KEY
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion using OpenAI."""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a planning assistant that solves problems step by step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return f"Error generating response: {str(e)}"


class AnthropicClient(LLMClient):
    """Client for Anthropic API."""
    
    def __init__(self):
        """Initialize the Anthropic client."""
        self.model = Config.ANTHROPIC_MODEL_NAME
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion using Anthropic."""
        # Not implemented - would need Anthropic SDK
        return "Anthropic client not implemented yet."


class GoogleClient(LLMClient):
    """Client for Google Gemini API."""
    
    def __init__(self):
        """Initialize the Google Gemini client."""
        self.model = Config.GOOGLE_MODEL_NAME
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1000) -> str:
        """Generate a completion using Google Gemini."""
        # Not implemented - would need Google AI SDK
        return "Google Gemini client not implemented yet."


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Factory function to get the appropriate LLM client based on available configurations.
    
    Args:
        provider: Optional provider name to use. If None, uses the first available provider.
        
    Returns:
        An instance of LLMClient
        
    Raises:
        ValueError: If no provider is available or the specified provider is not configured
    """
    if provider is None:
        provider = Config.get_default_provider()
        
    if provider is None:
        raise ValueError("No LLM provider is configured. Please check your .env file.")
    
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