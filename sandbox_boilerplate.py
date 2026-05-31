"""Princeton AI Sandbox — minimal boilerplate. Needs: pip install openai"""
import os
from openai import OpenAI

API_KEY  = os.environ["SANDBOX_API_KEY"]   # export this before running
BASE_URL = "https://api.portkey.ai/v1"

MODELS = [
    "gpt-4o-mini",
    "claude-sonnet-4-6",
    "gemini-2.0-flash",
]

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def ask(model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    prompt = "Say hello and tell me which model you are."
    for model in MODELS:
        print(f"\n--- {model} ---")
        print(ask(model, prompt))
