"""Layer 1: LLM client — Azure OpenAI wrapper."""

from __future__ import annotations

from openai import AzureOpenAI

from rag.core.config import Settings


def create_chat_client(settings: Settings) -> AzureOpenAI:
    """Create an Azure OpenAI client for chat completions."""
    return AzureOpenAI(
        api_key=settings.azure_openai.api_key,
        azure_endpoint=settings.azure_openai.endpoint,
        api_version=settings.azure_openai.api_version,
    )


def chat_completion(
    client: AzureOpenAI,
    messages: list[dict],
    deployment: str,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Call Azure OpenAI chat completion and return content string."""
    kwargs: dict = {
        "model": deployment,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
