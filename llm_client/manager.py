import os
from collections.abc import AsyncIterator

from .anthropic_client import AnthropicClient
from .base import DEFAULT_MAX_RETRIES, BaseLLMClient
from .exceptions import LLMError
from .openai_client import OpenAIClient
from .schemas import ChatMessage, ModelConfig, ModelResponse

PROVIDERS: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


class AsyncLLMManager(BaseLLMClient):
    def __init__(self, provider: str | None = None) -> None:
        self.provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        if self.provider not in PROVIDERS:
            raise LLMError(
                f"unknown provider {self.provider!r}, expected one of {sorted(PROVIDERS)}"
            )
        self._client = PROVIDERS[self.provider](max_retries=self.max_retries)

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        return await self._client.generate(messages, config)

    def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        return self._client.stream(messages, config)
