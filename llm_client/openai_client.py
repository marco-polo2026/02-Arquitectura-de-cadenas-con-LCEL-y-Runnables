from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI, OpenAIError

from .base import DEFAULT_MAX_RETRIES, BaseLLMClient
from .exceptions import LLMError
from .schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self._client = AsyncOpenAI(max_retries=max_retries)

    def _payload(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> dict[str, Any]:
        return {
            "model": config.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            **config.extra,
        }

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                **self._payload(messages, config)
            )
        except OpenAIError as error:
            return ModelResponse(content="", model=config.model, error=str(error))

        return ModelResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        try:
            stream = await self._client.chat.completions.create(
                stream=True, **self._payload(messages, config)
            )
            async with stream:
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        except OpenAIError as error:
            raise LLMError(str(error)) from error
