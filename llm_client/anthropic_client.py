from collections.abc import AsyncIterator
from typing import Any

from anthropic import AnthropicError, AsyncAnthropic

from .base import DEFAULT_MAX_RETRIES, BaseLLMClient
from .exceptions import LLMError
from .schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self._client = AsyncAnthropic(max_retries=max_retries)

    def _payload(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> dict[str, Any]:
        instructions = [
            message.content for message in messages if message.role == "system"
        ]
        conversation = [
            message.model_dump() for message in messages if message.role != "system"
        ]
        payload = {
            "model": config.model,
            "messages": conversation,
            "max_tokens": config.max_tokens,
            **config.extra,
        }
        if instructions:
            payload["system"] = "\n".join(instructions)
        return payload

    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse:
        try:
            response = await self._client.messages.create(
                **self._payload(messages, config)
            )
        except AnthropicError as error:
            return ModelResponse(content="", model=config.model, error=str(error))

        return ModelResponse(
            content="".join(
                block.text for block in response.content if block.type == "text"
            ),
            model=response.model,
        )

    async def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(
                **self._payload(messages, config)
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except AnthropicError as error:
            raise LLMError(str(error)) from error
