from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .schemas import ChatMessage, ModelConfig, ModelResponse

DEFAULT_MAX_RETRIES = 2


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> ModelResponse: ...

    @abstractmethod
    def stream(
        self, messages: list[ChatMessage], config: ModelConfig
    ) -> AsyncIterator[str]: ...
