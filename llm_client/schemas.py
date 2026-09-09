from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelConfig(BaseModel):
    model: str
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    extra: dict[str, Any] = {}


class ModelResponse(BaseModel):
    content: str
    model: str
    error: str | None = None
