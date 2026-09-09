import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

PROVIDERS = {
    "openai": (
        ChatOpenAI,
        {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": TEMPERATURE,
            "request_timeout": TIMEOUT,
        },
    ),
    "anthropic": (
        ChatAnthropic,
        {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
            "temperature": TEMPERATURE,
            "default_request_timeout": TIMEOUT,
        },
    ),
}

WORTH_RETRYING = (ModelConnectionError, ModelTimeoutError, ValidationError)

WORTH_SWITCHING = (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)


def build(name: str, schema: type[BaseModel] | None = None) -> Runnable:
    model_class, model_kwargs = PROVIDERS[name]
    model = model_class(**model_kwargs)
    if schema is not None:
        model = model.with_structured_output(schema)
    return model.with_retry(
        retry_if_exception_type=WORTH_RETRYING, stop_after_attempt=MAX_ATTEMPTS
    )


def resilient(
    primary: str, backups: list[str], schema: type[BaseModel] | None = None
) -> Runnable:
    return build(primary, schema).with_fallbacks(
        [build(backup, schema) for backup in backups],
        exceptions_to_handle=WORTH_SWITCHING,
    )
