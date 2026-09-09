import asyncio
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

load_dotenv()

WORTH_RETRYING = (ModelConnectionError, ModelTimeoutError)

WORTH_SWITCHING = (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)

prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

wrong_key = ChatOpenAI(model="gpt-4o-mini", api_key="invalid-key", max_retries=0)

unreachable = ChatOpenAI(
    model="gpt-4o-mini", base_url="http://localhost:9999/v1", max_retries=0
)

too_slow = ChatOpenAI(model="gpt-4o-mini", request_timeout=0.001, max_retries=0)

backup = ChatAnthropic(model="claude-haiku-4-5")


def retrying(model: Runnable) -> Runnable:
    return model.with_retry(
        retry_if_exception_type=WORTH_RETRYING, stop_after_attempt=3
    )


def resilient(primary: Runnable, secondary: Runnable) -> Runnable:
    return retrying(primary).with_fallbacks(
        [retrying(secondary)], exceptions_to_handle=WORTH_SWITCHING
    )


async def run(label, model) -> None:
    start = time.monotonic()
    try:
        answer = await (prompt | model | StrOutputParser()).ainvoke(
            {"question": "Decí solo tu nombre de modelo, nada más."}
        )
        print(f"{label}: ok after {time.monotonic() - start:.1f}s -> {answer.strip()}")
    except Exception as error:
        elapsed = time.monotonic() - start
        print(f"{label}: failed after {elapsed:.1f}s -> {type(error).__name__}")


async def main() -> None:
    await run("retry only, down     ", retrying(unreachable))
    await run("fallback only, down  ", unreachable.with_fallbacks(
        [backup], exceptions_to_handle=WORTH_SWITCHING
    ))

    print()

    await run("resilient, down      ", resilient(unreachable, backup))
    await run("resilient, wrong key ", resilient(wrong_key, backup))
    await run("resilient, all fine  ", resilient(
        ChatOpenAI(model="gpt-4o-mini"), backup
    ))

    print()

    await run("timeout, bare        ", too_slow)
    await run("timeout, resilient   ", resilient(too_slow, backup))


if __name__ == "__main__":
    asyncio.run(main())
