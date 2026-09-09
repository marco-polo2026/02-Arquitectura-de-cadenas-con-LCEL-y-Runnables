import asyncio
import os

from dotenv import load_dotenv

from llm_client.exceptions import LLMError
from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, ModelConfig

load_dotenv()

QUESTION = "¿Qué es la entropía? Respondé en una oración."

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5"}


async def main() -> None:
    manager = AsyncLLMManager()
    config = ModelConfig(
        model=os.getenv("LLM_MODEL") or DEFAULT_MODELS[manager.provider]
    )
    messages = [ChatMessage(role="user", content=QUESTION)]

    print(f"provider: {manager.provider} | model: {config.model}")

    print("\n=== generate ===")
    response = await manager.generate(messages, config)
    print(response.error or response.content)

    print("\n=== stream ===")
    try:
        async for token in manager.stream(messages, config):
            print(token, end="", flush=True)
        print()
    except LLMError as error:
        print(error)


if __name__ == "__main__":
    asyncio.run(main())
