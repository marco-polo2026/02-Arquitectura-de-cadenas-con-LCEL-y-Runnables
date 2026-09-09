import asyncio
import os
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from providers import PROVIDERS, resilient

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Respondé en una sola oración, en español."),
        ("human", "{question}"),
    ]
)

QUESTION = "¿Qué es la entropía?"

QUESTIONS = [
    "¿Qué es un agujero negro?",
    "¿Qué es el ADN?",
    "¿Qué es la fotosíntesis?",
]

provider = os.getenv("LLM_PROVIDER", "openai")

backup_names = [name for name in PROVIDERS if name != provider]

model = resilient(provider, backup_names)

chain = ANSWER_PROMPT | model | StrOutputParser()


async def main() -> None:
    print(f"provider: {provider} | backups: {backup_names}")

    print("\n=== ainvoke ===")
    print(await chain.ainvoke({"question": QUESTION}))

    print("\n=== astream ===")
    async for chunk in chain.astream({"question": QUESTION}):
        print(chunk, end="", flush=True)
    print()

    print("\n=== one by one ===")
    start = time.monotonic()
    for question in QUESTIONS:
        await chain.ainvoke({"question": question})
    print(f"{time.monotonic() - start:.1f}s")

    print("\n=== abatch ===")
    start = time.monotonic()
    answers = await chain.abatch([{"question": question} for question in QUESTIONS])
    print(f"{time.monotonic() - start:.1f}s")
    for answer in answers:
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
