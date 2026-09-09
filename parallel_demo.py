import asyncio
import os
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from providers import PROVIDERS, resilient

REVIEW = "Llevo 3 días esperando que respondan mi reclamo y nadie me contesta."

provider = os.getenv("LLM_PROVIDER", "openai")

backup_names = [name for name in PROVIDERS if name != provider]

model = resilient(provider, backup_names)

parser = StrOutputParser()

sentiment_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Clasificá el sentimiento. Respondé con una sola palabra: "
                "positivo, negativo o neutro. Sin explicación, sin markdown, "
                "sin punto final.",
            ),
            ("human", "{review}"),
        ]
    )
    | model
    | parser
)

topic_chain = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Respondé con el tema principal en máximo 3 palabras. "
                "Sin explicación, sin markdown, sin punto final.",
            ),
            ("human", "{review}"),
        ]
    )
    | model
    | parser
)

analysis = RunnableParallel(sentiment=sentiment_chain, topic=topic_chain)

TICKET_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Escribí una sola línea de ticket de soporte, máximo 15 palabras. "
            "Sin título, sin markdown, sin prefijos. Solo la línea.",
        ),
        ("human", "Sentimiento: {sentiment}\nTema: {topic}"),
    ]
)

full_chain = analysis | TICKET_PROMPT | model | parser


async def main() -> None:
    start = time.monotonic()
    one_by_one = {
        "sentiment": await sentiment_chain.ainvoke({"review": REVIEW}),
        "topic": await topic_chain.ainvoke({"review": REVIEW}),
    }
    print(f"=== one by one === {time.monotonic() - start:.1f}s")
    print(one_by_one)

    start = time.monotonic()
    parallel = await analysis.ainvoke({"review": REVIEW})
    print(f"\n=== RunnableParallel === {time.monotonic() - start:.1f}s")
    print(parallel)

    print("\n=== full chain ===")
    print(await full_chain.ainvoke({"review": REVIEW}))


if __name__ == "__main__":
    asyncio.run(main())
