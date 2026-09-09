import asyncio
import os
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from providers import PROVIDERS, resilient


class Sentiment(BaseModel):
    value: Literal["positivo", "negativo", "neutro"]
    confidence: float = Field(ge=0, le=1)


REVIEWS = [
    "Llevo 3 días esperando que respondan mi reclamo y nadie me contesta.",
    "Excelente atención, me resolvieron todo en minutos.",
    "El producto llegó bien pero tardó más de lo prometido.",
    "Una vergüenza, me cobraron dos veces y nadie se hace cargo.",
]

provider = os.getenv("LLM_PROVIDER", "openai")

backup_names = [name for name in PROVIDERS if name != provider]

model = resilient(provider, backup_names)

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
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

text_chain = SENTIMENT_PROMPT | model | StrOutputParser()

structured_chain = SENTIMENT_PROMPT | resilient(provider, backup_names, Sentiment)


async def main() -> None:
    answers = await text_chain.abatch([{"review": review} for review in REVIEWS])

    detected = 0
    for answer in answers:
        is_negative = answer == "negativo"
        if is_negative:
            detected += 1
        print(f"{answer!r:45} == 'negativo' -> {is_negative}")

    print(f"\nnegativas detectadas: {detected} de 2 reales")

    print("\n=== con molde ===")
    results = await structured_chain.abatch([{"review": review} for review in REVIEWS])
    for result in results:
        print(f"{result!r:60} .value == 'negativo' -> {result.value == 'negativo'}")


if __name__ == "__main__":
    asyncio.run(main())
