import asyncio
import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from providers import PROVIDERS, resilient

PRODUCT_NAMES = ["termo de acero 1L", "mochila antirrobo"]

PRODUCTS = [{"product": name} for name in PRODUCT_NAMES]

provider = os.getenv("LLM_PROVIDER", "openai")

backup_names = [name for name in PROVIDERS if name != provider]

model = resilient(provider, backup_names)

parser = StrOutputParser()

description_chain = (
    ChatPromptTemplate.from_template(
        "Escribí una descripción de producto de 2 oraciones para: {product}. "
        "Sin markdown, sin títulos."
    )
    | model
    | parser
)

tweet_chain = (
    ChatPromptTemplate.from_template(
        "Resumí esto en un tweet de menos de 120 caracteres, con 1 emoji. "
        "Sin comillas, sin markdown: {description}"
    )
    | model
    | parser
)

tweet_only = description_chain | (lambda text: {"description": text}) | tweet_chain

keep_both = {"description": description_chain} | RunnablePassthrough.assign(
    tweet=tweet_chain
)


async def main() -> None:
    print("=== tweet_only ===")
    tweets = await tweet_only.abatch(PRODUCTS)
    for name, tweet in zip(PRODUCT_NAMES, tweets):
        print(f"{name}: {tweet}")

    print("\n=== keep_both ===")
    for result in await keep_both.abatch(PRODUCTS):
        print(f"description: {result['description']}")
        print(f"tweet      : {result['tweet']}\n")


if __name__ == "__main__":
    asyncio.run(main())
