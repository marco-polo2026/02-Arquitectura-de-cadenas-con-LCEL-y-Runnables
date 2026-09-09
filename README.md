# Cliente de LLM robusto y asíncrono

Cliente unificado y asíncrono para OpenAI y Anthropic bajo una misma interfaz.
El proveedor se elige por configuración, sin tocar el código.

## Requisitos

- Python 3.12 o superior
- Una API key de OpenAI y/o de Anthropic

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Variables de entorno

Copiá `.env.example` a `.env` y completá los valores:

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | si usás OpenAI | API key de OpenAI |
| `ANTHROPIC_API_KEY` | si usás Anthropic | API key de Anthropic |
| `LLM_PROVIDER` | no | `openai` (por defecto) o `anthropic` |
| `LLM_MODEL` | no | Modelo a usar. Si no está, se usa el default del proveedor |
| `LLM_MAX_RETRIES` | no | Reintentos ante fallas transitorias (por defecto `2`) |

Modelos por defecto: `gpt-4o-mini` para OpenAI, `claude-haiku-4-5` para Anthropic.

`.env` está en `.gitignore` y no debe subirse al repositorio.

## Ejecución

```bash
python main.py
```

El script hace la misma pregunta corta dos veces: una con respuesta completa
(`generate`) y otra token por token (`stream`).

Para cambiar de proveedor, editá `LLM_PROVIDER` en el `.env` y volvé a ejecutar.
También podés hacerlo en la misma línea:

```bash
LLM_PROVIDER=anthropic python main.py
```

## Estructura

```
llm_client/
├── schemas.py           Modelos Pydantic: ChatMessage, ModelConfig, ModelResponse
├── base.py              BaseLLMClient, la interfaz común (clase abstracta)
├── exceptions.py        LLMError
├── openai_client.py     Adaptador de OpenAI (AsyncOpenAI)
├── anthropic_client.py  Adaptador de Anthropic (AsyncAnthropic)
└── manager.py           AsyncLLMManager, elige el cliente por configuración
main.py                  Script de validación
```

## Uso desde código

```python
import asyncio

from dotenv import load_dotenv

from llm_client.manager import AsyncLLMManager
from llm_client.schemas import ChatMessage, ModelConfig

load_dotenv()


async def main():
    manager = AsyncLLMManager("anthropic")
    messages = [ChatMessage(role="user", content="¿Qué es la entropía?")]
    config = ModelConfig(model="claude-haiku-4-5", temperature=0.2)

    response = await manager.generate(messages, config)
    print(response.content)

    async for token in manager.stream(messages, config):
        print(token, end="", flush=True)


asyncio.run(main())
```

## Manejo de errores

Los errores de red, API key inválida y límite de tasa nunca cortan el programa:

- `generate()` devuelve un `ModelResponse` con el motivo en el campo `error`
  y `content` vacío.
- `stream()` lanza `LLMError`, porque al iniciar la emisión de tokens ya no hay
  un valor de retorno donde ubicar el error. Se captura con `try/except`
  alrededor del `async for`.

`LLMError` es común a todos los proveedores: quien consume el cliente no
necesita conocer las excepciones propias de cada SDK.

### Reintentos

Los reintentos los aplican los SDKs oficiales, no este código. Ante un rate
limit (429), un error de servidor (5xx) o un corte de red, reintentan con
backoff exponencial acotado más jitter, y respetan el header `Retry-After`
cuando el servidor lo envía. Los errores que no se resuelven reintentando,
como una API key inválida (401) o un pedido mal formado (400), no se
reintentan.

La cantidad de reintentos se configura con `LLM_MAX_RETRIES` y llega hasta el
SDK a través del manager. Agotados los reintentos, el error se maneja como se
describe arriba.

## Limitaciones conocidas

- **`temperature` solo aplica a OpenAI.** Los modelos actuales de Anthropic no
  aceptan el parámetro y el SDK lo rechaza, por lo que `AnthropicClient` no lo
  envía. El valor se valida igual (rango 0 a 2) pero no tiene efecto con
  Anthropic.
- `ModelConfig` expone únicamente los parámetros comunes a ambos proveedores.
  Para parámetros específicos de uno solo, usá el campo `extra`:

  ```python
  ModelConfig(model="gpt-4o-mini", extra={"frequency_penalty": 0.5})
  ```

  Ese diccionario se pasa tal cual al SDK correspondiente y no se valida.
