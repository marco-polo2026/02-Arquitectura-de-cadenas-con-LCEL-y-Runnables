# Arquitectura de cadenas con LCEL y Runnables

Módulo 2 del curso de AI Engineering.

Refactorización del cliente de LLM del Módulo 1 a una arquitectura declarativa
con LCEL. El flujo principal es una cadena `prompt | modelo | parser` que se
ejecuta de forma asíncrona. El proveedor se elige por configuración, y si se
cae, la cadena responde igual usando el otro.

## Requisitos

- Python 3.12 o superior (desarrollado sobre 3.13)
- Una API key de OpenAI y/o de Anthropic

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuración

```bash
cp .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | si usás OpenAI | API key de OpenAI |
| `ANTHROPIC_API_KEY` | si usás Anthropic | API key de Anthropic |
| `OPENAI_MODEL` | no | Modelo de OpenAI (por defecto `gpt-4o-mini`) |
| `ANTHROPIC_MODEL` | no | Modelo de Anthropic (por defecto `claude-haiku-4-5`) |
| `LLM_PROVIDER` | no | `openai` (por defecto) o `anthropic` |
| `LLM_TEMPERATURE` | no | Variabilidad de la respuesta (por defecto `0.7`) |
| `LLM_TIMEOUT` | no | Segundos de espera por llamada (por defecto `30`) |
| `LLM_MAX_ATTEMPTS` | no | Intentos totales, no reintentos (por defecto `3`) |

El prefijo indica el alcance: `OPENAI_` y `ANTHROPIC_` pertenecen a un
proveedor, `LLM_` a la aplicación.

`.env` está en `.gitignore` y no debe subirse al repositorio.

## Ejecución

```bash
python main.py
```

Ejercita la misma cadena de cuatro formas: `ainvoke`, `astream`, un bucle
secuencial y `abatch`, midiendo el tiempo de las dos últimas.

Para cambiar de proveedor sin editar el `.env`:

```bash
LLM_PROVIDER=anthropic python main.py
```

## Estructura

```
providers.py        Registro de proveedores, política de fallos y construcción
main.py             Elige el proveedor, compone la cadena y la ejercita
recovery_demo.py    Reintentos, fallbacks y timeouts contra un proveedor roto
parallel_demo.py    RunnableParallel comparado con llamadas secuenciales
sequential_demo.py  Pasos encadenados: lambda y RunnablePassthrough.assign
```

`providers.py` sabe qué proveedores hay, cómo se construyen y qué hacer cuando
fallan. No sabe qué se les va a preguntar. `main.py` decide cuál se usa, cuáles
quedan de suplentes y cómo se compone la cadena.

## La cadena

```python
model = resilient(provider, backup_names)

chain = ANSWER_PROMPT | model | StrOutputParser()

answer = await chain.ainvoke({"question": "¿Qué es la entropía?"})
```

Las tres piezas son Runnables. Componerlas con `|` devuelve otro Runnable, y
eso expone tres formas de consumo sin trabajo extra:

| Método | Qué hace |
|---|---|
| `ainvoke` | una entrada, una respuesta completa |
| `astream` | la misma respuesta, de a fragmentos |
| `abatch` | varias entradas a la vez, en paralelo |

## Resiliencia

Cada proveedor se construye con sus propios reintentos, y el fallback actúa
entre ellos:

```
OpenAI con sus reintentos  →  si igual falla  →  Anthropic con sus reintentos
```

Qué se hace ante cada error lo definen dos tuplas en `providers.py`:

| Error | Reintenta | Cambia de proveedor |
|---|---|---|
| Conexión cortada | sí | sí |
| Timeout | sí | sí |
| Error 5xx del servidor | no | sí |
| Rate limit | no | sí |
| API key inválida | no | no |
| Pedido mal formado | no | no |

Los errores causados por uno mismo no reintentan ni cambian de proveedor:
hacerlo escondería el problema en vez de resolverlo.

## Manejo de errores

Agotados los reintentos y los suplentes, la excepción sale por donde se llamó a
la cadena. `main.py` la atrapa y termina con un mensaje de una línea y código
de salida distinto de cero:

```python
except ModelError as error:
    raise SystemExit(f"{type(error).__name__}: {error}")
```

`ModelError` es la familia común de `langchain_core.exceptions`: los errores de
los dos proveedores heredan de ella, así que se atrapan sin saber quién
contestó. Es lo que en el Módulo 1 resolvía la excepción `LLMError` propia.

Se atrapa `ModelError` y no `Exception` a propósito: un error del código, como
una llave del prompt que no coincide con el diccionario, tiene que explotar con
su traceback completo.

## Limitaciones conocidas

- **El nombre del modelo que se reporta es el configurado, no el que
  respondió.** `StrOutputParser` devuelve texto plano y descarta la metadata,
  donde vendría la versión exacta.
- **El formato de salida no está garantizado.** El prompt pide un formato pero
  el modelo puede desviarse. Cuando esa salida alimenta al siguiente eslabón,
  el problema se propaga en silencio. La solución es salida estructurada.
- **Los reintentos suman latencia.** Con un timeout agresivo, tres intentos con
  backoff exponencial tardan más que un solo intento largo.
