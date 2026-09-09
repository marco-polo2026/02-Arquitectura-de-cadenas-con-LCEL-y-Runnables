# Arquitectura de cadenas con LCEL y Runnables

Refactorización del cliente de LLM del Módulo 1 a una arquitectura declarativa
con LCEL (LangChain Expression Language).

El flujo principal es una cadena `prompt | modelo | parser`, ejecutada de forma
asíncrona. El proveedor se elige por configuración, y si se cae, la cadena
sigue funcionando contra el otro.

## Requisitos

- Python 3.12 o superior (desarrollado sobre 3.13)
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
| `OPENAI_MODEL` | no | Modelo de OpenAI (por defecto `gpt-4o-mini`) |
| `ANTHROPIC_MODEL` | no | Modelo de Anthropic (por defecto `claude-haiku-4-5`) |
| `LLM_PROVIDER` | no | `openai` (por defecto) o `anthropic` |
| `LLM_TEMPERATURE` | no | Variabilidad de la respuesta (por defecto `0.7`) |
| `LLM_TIMEOUT` | no | Segundos de espera por llamada (por defecto `30`) |
| `LLM_MAX_ATTEMPTS` | no | Intentos totales, no reintentos (por defecto `3`) |

El prefijo indica el alcance: `OPENAI_` y `ANTHROPIC_` son de un proveedor,
`LLM_` son de la aplicación.

`.env` está en `.gitignore` y no debe subirse al repositorio.

## Ejecución

```bash
python main.py
```

Ejercita la misma cadena de cuatro formas: `ainvoke`, `astream`, un bucle
secuencial y `abatch`, midiendo el tiempo de las dos últimas para mostrar la
diferencia.

Para cambiar de proveedor, editá `LLM_PROVIDER` o pasalo en la misma línea:

```bash
LLM_PROVIDER=anthropic python main.py
```

## Estructura

```
providers.py        Registro de proveedores, política de fallos y construcción
main.py             Elige el proveedor, compone la cadena y la ejercita
recovery_demo.py    Reintentos, fallbacks y timeouts contra un proveedor roto
parallel_demo.py    RunnableParallel comparado con llamadas secuenciales
```

`providers.py` es la capa reutilizable: sabe qué proveedores hay, cómo se
construyen y qué hacer cuando fallan. No sabe qué se les va a preguntar.

`main.py` decide: cuál se usa, cuáles quedan de suplentes y cómo se compone la
cadena.

## La cadena

```python
model = resilient(provider, backup_names)

chain = ANSWER_PROMPT | model | StrOutputParser()

answer = await chain.ainvoke({"question": "¿Qué es la entropía?"})
```

Las tres piezas son Runnables, que es la interfaz común de LangChain. El
operador `|` funciona porque `Runnable` define `__or__`, y lo que devuelve es
otro Runnable: por eso se pueden seguir encadenando.

La cadena expone las tres formas de consumo sin trabajo extra:

| Método | Qué hace |
|---|---|
| `ainvoke` | una entrada, una respuesta completa |
| `astream` | la misma respuesta, de a fragmentos |
| `abatch` | varias entradas a la vez, en paralelo |

## Resiliencia

Cada proveedor se construye con sus propios reintentos, y el fallback actúa
entre ellos: si el principal no da más, la misma pregunta va al suplente.

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

Un rate limit no se reintenta contra el mismo proveedor: la cuota no se
recupera esperando unos segundos, y el otro proveedor tiene cuota propia. Un
error 5xx tampoco: si el servidor está roto, insistirle no ayuda.

Los errores causados por uno mismo —una key mal escrita, un pedido inválido—
no reintentan ni cambian de proveedor. Cambiar de proveedor ante esos errores
esconde el problema en vez de resolverlo.

El timeout se configura en el modelo, no en la cadena: LangChain no ofrece
timeout a nivel Runnable. Se hace así para que el corte produzca un
`ModelTimeoutError`, que las dos tuplas reconocen. Un `asyncio.timeout` sería
invisible para ellas.

## Manejo de errores

Agotados los reintentos y los suplentes, la excepción sale por donde se llamó a
la cadena. `main.py` la atrapa y termina con un mensaje de una línea:

```python
except ModelError as error:
    raise SystemExit(f"{type(error).__name__}: {error}")
```

`ModelError` es la familia común de `langchain_core.exceptions`. Los errores de
los dos proveedores heredan de ella además de heredar del error de su propio
SDK, así que se atrapan sin saber quién contestó. Es lo que en el Módulo 1
resolvía la excepción `LLMError` propia.

Se atrapa `ModelError` y no `Exception` a propósito: un error del código, como
una llave del prompt que no coincide con el diccionario, tiene que explotar con
su traceback completo.

`SystemExit` termina con código distinto de cero, para que quien haya invocado
el script se entere de que falló.

Para `astream` la excepción es la única salida posible: cuando el stream falla
ya se emitieron fragmentos, y no queda un valor de retorno donde ubicar el
error.

## Decisiones

**No se instala el paquete `langchain`.** Con `langchain-core`,
`langchain-openai` y `langchain-anthropic` alcanza. El paquete grande agrega
LangGraph y baja la versión de `websockets` ya instalada, a cambio de una
función que acá se resuelve con un diccionario de nombre a clase.

**El modelo se configura por proveedor, no con una variable única.** El Módulo
1 usaba `LLM_MODEL`, pero una sola variable no puede servir a dos proveedores:
al cambiar de proveedor pediría un modelo que no existe. Con el fallback
automático el problema sería peor, porque el suplente siempre es el otro.

**`PROVIDERS` guarda la clase y un diccionario de argumentos**, en lugar de la
clase y un nombre. Cada proveedor nombra distinto algunos parámetros
—`request_timeout` en OpenAI, `default_request_timeout` en Anthropic— y ese
diccionario absorbe la diferencia. Es el reemplazo del campo `extra` del
Módulo 1.

**Los suplentes se derivan, no se declaran.** Son todos los proveedores menos
el elegido. Agregar uno nuevo al diccionario lo suma como suplente de los
demás sin tocar nada más.

## Limitaciones conocidas

- **El nombre del modelo que se reporta es el configurado, no el que
  respondió.** `StrOutputParser` devuelve texto plano y descarta la metadata,
  donde vendría la versión exacta (por ejemplo `gpt-4o-mini-2024-07-18`). El
  criterio del ejercicio pide el parser, así que se acepta la pérdida.
- **El formato de salida no está garantizado.** El prompt pide un formato pero
  el modelo puede desviarse: pedirle una palabra y recibir un párrafo con
  markdown. Cuando esa salida alimenta al siguiente eslabón, el problema se
  propaga en silencio. La solución es salida estructurada, que corresponde a la
  siguiente unidad.
- **Los reintentos suman latencia.** Con un timeout agresivo, tres intentos con
  backoff exponencial tardan más que un solo intento largo. Si hay un
  presupuesto de tiempo, conviene bajar `LLM_MAX_ATTEMPTS` o sacar el timeout
  de la lista de errores que se reintentan.
