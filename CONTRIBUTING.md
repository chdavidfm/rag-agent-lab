# Cómo contribuir

Gracias por el interés. Este proyecto prioriza el código legible y la
calidad demostrable: todo lo que entra debe poder ejecutarse y medirse.

## Preparar el entorno

```bash
git clone https://github.com/chdavidfm/rag-agent-lab.git
cd rag-agent-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Antes de abrir un pull request

Estas cuatro comprobaciones son las mismas que ejecuta la integración
continua. Si pasan en local, pasarán en el servidor:

```bash
ruff format .   # aplica el formato
ruff check .    # lint
mypy            # tipos
pytest          # tests
```

Los tests marcados como `integracion` descargan modelos reales y quedan
fuera de la ejecución normal:

```bash
pytest -m integracion
```

## Calidad de la recuperación

Cualquier cambio que afecte a cómo se buscan los fragmentos debe medirse.
La integración continua rechaza los cambios que empeoren las métricas:

```bash
rag-eval --min-hit-rate 0.85 --min-mrr 0.75
```

Si tu cambio **mejora** los números, actualiza los umbrales en
`.github/workflows/ci.yml` para consolidar la mejora.

## Criterios para el código

- Cada módulo tiene una responsabilidad y la explica en su docstring.
- Las dependencias pesadas se importan de forma perezosa y viven en un
  extra opcional, nunca en el núcleo.
- Toda función pública lleva anotaciones de tipo.
- Los comentarios explican **por qué**, no **qué**: el código ya dice qué.

## Mensajes de commit

Una primera línea en imperativo que quepa en 72 caracteres, y un cuerpo
que explique el motivo del cambio cuando no sea evidente.

```
Añade fusión híbrida de recuperadores

Las estrategias léxica y densa fallan en casos distintos. RRF las combina
por posición, sin necesidad de calibrar escalas de puntuación.
```
