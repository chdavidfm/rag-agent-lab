# Registro de cambios

Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/);
versionado según [SemVer](https://semver.org/lang/es/).

## [0.4.0]

### Añadido
- Búsqueda híbrida mediante Reciprocal Rank Fusion, que combina la señal
  léxica y la semántica sin necesidad de calibrar escalas.
- Análisis de seguridad con CodeQL en cada cambio y semanalmente.
- Actualización automática de dependencias con Dependabot.
- Evaluación programada que abre una incidencia si la calidad decae.
- Guías de contribución y política de seguridad.

## [0.3.0]

### Añadido
- Recuperación densa con embeddings neuronales y carga perezosa del modelo.
- Contrato `Retriever` para intercambiar estrategias sin tocar el pipeline.
- Evaluación con métricas Hit@k, MRR y Recall@k, y comando `rag-eval` con
  umbrales que fallan la integración continua ante una regresión.
- Comprobación de tipos con mypy.

### Corregido
- El archivo `.env` documentado no se cargaba: el modo LLM nunca llegaba a
  activarse.
- La lectura de documentos ilegibles propagaba una excepción sin contexto.

## [0.2.0]

### Añadido
- API REST con FastAPI y despliegue en Docker.
- Batería de tests ampliada y matriz de Python 3.10, 3.11 y 3.12.

## [0.1.0]

### Añadido
- Pipeline RAG completo, ejecutable en local y sin claves.
