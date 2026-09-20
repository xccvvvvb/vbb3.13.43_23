# Hyperion Local Intelligence

IA neuro-simbólica local, auditable y sin APIs externas. El motor combina recuperación BM25, embeddings dispersos creados desde cero, detección de intenciones y un intérprete matemático AST seguro.

## Ejecutar

Requiere **Python 3.11+**. No necesita instalar paquetes.

```bash
python server.py --host 0.0.0.0 --port 8000
```

Abre `http://localhost:8000`.

## Pruebas

```bash
python -m unittest -v
```

## Garantías y límites

- Cero SDKs de IA, modelos remotos, telemetría o llamadas salientes.
- CSP restrictiva y servidor basado solo en la biblioteca estándar.
- Historial almacenado en `localStorage` del navegador.
- No pretende ser un LLM de miles de millones de parámetros. Es un motor local pequeño, transparente y extensible; responde únicamente desde su conocimiento incluido y sus herramientas simbólicas.

## Estructura

- `core/engine.py`: tokenización, embeddings, ranking, intenciones y matemáticas.
- `core/knowledge.py`: corpus local auditable.
- `server.py`: API y servidor web local.
- `public/`: interfaz.
- `tests/`: pruebas unitarias y de seguridad.
