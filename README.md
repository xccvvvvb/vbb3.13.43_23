# Hyperion Local Intelligence

IA neuro-simbólica local, auditable y sin APIs externas. Hyperion XL combina recuperación BM25, embeddings dispersos creados desde cero, detección de intenciones, un intérprete matemático AST seguro y memoria expandible en RAM.

Incluye 36 núcleos multidisciplinares y puede ingerir hasta 500.000 caracteres por documento desde la interfaz. El contenido se divide en segmentos solapados, se vectoriza y queda disponible inmediatamente para consultas.

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

## Modelo neuronal gigante

El repositorio incluye ahora un Transformer autoregresivo propio entrenable desde cero con presets de **1.3B, 7B y 13B parámetros**. Incluye RoPE, RMSNorm, GQA, SwiGLU, DDP multi-GPU, BF16, AdamW, checkpoints e inferencia local. Consulta [`NEURAL_MODEL.md`](NEURAL_MODEL.md).

```bash
# Construir corpus local
python -m neural.prepare_corpus ./documentos --output data/corpus.txt

# Entrenar 7B en 8 GPUs locales
pip install -r requirements-neural.txt
torchrun --standalone --nproc_per_node=8 -m neural.train --preset 7b --data data/corpus.txt
```

PyTorch funciona únicamente como runtime matemático/CUDA; no es una API de IA y no aporta pesos preentrenados. El entrenamiento y los datos son enteramente locales.

## Garantías y límites

- Cero APIs de IA, modelos remotos, telemetría o llamadas salientes.
- CSP restrictiva y servidor basado solo en la biblioteca estándar.
- Historial almacenado en `localStorage` del navegador.
- No pretende ser un LLM de miles de millones de parámetros. Es un motor local pequeño, transparente y extensible; responde únicamente desde su conocimiento incluido y sus herramientas simbólicas.

## Estructura

- `core/engine.py`: tokenización, embeddings, ranking, intenciones y matemáticas.
- `core/knowledge.py` y `core/knowledge_extended.py`: corpus local multidisciplinar auditable.
- Memoria expandible: documentos del usuario indexados en RAM, eliminados al reiniciar.
- `server.py`: API y servidor web local.
- `public/`: interfaz.
- `tests/`: pruebas unitarias y de seguridad.
