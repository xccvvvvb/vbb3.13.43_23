# Entrenar Hyperion con Soup (CPU / poca RAM)

Dos rutas complementarias, ambas sin APIs de IA y sin pesos preentrenados
descargados por el repositorio:

| Ruta | Qué entrena | Pesos | Requiere |
|---|---|---|---|
| `neural/sft.py` | El Transformer propio de Hyperion, desde cero | Ninguno: los crea el repositorio | PyTorch (solo tensores) |
| `soup.yaml` | Un modelo pequeño preentrenado, con LoRA | `SmolLM2-360M-Instruct` (lo descarga **tu** instalación de Soup) | `soup-cli[train]` |

La primera es la que se ejecuta aquí: es la que hace que Hyperion hable como
Hyperion usando solo el hardware de este repositorio. La segunda es la receta
para cuando quieras partir de un modelo de lenguaje ya preentrenado y ajustarlo
en una laptop.

## 1. Dataset (la "personalidad" de Hyperion)

`35` ejemplos alpaca elaborados a partir de la documentación del repositorio:
arquitectura, privacidad, matemáticas seguras, presets neuronales, límites
reales y cómo usar Soup en CPU.

```bash
python soup/build_dataset.py            # -> soup/data/train.jsonl
```

Cada línea: `{"instruction": ..., "input": ..., "output": ...}` (UTF-8, JSONL).
Si quieres añadir conocimiento propio, edita `EXAMPLES` en
`soup/build_dataset.py` y vuelve a generar el archivo.

## 2. Alineación del Transformer propio (sin GPU)

El preentrenamiento (`neural.train`) aprende a continuar texto. `neural/sft.py`
añade el formato de diálogo y **solo calcula la pérdida sobre los tokens de la
respuesta** (el prompt va enmascarado con `-100`), que es lo que convierte un
autocompletador en un asistente.

```bash
pip install torch         # solo runtime de tensores

# 600 pasos en CPU (≈ 20 min en 2 núcleos, preset tiny de 9.5M parámetros)
python -m neural.sft \
  --dataset soup/data/train.jsonl \
  --preset tiny \
  --steps 600 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 3e-4 \
  --output checkpoints/sft-hyperion.pt \
  --threads 2

# Conversar con el resultado
python -m neural.chat --checkpoint checkpoints/sft-hyperion.pt
```

Si ya preentrenaste en el clúster, encadena: `--resume checkpoints/hyperion-tiny-step-5000.pt`.
El optimizador usa decaimiento separado para pesos y sesgos/normas, recorte de
gradiente a 1.0, warmup con decaimiento coseno y evaluación sobre una partición
de validación reservada.

Ajustes útiles cuando la RAM aprieta:

- `--preset tiny` (9.5M) o un `ModelConfig` propio más pequeño.
- `--max-length 256` o `512` y `--batch-size 1` con más acumulación.
- `--threads` igual al número de núcleos físicos, no lógicos.
- Los checkpoints están en `.gitignore` (`checkpoints/`, `*.pt`).

## 3. Receta Soup para un modelo preentrenado (LoRA en CPU)

`soup.yaml` ya está en la raíz y validado con `soup train --dry-run`.

```bash
python -m venv .soup-venv
. .soup-venv/bin/activate
pip install "soup-cli[train]"      # torch + transformers + peft

soup train --config soup.yaml      # validación previa: --dry-run
soup chat  --model ./soup/output
```

Decisiones tomadas para hardware modesto:

- **Base de 360M en vez de 7B.** En CPU, Soup avisa y desactiva la cuantización
  de 4 bits (no existe kernel NF4 en CPU), así que el modelo se entrena en fp32:
  el tamaño es lo único que cabe en 4 GB de RAM.
- **`batch_size: 1` + `gradient_accumulation_steps: 8`.** Ocho micro-lotes para
  obtener un lote efectivo de 8 sin picos de memoria. (La clave debe llamarse
  `gradient_accumulation_steps`; Soup rechaza `gradient_accumulation`.)
- **`gradient_checkpointing: true`** para cambiar cómputo por memoria.
- **`max_length: 512`** porque los ejemplos son cortos y el coste de atención
  crece con el cuadrado de la longitud.
- **LoRA `r=8`** para entrenar ~1 % de los parámetros.
- **`val_split: 0.1`** para ver si generaliza o solo memoriza.

Si tienes GPU, sube a `r=64`, `batch_size: 4`, `max_length: 2048` y añade
`stream_layers: true` para que el modelo base viva en RAM y se envíe capa por
capa a la VRAM.

## 4. Exportar y medir

```bash
soup merge  --adapter ./soup/output
soup export --model ./soup/output --format gguf --quant q4_k_m
soup diff   --base HuggingFaceTB/SmolLM2-360M-Instruct --tuned ./soup/output
```

GGUF + `q4_k_m` corre en llama.cpp u Ollama en CPU, que es el escenario de poca
RAM. `soup diff` compara respuestas antes y después para decidir si el ajuste
mejoró algo o si solo memorizó el dataset.

## 5. Honestidad sobre el resultado

Un modelo de 360M ajustado con 35 ejemplos aprende **estilo y formato**: a
responder en español, en tercera persona de Hyperion, con las ideas del corpus.
No adquiere conocimiento nuevo fiable ni razona mejor. Con `tiny` (9.5M, byte
tokenizer, entrenado en minutos) pasa lo mismo a menor escala: el valor real es
el pipeline auditable, reproducible y sin dependencias de red, no la puntuación
en un benchmark.
