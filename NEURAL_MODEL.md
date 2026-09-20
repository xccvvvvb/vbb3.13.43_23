# Hyperion Transformer — entrenamiento desde cero

Esta es la ruta neuronal real de Hyperion. La arquitectura y el pipeline son propios; PyTorch se utiliza únicamente como runtime de tensores/CUDA. No descarga pesos, no llama APIs y no contiene clientes de red.

## Arquitectura

- Decoder Transformer autoregresivo.
- RMSNorm pre-normalization.
- Rotary Position Embeddings (RoPE).
- Grouped-Query Attention (GQA).
- Flash/SDPA causal cuando el hardware lo soporta.
- SwiGLU feed-forward.
- Embeddings de entrada/salida compartidos.
- Entrenamiento DDP multi-GPU, BF16, AdamW, warmup y cosine decay.
- Checkpoints atómicos y clipping de gradiente.

| Preset | Dimensión | Capas | Heads Q/KV | Contexto | Escala aproximada |
|---|---:|---:|---:|---:|---:|
| `tiny` | 384 | 6 | 6/2 | 512 | pruebas locales |
| `1.3b` | 2.048 | 24 | 16/4 | 4.096 | ~1.3B |
| `7b` | 4.096 | 32 | 32/8 | 8.192 | ~6.5B |
| `13b` | 5.120 | 40 | 40/8 | 8.192 | ~12.6B |

## 1. Runtime local

Requiere Python 3.11+, PyTorch 2.2+ y, para variantes grandes, Linux con CUDA/NCCL.

```bash
pip install --no-index --find-links /ruta/a/wheels -r requirements-neural.txt
```

`--no-index` garantiza instalación desde paquetes previamente descargados en un medio local.

## 2. Crear corpus

```bash
python -m neural.prepare_corpus /ruta/documentos --output data/corpus.txt
```

Usa exclusivamente archivos locales. La calidad, licencia y eliminación de datos sensibles del corpus son responsabilidad del operador.

## 3. Prueba funcional pequeña

```bash
python -m neural.train \
  --preset tiny \
  --data data/corpus.txt \
  --steps 1000 \
  --batch-size 2 \
  --gradient-accumulation 8
```

## 4. Entrenamiento gigante multi-GPU

```bash
torchrun --standalone --nproc_per_node=8 -m neural.train \
  --preset 7b \
  --data data/corpus.txt \
  --steps 250000 \
  --batch-size 1 \
  --gradient-accumulation 32 \
  --learning-rate 3e-4 \
  --output checkpoints
```

Para 13B cambia `--preset 13b`. Esas configuraciones exigen un clúster de GPUs y un corpus enorme; si el modelo no se entrena, sus pesos son aleatorios y no puede conversar coherentemente. El repositorio no finge incluir un entrenamiento que no ocurrió.

## 5. Inferencia

```bash
python -m neural.chat --checkpoint checkpoints/hyperion-7b-step-250000.pt
```

## Capacidad necesaria

Un preentrenamiento competitivo de 7B suele requerir cientos de miles de millones de tokens. El coste real depende del hardware, precisión, paralelismo y longitud de contexto. Los checkpoints son deliberadamente ignorados por Git porque pueden ocupar decenas de gigabytes.
