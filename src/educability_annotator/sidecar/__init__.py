"""Rótulos laterais: anota corpora Parquet grandes e grava só os rótulos, chaveados por ID.

Modo opt-in, ao lado do pipeline original (prepare/run/verify), para corpora em que reescrever os
registros inteiros não faz sentido. Mantém a semântica do Classifier original: chunks de tokens sem
sobreposição, softmax com a temperatura do modelo uma vez por chunk, média ponderada por tokens e
desempate pela menor classe. O que muda é a execução:

- tokeniza só o prefixo que o modelo vai ver (chunking.py), em vários processos por GPU;
- empacota documentos inteiros em lotes por orçamento de tokens, sem padding (packing.py);
- infere pelo caminho FlashAttention-2 do ModernBERT com cu_seqlens, ou SDPA com padding como
  alternativa (encoder.py);
- grava um arquivo de rótulos por arquivo de entrada, com marcador de commit, e distribui as tarefas
  dinamicamente entre os processos (tasks.py, writer.py, runner.py).

Uso: python -m educability_annotator.sidecar {prepare,run,verify,status} --help
"""
