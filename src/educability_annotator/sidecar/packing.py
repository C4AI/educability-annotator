"""Lotes por orçamento de tokens, com documentos inteiros e sem padding.

Cada chunk leva os tokens especiais do pós-processador (ex.: [CLS] … [SEP]) e entra concatenado no
lote, delimitado por `cu_seqlens`, como o caminho desempacotado do FlashAttention-2 espera. Um documento
nunca fica dividido entre lotes, então a agregação dos chunks acontece dentro do lote.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .chunking import OK, Chunker, Tokenized


def _chunks(chunker: Chunker, ids: np.ndarray) -> list[np.ndarray]:
    prefix = np.asarray(chunker.prefix_ids, dtype=np.int32)
    suffix = np.asarray(chunker.suffix_ids, dtype=np.int32)
    return [np.concatenate([prefix, ids[start:start + chunker.capacity], suffix])
            for start in range(0, len(ids), chunker.capacity)]


def pack(chunker: Chunker, tokenized: Tokenized, doc_ids: np.ndarray, budget: int) -> Iterator[dict]:
    """Lotes em ordem com no máximo `budget` tokens. O maior documento precisa caber num lote."""
    limit = (chunker.max_tokens + chunker.capacity - 1) // chunker.capacity * (chunker.capacity + chunker.special_count)
    if budget < limit:
        raise ValueError(f"orçamento de {budget} tokens não comporta um documento de {limit}")
    count = len(doc_ids)
    chunk_count = chunker.chunk_count(tokenized.tokens_used)
    start = 0
    while start < count:
        pieces, chunk_doc, chunk_weight, chunk_first = [], [], [], []
        used = 0
        end = start
        while end < count:
            if tokenized.status[end] == OK:
                document = _chunks(chunker, tokenized.ids[end])
                size = sum(len(piece) for piece in document)
                if used + size > budget:
                    break
                for position, piece in enumerate(document):
                    pieces.append(piece)
                    chunk_doc.append(end - start)
                    chunk_weight.append(len(piece) - chunker.special_count)
                    chunk_first.append(position == 0)
                used += size
            end += 1
        lengths = np.fromiter((len(piece) for piece in pieces), dtype=np.int32, count=len(pieces))
        yield {
            "doc_id": doc_ids[start:end],
            "status": tokenized.status[start:end],
            "tokens_used": tokenized.tokens_used[start:end],
            "truncated": tokenized.truncated[start:end],
            "chunk_count": np.where(tokenized.status[start:end] == OK, chunk_count[start:end], 0).astype(np.int16),
            "input_ids": np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int32),
            "cu_seqlens": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32),
            "chunk_doc": np.asarray(chunk_doc, dtype=np.int64),
            "chunk_weight": np.asarray(chunk_weight, dtype=np.int64),
            "chunk_first": np.asarray(chunk_first, dtype=bool),
            "max_seqlen": int(lengths.max()) if len(lengths) else 0,
        }
        start = end
