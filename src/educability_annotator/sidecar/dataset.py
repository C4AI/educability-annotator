"""Fluxo de lotes para os processos do DataLoader: reivindica tarefa, lê, filtra, tokeniza e empacota.

Cada processo do DataLoader tem seu próprio tokenizer e suas listas de IDs. Ele tira tarefas do contador
da tentativa até a lista acabar e lê só as colunas de ID e texto. O prefixo é fatiado ainda no Arrow,
então nenhum documento gigante vira string Python inteira, salvo nas refações. Os lotes de uma tarefa
saem em ordem; o último leva `last` e as estatísticas da tarefa.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

from ..common import AnnotationError, read_json
from .chunking import Chunker
from .packing import pack
from .tasks import ATTEMPT, TOKENIZER, IdSet, claim, load_plan


def _empty_batch() -> dict:
    return {
        "doc_id": np.empty(0, dtype=np.int64), "status": np.empty(0, dtype=np.int8),
        "tokens_used": np.empty(0, dtype=np.int32), "truncated": np.empty(0, dtype=bool),
        "chunk_count": np.empty(0, dtype=np.int16), "input_ids": np.empty(0, dtype=np.int32),
        "cu_seqlens": np.zeros(1, dtype=np.int32), "chunk_doc": np.empty(0, dtype=np.int64),
        "chunk_weight": np.empty(0, dtype=np.int64), "chunk_first": np.empty(0, dtype=bool), "max_seqlen": 0,
    }


class TaskStream(IterableDataset):
    def __init__(self, output: str, read_rows: int, batch_tokens: int):
        super().__init__()
        self.output = output
        self.read_rows = read_rows
        self.batch_tokens = batch_tokens

    def __iter__(self):
        output = Path(self.output)
        plan = load_plan(output)
        attempt = read_json(output / ATTEMPT)
        if attempt["plan_id"] != plan["plan_id"]:
            raise AnnotationError("Tentativa de outro plano.")
        tasks = {task["id"]: task for task in plan["tasks"]}
        chunking = plan["chunking"]
        chunker = Chunker((output / TOKENIZER).read_text(encoding="utf-8"), chunking["capacity"],
                          chunking["max_document_tokens"], chunking["chars_per_token"])
        exclude = IdSet(plan["exclude"], plan["id_column"])
        include = IdSet(plan["include"], plan["id_column"])
        pending = attempt["pending"]
        while True:
            index = claim(output)
            if index >= len(pending):
                return
            yield from self.task_batches(plan, tasks[pending[index]], chunker, exclude, include)

    def task_batches(self, plan: dict, task: dict, chunker: Chunker, exclude: IdSet, include: IdSet):
        start = time.monotonic()
        path = Path(plan["input_root"]) / task["path"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (task["size"], task["mtime_ns"]):
            raise AnnotationError(f"Entrada alterada desde o plano: {path}")
        removed = exclude.between(task["id_min"], task["id_max"]) if exclude else None
        wanted = include.between(task["id_min"], task["id_max"]) if include else None
        stats = {"rows_in_file": 0, "rows_excluded": 0, "retried": 0}
        previous = None
        with pq.ParquetFile(path) as parquet:
            for record in parquet.iter_batches(batch_size=self.read_rows,
                                               columns=[plan["id_column"], plan["text_column"]]):
                ids = record.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
                stats["rows_in_file"] += len(ids)
                keep = np.ones(len(ids), dtype=bool)
                if removed is not None and len(removed):
                    keep &= ~np.isin(ids, removed)
                if wanted is not None:
                    keep &= np.isin(ids, wanted)
                stats["rows_excluded"] += int((~keep).sum())
                if not keep.any():
                    continue
                texts = record.column(1).filter(pa.array(keep))
                longer = pc.fill_null(pc.greater(pc.utf8_length(texts), chunker.prefix_chars), False)
                sliced = pc.utf8_slice_codeunits(texts, 0, chunker.prefix_chars).to_pylist()
                tokenized = chunker.tokenize(sliced, longer.to_numpy(zero_copy_only=False),
                                             full_text=lambda position: texts[position].as_py())
                stats["retried"] += tokenized.retried
                for batch in pack(chunker, tokenized, ids[keep], self.batch_tokens):
                    if previous is not None:
                        yield previous
                    previous = batch | {"task": task["id"], "last": False}
        last = (previous or _empty_batch() | {"task": task["id"]}) | {"last": True}
        last["stats"] = stats | {"read_tokenize_seconds": round(time.monotonic() - start, 3)}
        yield last
