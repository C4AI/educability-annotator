"""Arquivo de rótulos por tarefa e o marcador de commit.

Mesmo protocolo do pipeline original: escreve .tmp, fsync, rename atômico, sincroniza o diretório e só
então publica o marcador .commit.json com o SHA-256. Um arquivo sem marcador não é resultado. Os
campos são os da ferramenta (prefixo `educability_`), em precisão simples, mais o maior score esperado
entre os chunks e o do 1º chunk. Documentos inválidos ficam com tudo nulo, menos o status.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..common import atomic_json, sha256, sync_directory
from .chunking import OK, STATUSES
from .tasks import commit_path, label_path

PREFIX = "educability_"


def schema(id_column: str) -> pa.Schema:
    return pa.schema([
        pa.field(id_column, pa.int64()),
        pa.field(PREFIX + "class", pa.int8()),
        pa.field(PREFIX + "label", pa.string()),
        *[pa.field(f"{PREFIX}p{i}", pa.float32()) for i in range(1, 5)],
        pa.field(PREFIX + "expected_score", pa.float32()),
        pa.field(PREFIX + "max_chunk_score", pa.float32()),
        pa.field(PREFIX + "first_chunk_score", pa.float32()),
        pa.field(PREFIX + "tokens_used", pa.int32()),
        pa.field(PREFIX + "chunk_count", pa.int16()),
        pa.field(PREFIX + "truncated", pa.bool_()),
        pa.field(PREFIX + "status", pa.string()),
    ])


class Accumulator:
    """Resultados de uma tarefa, na ordem das linhas do arquivo de entrada."""

    def __init__(self):
        self.parts: list[dict] = []

    def add(self, batch: dict, documents: dict | None) -> None:
        count = len(batch["doc_id"])
        if documents is None:
            documents = {"probs": np.zeros((count, 4)), "class": np.ones(count, dtype=np.int64),
                         **{key: np.zeros(count) for key in ("expected_score", "max_chunk_score", "first_chunk_score")}}
        self.parts.append({key: np.asarray(batch[key]) for key in
                           ("doc_id", "status", "tokens_used", "truncated", "chunk_count")} | documents)

    def table(self, id_column: str) -> pa.Table:
        target = schema(id_column)
        if not self.parts:
            return target.empty_table()
        joined = {key: np.concatenate([part[key] for part in self.parts]) for key in self.parts[0]}
        invalid = joined["status"] != OK
        classes = joined["class"].astype(np.int8)
        columns = {
            id_column: pa.array(joined["doc_id"].astype(np.int64)),
            PREFIX + "class": pa.array(classes, mask=invalid),
            PREFIX + "label": pa.array(np.char.add("EDUCABILITY_", classes.astype(str)), mask=invalid),
            **{f"{PREFIX}p{i + 1}": pa.array(joined["probs"][:, i].astype(np.float32), mask=invalid) for i in range(4)},
            **{PREFIX + key: pa.array(joined[key].astype(np.float32), mask=invalid)
               for key in ("expected_score", "max_chunk_score", "first_chunk_score")},
            PREFIX + "tokens_used": pa.array(joined["tokens_used"].astype(np.int32), mask=invalid),
            PREFIX + "chunk_count": pa.array(joined["chunk_count"].astype(np.int16), mask=invalid),
            PREFIX + "truncated": pa.array(joined["truncated"].astype(bool), mask=invalid),
            PREFIX + "status": pa.array(np.asarray(STATUSES)[joined["status"]]),
        }
        return pa.table(columns, schema=target)

    @property
    def rows(self) -> int:
        return sum(len(part["doc_id"]) for part in self.parts)


def write_task(output: Path, plan: dict, attempt: dict, task: dict, table: pa.Table, stats: dict) -> dict:
    start = time.monotonic()
    expected = stats.get("rows_in_file", table.num_rows) - stats.get("rows_excluded", 0)
    if table.num_rows != expected:
        raise RuntimeError(f"{task['path']}: {table.num_rows} rótulos para {expected} linhas mantidas")
    path = label_path(output, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    metadata = {"educability_sidecar": json.dumps({
        "plan_id": plan["plan_id"], "contract_id": attempt["contract_id"], "task_id": task["id"],
        "input": task["path"], "model": {key: plan["model"][key] for key in ("id", "revision", "tokenizer_digest")},
        "chunking": plan["chunking"],
    }, sort_keys=True)}
    with temporary.open("wb") as stream:
        pq.write_table(table.replace_schema_metadata(metadata), stream, compression="zstd")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)
    status = table.column(PREFIX + "status").to_numpy(zero_copy_only=False)
    commit = {
        "task_id": task["id"], "plan_id": plan["plan_id"], "contract_id": attempt["contract_id"],
        "attempt_id": attempt["attempt_id"], "rows": table.num_rows,
        "invalid": int((status != STATUSES[OK]).sum()),
        "chunks": int(np.nansum(table.column(PREFIX + "chunk_count").to_numpy(zero_copy_only=False).astype(float))),
        "tokens": int(np.nansum(table.column(PREFIX + "tokens_used").to_numpy(zero_copy_only=False).astype(float))),
        "size": path.stat().st_size, "sha256": sha256(path), **stats,
        "write_seconds": round(time.monotonic() - start, 3),
    }
    atomic_json(commit_path(output, task), commit)
    return commit
