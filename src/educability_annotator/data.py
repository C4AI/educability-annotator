"""Incremental readers and lossless append-only annotation writers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .common import AnnotationError, FIELDS, annotation_names, sync_directory


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"chave JSON duplicada: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"constante JSON inválida: {value}")


def parse_record(raw: bytes, location: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object, parse_float=Decimal,
            parse_constant=_constant,
        )
        if not isinstance(value, dict):
            raise ValueError("cada linha precisa ser um objeto JSON")
        return value
    except (ValueError, UnicodeError) as exc:
        raise AnnotationError(f"JSONL inválido em {location}: {exc}") from exc


def check_columns(names, prefix: str) -> None:
    collisions = set(names).intersection(annotation_names(prefix))
    if collisions:
        raise AnnotationError(f"Colunas de saída já existem: {', '.join(sorted(collisions))}")
    if len(names) != len(set(names)):
        raise AnnotationError("Colunas duplicadas na entrada.")


@dataclass
class Window:
    payload: pa.Table | list[bytes]
    texts: list[dict]
    start: int
    end: int
    stream_state: dict | None = None

    @property
    def rows(self) -> int:
        return len(self.texts)


def task_start(task: dict) -> int:
    if task.get("source_kind") == "hub":
        return 0
    return task["start"] if task["format"] == "jsonl" else 0


def task_end(task: dict) -> int | None:
    if task.get("source_kind") == "hub":
        return None
    return task["end"] if task["format"] == "jsonl" else task["rows"]


def read_windows(
    task: dict, cursor: int, text_column: str, prefix: str,
    read_rows: int = 256, window_bytes: int = 8 * 1024 * 1024,
    max_record_bytes: int = 64 * 1024 * 1024,
    stream_state: dict | None = None,
) -> Iterator[Window]:
    if task.get("source_kind") == "hub":
        from .hub import read_windows as hub_windows
        yield from hub_windows(task, cursor, text_column, prefix, read_rows, window_bytes, max_record_bytes, stream_state)
        return
    path = Path(task["path"])
    if task["format"] == "jsonl":
        with path.open("rb") as stream:
            stream.seek(cursor)
            while stream.tell() < task["end"]:
                start, raw_rows, texts, byte_count = stream.tell(), [], [], 0
                while stream.tell() < task["end"] and len(texts) < read_rows:
                    offset = stream.tell()
                    raw = stream.readline(max_record_bytes + 1)
                    if len(raw) > max_record_bytes:
                        raise AnnotationError(f"Linha excede --max-record-bytes: {path}, byte {offset}")
                    if not raw:
                        raise AnnotationError(f"Fim inesperado em {path}, byte {offset}")
                    if stream.tell() > task["end"]:
                        raise AnnotationError(f"Fronteira de tarefa inválida: {task['id']}")
                    row = parse_record(raw, f"{path}, byte {offset}")
                    check_columns(list(row), prefix)
                    texts.append({text_column: row[text_column]} if text_column in row else {})
                    raw_rows.append(raw)
                    byte_count += len(raw)
                    if byte_count >= window_bytes:
                        break
                yield Window(raw_rows, texts, start, stream.tell())
        return
    with pq.ParquetFile(path) as parquet:
        check_columns(parquet.schema_arrow.names, prefix)
        offset = 0
        groups = [task["row_group"]] if task["row_group"] is not None else []
        for batch in parquet.iter_batches(batch_size=read_rows, row_groups=groups):
            end = offset + batch.num_rows
            if end <= cursor:
                offset = end
                continue
            if cursor > offset:
                batch = batch.slice(cursor - offset)
                offset = cursor
            table = pa.Table.from_batches([batch])
            if text_column in table.column_names:
                texts = [{text_column: value} for value in table[text_column].to_pylist()]
            else:
                texts = [{} for _ in range(table.num_rows)]
            yield Window(table, texts, offset, end)
            offset = end


def output_schema(source: pa.Schema, prefix: str) -> pa.Schema:
    types = {
        "class": pa.int8(), "label": pa.string(),
        **{f"p{i}": pa.float64() for i in range(1, 5)},
        "expected_score": pa.float64(), "tokens_used": pa.int64(),
        "chunk_count": pa.int64(), "truncated": pa.bool_(), "status": pa.string(),
    }
    return pa.schema(
        list(source) + [pa.field(prefix + name, types[name]) for name in FIELDS],
        metadata=source.metadata,
    )


class PartWriter:
    """Stream one part; the caller publishes its commit record after finish()."""

    def __init__(self, path: Path, task: dict, prefix: str):
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.task = task
        self.prefix = prefix
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.temporary.open("wb")
        self.parquet = None
        self.schema = None
        if task["format"] == "parquet":
            if task.get("source_kind") == "hub":
                from .hub import schema_from_source
                self.schema = output_schema(schema_from_source(task["source"]), prefix)
            else:
                with pq.ParquetFile(task["path"]) as source:
                    self.schema = output_schema(source.schema_arrow, prefix)
            self.parquet = pq.ParquetWriter(
                self.stream, self.schema, compression="zstd", use_compliant_nested_type=False,
            )

    def write(self, window: Window, results: list[dict]) -> None:
        if window.rows != len(results):
            raise AnnotationError("Número de resultados diferente da entrada.")
        if self.parquet is not None:
            table = window.payload
            for name in FIELDS:
                field = self.schema.field(self.prefix + name)
                table = table.append_column(field, pa.array([row[name] for row in results], type=field.type))
            self.parquet.write_table(table)
        else:
            for raw, result in zip(window.payload, results, strict=True):
                original = raw.rstrip()
                # Append members to the original object. This preserves arbitrary JSON
                # numbers exactly instead of round-tripping them through Python floats.
                annotations = json.dumps(
                    {self.prefix + key: result[key] for key in FIELDS},
                    ensure_ascii=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
                empty = not original.lstrip()[1:-1].strip()
                self.stream.write(original[:-1] + (b"" if empty else b",") + annotations[1:] + b"\n")

    def finish(self) -> None:
        if self.parquet is not None:
            self.parquet.close()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.temporary, self.path)
        sync_directory(self.path.parent)

    def abort(self) -> None:
        try:
            if self.parquet is not None:
                self.parquet.close()
        finally:
            self.stream.close()
            self.temporary.unlink(missing_ok=True)


def iter_output(path: Path, file_format: str):
    if file_format == "jsonl":
        with path.open("rb") as stream:
            for line, raw in enumerate(stream, 1):
                # Here float conversion is intentional for validation of generated probabilities.
                yield json.loads(raw)
    else:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=256):
                yield from batch.to_pylist()
