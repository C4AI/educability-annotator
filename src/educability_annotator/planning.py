"""Immutable task manifests; preparation never modifies source files."""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from .common import (
    FORMAT_VERSION, AnnotationError, atomic_json, check_identity, digest,
    identity, read_json, sync_directory,
)
from .data import check_columns


def discover(patterns: list[str], output: Path) -> list[Path]:
    paths = set()
    for pattern in patterns:
        literal = Path(pattern).expanduser()
        matches = [str(literal)] if literal.exists() else glob.glob(str(literal), recursive=True)
        if not matches:
            raise AnnotationError(f"Entrada não encontrada: {pattern}")
        for match in matches:
            candidate = Path(match).resolve()
            if candidate.is_dir():
                found = (p for p in candidate.rglob("*") if p.suffix.lower() in {".jsonl", ".parquet"})
            else:
                if candidate.suffix.lower() not in {".jsonl", ".parquet"}:
                    raise AnnotationError(f"Formato não suportado: {candidate}")
                found = [candidate]
            for path in found:
                path = path.resolve()
                if path.is_relative_to(output):
                    raise AnnotationError("A entrada inclui o diretório de saída.")
                if path.is_file():
                    paths.add(path)
    if not paths:
        raise AnnotationError("Nenhum arquivo JSONL ou Parquet encontrado.")
    return sorted(paths)


def jsonl_ranges(path: Path, target_bytes: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    starts = [0]
    covered_until = 0
    with path.open("rb") as stream:
        for target in range(target_bytes, size, target_bytes):
            if target <= covered_until:
                continue
            stream.seek(target - 1)
            if stream.read(1) != b"\n":
                # Bounded reads: a very long record must not be loaded during planning.
                while True:
                    piece = stream.readline(1024 * 1024)
                    if not piece or piece.endswith(b"\n"):
                        break
            position = stream.tell()
            covered_until = position
            if starts[-1] < position < size:
                starts.append(position)
    return list(zip(starts, starts[1:] + [size], strict=True))


def reshard_parquet(source: Path, destination: Path, rows: int) -> None:
    temporary = destination.with_suffix(".parquet.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetFile(source) as parquet:
        with pq.ParquetWriter(
            temporary, parquet.schema_arrow, compression="zstd", use_compliant_nested_type=False,
        ) as writer:
            for batch in parquet.iter_batches(batch_size=rows):
                writer.write_batch(batch, row_group_size=rows)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    sync_directory(destination.parent)


def prepare(
    inputs: list[str], output: Path, text_column: str, model_id: str,
    revision: str | None = None, max_document_tokens: int | None = None,
    prefix: str = "educability_", jsonl_task_bytes: int = 64 * 1024 * 1024,
    parquet_shard_rows: int | None = None, strict: bool = False, offline: bool = False,
    dataset: str | None = None, dataset_config: str | None = None,
    dataset_split: str | None = None, dataset_revision: str | None = None,
    output_format: str | None = None,
) -> dict:
    from .model import inspect_model

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise AnnotationError("prepare requer diretório de saída vazio ou inexistente.")
    if dataset and (inputs or offline or strict or parquet_shard_rows):
        raise AnnotationError("Hub não aceita --input, --offline, --strict-fingerprints ou --parquet-shard-rows.")
    if not dataset and any((dataset_config, dataset_split, dataset_revision, output_format)):
        raise AnnotationError("Opções --dataset-* e --output-format requerem --dataset.")
    paths = [] if dataset else discover(inputs, output)
    spec = inspect_model(model_id, revision, offline)
    max_document_tokens = max_document_tokens or spec["chunk_limit"] - spec["special_tokens"]
    # Exclusive directory for concurrent preparation attempts.
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".prepare-lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise AnnotationError("Outra preparação está ativa nesta saída.") from exc
    try:
        sources, tasks, staged = [], [], []
        if dataset:
            from .hub import discover as discover_hub
            source, tasks = discover_hub(dataset, dataset_config, dataset_split, dataset_revision, prefix, output_format or "parquet")
            sources.append(source)
        for file_index, path in enumerate(paths):
            original = identity(path, strict)
            sources.append(original)
            file_format = path.suffix.lower()[1:]
            read_path = path
            if file_format == "parquet":
                with pq.ParquetFile(path) as parquet:
                    check_columns(parquet.schema_arrow.names, prefix)
                if parquet_shard_rows:
                    read_path = output / "staging" / f"{file_index:08d}.parquet"
                    reshard_parquet(path, read_path, parquet_shard_rows)
                    staged.append(identity(read_path, strict))
                with pq.ParquetFile(read_path) as parquet:
                    groups = [
                        {"row_group": i, "rows": parquet.metadata.row_group(i).num_rows,
                         "cost": max(1, parquet.metadata.row_group(i).total_byte_size)}
                        for i in range(parquet.num_row_groups)
                    ] or [{"row_group": None, "rows": 0, "cost": 1}]
            else:
                groups = [
                    {"start": start, "end": end, "cost": max(1, end - start)}
                    for start, end in jsonl_ranges(path, jsonl_task_bytes)
                ]
            for group_index, group in enumerate(groups):
                tasks.append({
                    "id": f"f{file_index:08d}-t{group_index:08d}", "file_index": file_index,
                    "path": str(read_path), "format": file_format, **group,
                })
            check_identity(original, full=strict)
        body = {
            "format_version": FORMAT_VERSION,
            "config": {"text_column": text_column, "prefix": prefix, "max_document_tokens": max_document_tokens},
            "model": spec, "sources": sources, "staged": staged, "tasks": tasks,
        }
        manifest = body | {"manifest_id": digest(body)}
        atomic_json(output / "manifest.json", manifest)
        return manifest
    finally:
        shutil.rmtree(lock)


def load_manifest(output: Path) -> dict:
    manifest = read_json(output / "manifest.json")
    body = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("format_version") != FORMAT_VERSION or digest(body) != manifest.get("manifest_id"):
        raise AnnotationError("Manifesto incompatível ou alterado.")
    return manifest


def assignments(tasks: list[dict], world_size: int) -> list[list[str]]:
    """Largest-first greedy assignment, deterministic even for tied costs."""
    loads = [0] * world_size
    ranks = [[] for _ in range(world_size)]
    for task in sorted(tasks, key=lambda item: (-item["cost"], item["id"])):
        rank = min(range(world_size), key=lambda i: (loads[i], i))
        ranks[rank].append(task["id"])
        loads[rank] += task["cost"]
    return ranks


def check_sources(manifest: dict, full: bool = False) -> None:
    for source in manifest["sources"] + manifest["staged"]:
        if source.get("kind") == "hub":
            from .hub import check_source
            check_source(source)
        else:
            check_identity(source, full=full)
