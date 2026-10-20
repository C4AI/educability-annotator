"""Durable per-part commits and read-only validation."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from .common import (
    AnnotationError, atomic_json, read_json, sha256, validate_prediction,
)
from .data import iter_output, task_end, task_start


def part_path(output: Path, task: dict, sequence: int) -> Path:
    return output / "parts" / task["id"] / f"{sequence:08d}.{task['format']}"


def metadata_path(output: Path, task: dict, sequence: int) -> Path:
    return part_path(output, task, sequence).with_suffix(".commit.json")


def empty_state(task: dict) -> dict:
    return {
        "task_id": task["id"], "cursor": task_start(task), "next_sequence": 0,
        "rows": 0, "chunks": 0, "tokens": 0, "invalid": 0, "complete": False,
        "read_seconds": 0.0, "tokenize_seconds": 0.0, "infer_seconds": 0.0,
        "write_seconds": 0.0,
    }


def advance(state: dict, commit: dict) -> dict:
    result = state.copy()
    result.update(
        cursor=commit["end"], next_sequence=commit["sequence"] + 1, complete=commit["complete"],
    )
    if "stream_state" in commit:
        result["stream_state"] = commit["stream_state"]
    for field in (
        "rows", "chunks", "tokens", "invalid", "read_seconds", "tokenize_seconds",
        "infer_seconds", "write_seconds",
    ):
        result[field] += commit[field]
    return result


def publish_commit(output: Path, task: dict, commit: dict, state: dict) -> dict:
    path = part_path(output, task, commit["sequence"])
    commit = commit | {"size": path.stat().st_size, "sha256": sha256(path)}
    # The immutable marker is the commit point. The state file is only a cached summary.
    atomic_json(metadata_path(output, task, commit["sequence"]), commit)
    updated = advance(state, commit)
    atomic_json(output / "states" / f"{task['id']}.json", updated)
    return updated


def recover_state(
    output: Path, task: dict, manifest_id: str, contract_id: str | None,
    full: bool = False, validate_rows: bool = False, prefix: str = "educability_",
) -> dict:
    state = empty_state(task)
    directory = output / "parts" / task["id"]
    markers = sorted(directory.glob("*.commit.json"))
    for sequence, marker in enumerate(markers):
        if marker != metadata_path(output, task, sequence):
            raise AnnotationError(f"Sequência de checkpoints incompleta: {task['id']}")
        commit = read_json(marker)
        expected = {
            "task_id": task["id"], "sequence": sequence, "manifest_id": manifest_id,
            "start": state["cursor"],
        }
        if contract_id is not None:
            expected["contract_id"] = contract_id
        if any(commit.get(key) != value for key, value in expected.items()):
            raise AnnotationError(f"Checkpoint incompatível: {marker}")
        if state["complete"]:
            raise AnnotationError(f"Checkpoint após conclusão da tarefa: {marker}")
        end = commit["end"]
        limit = task_end(task)
        if limit is None:
            if end < state["cursor"] or commit["rows"] != end - state["cursor"]:
                raise AnnotationError(f"Intervalo Hub inválido: {marker}")
            if end == state["cursor"] and not commit["complete"]:
                raise AnnotationError(f"Checkpoint Hub sem avanço: {marker}")
            if commit["rows"] and not isinstance(commit.get("stream_state"), dict):
                raise AnnotationError(f"Estado do iterador ausente: {marker}")
        else:
            if not state["cursor"] <= end <= limit:
                raise AnnotationError(f"Intervalo inválido: {marker}")
            if end == state["cursor"] and limit != task_start(task):
                raise AnnotationError(f"Checkpoint sem avanço: {marker}")
            if commit["complete"] != (end == limit):
                raise AnnotationError(f"Conclusão incompatível com cobertura: {marker}")
            if task["format"] == "parquet" and commit["rows"] != end - state["cursor"]:
                raise AnnotationError(f"Contagem de linhas incompatível: {marker}")
        path = part_path(output, task, sequence)
        if not path.is_file() or path.stat().st_size != commit["size"]:
            raise AnnotationError(f"Parte ausente/incompleta: {path}")
        if full and sha256(path) != commit["sha256"]:
            raise AnnotationError(f"Checksum divergente: {path}")
        if validate_rows:
            count = 0
            for row in iter_output(path, task["format"]):
                validate_prediction(row, prefix)
                count += 1
            if count != commit["rows"]:
                raise AnnotationError(f"Número de registros divergente: {path}")
        elif task["format"] == "parquet":
            with pq.ParquetFile(path) as parquet:
                if parquet.metadata.num_rows != commit["rows"]:
                    raise AnnotationError(f"Número de registros divergente: {path}")
        state = advance(state, commit)
    return state


def summarize(output: Path, manifest: dict) -> dict:
    totals = {key: 0 for key in ("rows", "chunks", "tokens", "invalid", "complete")}
    timings = {key: 0.0 for key in ("read_seconds", "tokenize_seconds", "infer_seconds", "write_seconds")}
    for task in manifest["tasks"]:
        path = output / "states" / f"{task['id']}.json"
        if path.exists():
            state = read_json(path)
            for key in totals:
                totals[key] += state[key]
            for key in timings:
                timings[key] += state[key]
    return {
        "manifest_id": manifest["manifest_id"], "tasks": len(manifest["tasks"]),
        "completed_tasks": totals.pop("complete"), **totals, **timings,
        "finished": (output / "COMPLETE.json").exists(),
        "locked": (output / ".run-lock").exists(),
    }


def verify(output: Path, manifest: dict, full: bool = False) -> dict:
    from .planning import check_sources
    check_sources(manifest, full=full)
    contract_file = output / "execution.json"
    contract_id = read_json(contract_file)["contract_id"] if contract_file.exists() else None
    states = [
        recover_state(
            output, task, manifest["manifest_id"], contract_id, full=full,
            validate_rows=True, prefix=manifest["config"]["prefix"],
        )
        for task in manifest["tasks"]
    ]
    completed = sum(state["complete"] for state in states)
    return {
        "valid": True, "complete": completed == len(states), "tasks": len(states),
        "completed_tasks": completed, "rows": sum(state["rows"] for state in states),
        "full_checksums": full,
        "source_identity": "pinned_hub_revision" if any(s.get("kind") == "hub" for s in manifest["sources"]) else "local_files",
        "remote_content_checksums": False,
    }
