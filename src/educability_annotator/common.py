"""Small shared primitives for durable state and explicit failures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

MODEL_ID = "gregassagraf/educability-norberto-mini-4class-v1"
MODEL_REVISION = "6a114f1c8c43191137efe3e75f15e0d7af1ba292"
FORMAT_VERSION = 1
FIELDS = (
    "class", "label", "p1", "p2", "p3", "p4", "expected_score",
    "tokens_used", "chunk_count", "truncated", "status",
)


class AnnotationError(Exception):
    """An actionable input, state, or execution error."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnnotationError(f"Não foi possível ler {path}: {exc}") from exc


def identity(path: Path, strict: bool = False) -> dict:
    stat = path.stat()
    result = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if strict:
        result["sha256"] = sha256(path)
    return result


def check_identity(expected: dict, full: bool = False) -> None:
    path = Path(expected["path"])
    try:
        actual = identity(path, strict=full and "sha256" in expected)
    except OSError as exc:
        raise AnnotationError(f"Entrada indisponível: {path}: {exc}") from exc
    for field, value in actual.items():
        if expected.get(field) != value:
            raise AnnotationError(f"Entrada alterada ({field}): {path}; prepare outra saída.")


def annotation_names(prefix: str) -> list[str]:
    return [prefix + field for field in FIELDS]


def invalid_result(status: str) -> dict:
    return dict.fromkeys(FIELDS) | {"status": status}


def validate_prediction(row: dict, prefix: str) -> None:
    """Check persisted annotations without requiring torch or a tokenizer."""
    try:
        result = {key: row[prefix + key] for key in FIELDS}
        if result["status"] != "ok":
            if result["status"] not in {"missing_text", "null_text", "empty_text", "non_string_text"}:
                raise ValueError("status desconhecido")
            if any(value is not None for key, value in result.items() if key != "status"):
                raise ValueError("texto inválido com classificação")
            return
        probs = [result[f"p{i}"] for i in range(1, 5)]
        if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in probs):
            raise ValueError("probabilidades inválidas")
        if abs(sum(probs) - 1) > 1e-5:
            raise ValueError("probabilidades não somam 1")
        score = max(range(4), key=probs.__getitem__) + 1
        if result["class"] != score or result["label"] != f"EDUCABILITY_{score}":
            raise ValueError("classe/rótulo incompatível com probabilidades")
        expected = sum((i + 1) * p for i, p in enumerate(probs))
        if not math.isfinite(result["expected_score"]) or abs(expected - result["expected_score"]) > 1e-5:
            raise ValueError("score esperado incompatível")
        if result["tokens_used"] < 1 or result["chunk_count"] < 1:
            raise ValueError("contagem de tokens/chunks inválida")
        if not isinstance(result["truncated"], bool):
            raise ValueError("truncamento inválido")
    except (KeyError, ValueError, TypeError) as exc:
        raise AnnotationError(f"Anotação inconsistente: {exc}") from exc
