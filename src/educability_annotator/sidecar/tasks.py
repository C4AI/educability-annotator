"""Plano imutável, tentativas e reivindicação dinâmica de tarefas no modo de rótulos laterais.

Uma tarefa é um arquivo Parquet de entrada e produz um arquivo de rótulos com o mesmo caminho
relativo na saída. O custo estimado vem dos metadados do Parquet (bytes descomprimidos da coluna de
texto), sem ler dados. A faixa de IDs vem das estatísticas da coluna de ID e serve para ler só a parte
relevante das listas de exclusão, que são as decisões do dedup, ordenadas por ID.

Cada execução abre uma tentativa: a lista das tarefas sem commit válido e um contador sob flock, de
onde todo processo de todo rank tira a próxima tarefa. Quem termina primeiro pega mais, então não sobra
cauda numa GPU só. O protocolo exige um filesystem local ou com flock e rename atômicos, como o NVMe
do nó.
"""

from __future__ import annotations

import fcntl
import glob
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..common import AnnotationError, atomic_json, digest, read_json

FORMAT = "educability-sidecar/1"
PLAN = "plan.json"
CONTRACT = "contract.json"
ATTEMPT = "attempt.json"
COUNTER = "claims.counter"
TOKENIZER = "tokenizer.json"


def _expand(patterns: list[str]) -> list[str]:
    files = sorted({str(Path(match).resolve()) for pattern in patterns
                    for match in glob.glob(pattern, recursive=True) if Path(match).is_file()})
    if patterns and not files:
        raise AnnotationError(f"Nenhum arquivo encontrado em {patterns}")
    return files


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _column_index(metadata, name: str) -> int | None:
    for index in range(metadata.num_columns):
        if metadata.schema.column(index).path == name:
            return index
    return None


def describe_file(root: Path, path: Path, id_column: str, text_column: str) -> dict:
    with pq.ParquetFile(path) as parquet:
        metadata = parquet.metadata
        id_index = _column_index(metadata, id_column)
        text_index = _column_index(metadata, text_column)
        if id_index is None or text_index is None:
            raise AnnotationError(f"{path}: faltam as colunas {id_column!r} e/ou {text_column!r}")
        cost, minimum, maximum = 0, None, None
        for group in range(metadata.num_row_groups):
            row_group = metadata.row_group(group)
            cost += row_group.column(text_index).total_uncompressed_size
            stats = row_group.column(id_index).statistics
            if row_group.num_rows and (stats is None or not stats.has_min_max):
                minimum = maximum = None
                break
            if row_group.num_rows:
                minimum = stats.min if minimum is None else min(minimum, stats.min)
                maximum = stats.max if maximum is None else max(maximum, stats.max)
        if metadata.num_rows and minimum is None:
            ids = parquet.read(columns=[id_column]).column(0).to_numpy()
            minimum, maximum = int(ids.min()), int(ids.max())
    return {"path": str(path.relative_to(root)), "rows": metadata.num_rows, "cost": max(1, cost),
            "id_min": None if minimum is None else int(minimum), "id_max": None if maximum is None else int(maximum),
            **{key: value for key, value in _identity(path).items() if key != "path"}}


def build_plan(input_root: Path, pattern: str, exclude: list[str], include: list[str], id_column: str,
               text_column: str, spec: dict, chunking: dict, threads: int = 16,
               sample_tasks: int | None = None, seed: int = 0) -> dict:
    root = input_root.resolve()
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise AnnotationError(f"Nenhum Parquet em {root}/{pattern}")
    if sample_tasks is not None and sample_tasks < len(paths):
        # Testes em partes reais: arquivos sorteados, proporcionais ao número de partes de cada fonte.
        chosen = np.random.default_rng(seed).choice(len(paths), size=sample_tasks, replace=False)
        paths = [paths[index] for index in sorted(chosen)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        described = list(pool.map(lambda path: describe_file(root, path, id_column, text_column), paths))
    # Maior custo primeiro: os arquivos grandes não ficam para o fim.
    order = sorted(range(len(described)), key=lambda index: (-described[index]["cost"], described[index]["path"]))
    tasks = [{"id": f"t{position:07d}", **described[index]} for position, index in enumerate(order)]
    body = {
        "format": FORMAT, "input_root": str(root), "pattern": pattern, "id_column": id_column,
        "text_column": text_column, "exclude": [_identity(Path(path)) for path in _expand(exclude)],
        "include": [_identity(Path(path)) for path in _expand(include)], "model": spec, "chunking": chunking,
        "tasks": tasks,
    }
    return body | {"plan_id": digest(body)}


def load_plan(output: Path) -> dict:
    plan = read_json(output / PLAN)
    body = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("format") != FORMAT or digest(body) != plan.get("plan_id"):
        raise AnnotationError("Plano incompatível ou alterado.")
    return plan


def check_inputs(plan: dict) -> None:
    root = Path(plan["input_root"])
    for task in plan["tasks"]:
        stat = (root / task["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (task["size"], task["mtime_ns"]):
            raise AnnotationError(f"Entrada alterada desde o plano: {task['path']}")
    for item in plan["exclude"] + plan["include"]:
        stat = Path(item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise AnnotationError(f"Lista de IDs alterada desde o plano: {item['path']}")


def label_path(output: Path, task: dict) -> Path:
    return output / task["path"]


def commit_path(output: Path, task: dict) -> Path:
    path = label_path(output, task)
    return path.with_name(path.name + ".commit.json")


def committed(output: Path, task: dict, plan_id: str, contract_id: str) -> bool:
    marker = commit_path(output, task)
    if not marker.exists():
        return False
    commit = read_json(marker)
    labels = label_path(output, task)
    if commit.get("plan_id") != plan_id or commit.get("contract_id") != contract_id:
        raise AnnotationError(f"Commit de outro plano ou contrato: {marker}")
    if not labels.is_file() or labels.stat().st_size != commit["size"]:
        raise AnnotationError(f"Rótulos ausentes ou incompletos com commit: {labels}")
    return True


def open_attempt(output: Path, plan: dict, contract: dict, settings: dict) -> dict:
    """Nova tentativa: confere o contrato, limpa temporários e lista o que falta; zera o contador."""
    contract_file = output / CONTRACT
    contract = contract | {"plan_id": plan["plan_id"]}
    contract_id = digest(contract)
    if contract_file.exists():
        if read_json(contract_file) != contract | {"contract_id": contract_id}:
            raise AnnotationError("Precisão, atenção ou versões diferentes da execução anterior; use outra saída.")
    else:
        atomic_json(contract_file, contract | {"contract_id": contract_id})
    for temporary in output.rglob("*.tmp"):
        temporary.unlink()
    pending = [task["id"] for task in plan["tasks"] if not committed(output, task, plan["plan_id"], contract_id)]
    attempt = {"attempt_id": uuid.uuid4().hex, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "plan_id": plan["plan_id"], "contract_id": contract_id, "settings": settings, "pending": pending}
    atomic_json(output / ATTEMPT, attempt)
    (output / COUNTER).write_text("0")
    return attempt


def claim(output: Path) -> int:
    """Próximo índice da lista de pendentes desta tentativa (atômico entre processos e ranks)."""
    with open(output / COUNTER, "r+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        value = int(stream.read() or 0)
        stream.seek(0)
        stream.write(str(value + 1))
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
    return value


class IdSet:
    """IDs de arquivos Parquet (decisões do dedup, amostras), lidos por faixa com poda por estatísticas."""

    def __init__(self, items: list[dict], column: str):
        self.column = column
        self.dataset = ds.dataset([item["path"] for item in items], format="parquet") if items else None

    def __bool__(self) -> bool:
        return self.dataset is not None

    def between(self, low: int | None, high: int | None) -> np.ndarray:
        if self.dataset is None:
            return np.empty(0, dtype=np.int64)
        field = ds.field(self.column)
        condition = None if low is None or high is None else (field >= low) & (field <= high)
        table = self.dataset.to_table(columns=[self.column], filter=condition)
        return np.unique(table.column(0).to_numpy())
