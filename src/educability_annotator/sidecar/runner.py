"""Laço de inferência de um rank: lotes do DataLoader → GPU → agregação → escritor em thread.

Um processo por GPU (torchrun ou avulso). Os processos do DataLoader preparam os lotes; o processo
principal só copia para a GPU, infere, agrega e entrega as tarefas prontas ao escritor. Os ranks não
conversam entre si: as tarefas saem do contador da tentativa.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..common import AnnotationError, atomic_json, read_json
from .dataset import TaskStream
from .encoder import Encoder
from .tasks import ATTEMPT, CONTRACT, load_plan
from .writer import Accumulator, write_task

log = logging.getLogger(__name__)


def versions() -> dict:
    found = {}
    for package in ("torch", "transformers", "tokenizers", "pyarrow", "flash_attn"):
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return found


def topology() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    return rank, int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", rank))


def run(output: Path, device: str, loader_workers: int, read_rows: int, batch_tokens: int,
        prefetch: int, log_interval: float, torch_threads: int = 1, ranks_per_device: int = 1) -> dict:
    plan = load_plan(output)
    attempt = read_json(output / ATTEMPT)
    contract = read_json(output / CONTRACT)
    if attempt["contract_id"] != contract["contract_id"] or attempt["plan_id"] != plan["plan_id"]:
        raise AnnotationError("Tentativa, contrato e plano não batem; rode prepare de novo.")
    running = versions()
    if any(running[key] != contract["versions"].get(key) for key in running):
        raise AnnotationError(f"Versões {running} diferem do contrato {contract['versions']}.")
    rank, world, local_rank = topology()
    # Em CPU, a inferência usa estas threads; o torchrun fixa OMP_NUM_THREADS=1 com mais de um processo.
    torch.set_num_threads(torch_threads)
    if device == "cuda":
        # Com ranks_per_device > 1, processos vizinhos dividem a mesma GPU (rank local // k).
        index = local_rank // ranks_per_device
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise AnnotationError(f"GPU CUDA local {index} indisponível.")
        torch.cuda.set_device(index)
        device = f"cuda:{index}"
    encoder = Encoder(plan["model"], device, contract["dtype"], contract["attention"], offline=True)
    tasks = {task["id"]: task for task in plan["tasks"]}
    # spawn: os workers não herdam threads nem o contexto CUDA do processo principal (fork num processo
    # com threads pode travar o filho num lock herdado).
    loader = DataLoader(
        TaskStream(str(output), read_rows, batch_tokens), batch_size=None, num_workers=loader_workers,
        pin_memory=device.startswith("cuda"), prefetch_factor=prefetch if loader_workers else None,
        multiprocessing_context="spawn" if loader_workers else None,
    )
    log.info("rank=%d/%d dispositivo=%s precisão=%s atenção=%s workers=%d pendentes=%d", rank, world, device,
             contract["dtype"], contract["attention"], loader_workers, len(attempt["pending"]))
    counters = {"tasks": 0, "documents": 0, "chunks": 0, "tokens": 0, "infer_seconds": 0.0}
    started = last_log = time.monotonic()
    accumulators: dict[str, Accumulator] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="write") as writer:
        for batch in loader:
            documents = None
            if len(batch["cu_seqlens"]) > 1:
                begin = time.monotonic()
                to = lambda name: batch[name].to(device, non_blocking=True)
                probs = encoder.chunk_probabilities(to("input_ids"), to("cu_seqlens"), int(batch["max_seqlen"]))
                result = encoder.aggregate(probs, to("chunk_doc"), to("chunk_weight"), to("chunk_first"),
                                           len(batch["doc_id"]))
                documents = {key: value.cpu().numpy() for key, value in result.items()}
                counters["infer_seconds"] += time.monotonic() - begin
                counters["chunks"] += len(batch["cu_seqlens"]) - 1
                counters["tokens"] += len(batch["input_ids"])
            accumulator = accumulators.setdefault(batch["task"], Accumulator())
            accumulator.add({key: batch[key].numpy() for key in
                             ("doc_id", "status", "tokens_used", "truncated", "chunk_count")}, documents)
            counters["documents"] += len(batch["doc_id"])
            if batch["last"]:
                task = tasks[batch["task"]]
                stats = {key: int(value) if not isinstance(value, float) else value
                         for key, value in batch["stats"].items()}
                futures.append(writer.submit(write_task, output, plan, attempt, task,
                                             accumulators.pop(batch["task"]).table(plan["id_column"]), stats))
                counters["tasks"] += 1
                for future in [future for future in futures if future.done()]:
                    future.result()
                    futures.remove(future)
            now = time.monotonic()
            if now - last_log >= log_interval:
                elapsed = now - started
                log.info("rank=%d tarefas=%d documentos=%d (%.0f/s) tokens=%d (%.0f/s) inferência=%.0f%%",
                         rank, counters["tasks"], counters["documents"], counters["documents"] / elapsed,
                         counters["tokens"], counters["tokens"] / elapsed, 100 * counters["infer_seconds"] / elapsed)
                last_log = now
        for future in futures:
            future.result()
    if accumulators:
        raise AnnotationError(f"Tarefas sem o último lote: {sorted(accumulators)}")
    elapsed = time.monotonic() - started
    summary = counters | {"rank": rank, "world_size": world, "device": device, "elapsed_seconds": round(elapsed, 3),
                          "documents_per_second": counters["documents"] / elapsed,
                          "tokens_per_second": counters["tokens"] / elapsed}
    atomic_json(output / "attempts" / attempt["attempt_id"] / f"rank-{rank}.json", summary)
    log.info("rank=%d fim: %s", rank, summary)
    return summary
