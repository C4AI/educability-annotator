"""Execution lifecycle shared by CPU, single CUDA device, and torchrun."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import time
from pathlib import Path

import pyarrow as pa
import torch

from . import __version__
from .common import AnnotationError, atomic_json, digest, read_json
from .distributed import Job, Topology
from .model import Classifier, check_local_model, resolve_device
from .pipeline import process_task
from .planning import assignments, load_manifest, check_sources
from .state import recover_state, summarize

log = logging.getLogger(__name__)


def execute(args) -> dict:
    output = Path(args.output).resolve()
    manifest = load_manifest(output)
    is_hub = any(source.get("kind") == "hub" for source in manifest["sources"])
    if is_hub and args.offline:
        raise AnnotationError("Streaming Hub requer conectividade; remova --offline.")
    topology = Topology.from_environment()
    if is_hub and len(manifest["tasks"]) < topology.world_size:
        log.warning("Somente %d shards para %d processos; parte das GPUs ficará ociosa.", len(manifest["tasks"]), topology.world_size)
    torch.set_num_threads(args.torch_threads)
    pa.set_cpu_count(args.arrow_threads)
    pa.set_io_thread_count(args.arrow_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device, dtype = resolve_device(args.device, args.dtype, topology.local_rank)
    contract = {
        "manifest_id": manifest["manifest_id"], "dtype": dtype, "attention": args.attention,
        "annotator_version": __version__,
        "torch": torch.__version__.split("+")[0],
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "pyarrow": pa.__version__,
    }
    if is_hub:
        from .hub import dependencies
        contract["datasets"] = dependencies().__version__
    contract_id = digest(contract)
    job = Job(output, topology, args.join_timeout)
    started = time.monotonic()
    last_log = 0.0
    counters = {"rows": 0, "chunks": 0, "tokens": 0, "invalid": 0}

    def progress(force=False, **updates):
        nonlocal last_log
        for key, value in updates.items():
            counters[key] += value
        now = time.monotonic()
        if force or now - last_log >= args.log_interval:
            if job.attempt is not None:
                atomic_json(job.attempt / f"progress-{topology.rank}.json", counters | {
                    "rank": topology.rank, "elapsed_seconds": now - started,
                    "rows_per_second": counters["rows"] / (now - started),
                })
            if topology.rank == 0:
                summary = summarize(output, manifest)
                processed = sum(
                    read_json(path)["rows"] for path in job.attempt.glob("progress-*.json")
                )
                log.info(
                    "tarefas=%d/%d confirmados=%d processados_nesta_execução=%d "
                    "chunks_confirmados=%d tokens_confirmados=%d inválidos=%d",
                    summary["completed_tasks"], summary["tasks"], summary["rows"], processed,
                    summary["chunks"], summary["tokens"], summary["invalid"],
                )
            last_log = now

    try:
        job.join(
            contract | {"full_checksums": args.full_checksums},
            args.resume, args.recover_lock,
        )
        if topology.rank == 0:
            execution_path = output / "execution.json"
            if execution_path.exists():
                if not args.resume:
                    raise AnnotationError("Execução existente; use --resume.")
                if read_json(execution_path) != contract | {"contract_id": contract_id}:
                    raise AnnotationError("Precisão, versões ou configuração incompatíveis com a execução anterior.")
            else:
                atomic_json(execution_path, contract | {"contract_id": contract_id})
            check_local_model(manifest["model"])
            check_sources(manifest, full=args.full_checksums)
            pending = []
            for task in manifest["tasks"]:
                state = recover_state(
                    output, task, manifest["manifest_id"], contract_id, args.full_checksums,
                )
                atomic_json(output / "states" / f"{task['id']}.json", state)
                if not state["complete"]:
                    pending.append(task)
            (output / "COMPLETE.json").unlink(missing_ok=True)
            job.publish_ready({
                "assignments": assignments(pending, topology.world_size),
                "contract_id": contract_id,
                "initial_rows": summarize(output, manifest)["rows"],
            })
        ready = job.ready()
        selected = set(ready["assignments"][topology.rank])
        classifier = None
        for task in manifest["tasks"]:
            if task["id"] not in selected:
                continue
            if classifier is None:
                check_local_model(manifest["model"])
                classifier = Classifier(
                    manifest["model"], manifest["config"]["max_document_tokens"], device, dtype,
                    args.batch_size, args.max_batch_tokens, args.offline, args.attention,
                )
                log.info("rank=%d dispositivo=%s precisão=%s tarefas=%d",
                         topology.rank, device, dtype, len(selected))
            state = recover_state(output, task, manifest["manifest_id"], contract_id)
            process_task(
                output, task, manifest, contract_id, state, classifier, args, job.assert_owner, progress,
            )
        progress(force=True)
        job.finish_rank()
        if topology.rank != 0:
            return {"rank": topology.rank, "completed": True}
        job.wait_for_all(progress)
        check_sources(manifest, full=args.full_checksums)
        result = summarize(output, manifest)
        if result["completed_tasks"] != result["tasks"]:
            raise AnnotationError("Execução terminou com tarefas pendentes.")
        elapsed = time.monotonic() - started
        runtime = [read_json(job.attempt / f"progress-{rank}.json") for rank in range(topology.world_size)]
        result.update({
            "finished": True, "locked": False, "elapsed_seconds": elapsed,
            "new_rows": result["rows"] - ready["initial_rows"],
            "rows_per_second": (result["rows"] - ready["initial_rows"]) / elapsed,
            "world_size": topology.world_size, "device_type": device.split(":")[0],
            "dtype": dtype, "contract_id": contract_id,
            "tokens_per_second": sum(rank["tokens"] for rank in runtime) / elapsed,
            "rank_metrics": runtime,
        })
        atomic_json(output / "COMPLETE.json", result)
        progress(force=True)
        job.release()
        return result
    except BaseException as exc:
        job.fail(exc)
        raise
