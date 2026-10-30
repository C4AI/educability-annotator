"""CLI do modo de rótulos laterais: prepare, run (avulso ou sob torchrun), verify e status.

    python -m educability_annotator.sidecar prepare --input-root /data/unified/v1 \
        --pattern 'source=*/part-*.parquet' --exclude '/data/dedup/v1/exact/decisions/*/decisions.parquet' \
        --model /hf/.../snapshots/<rev> --offline --output /labels/educability-mini-6a114f1
    torchrun --standalone --nproc-per-node=8 -m educability_annotator.sidecar run --output ... --loader-workers 12
    python -m educability_annotator.sidecar verify --output ... [--full]

Códigos de saída do verify: 0 completo e íntegro, 3 parcial e íntegro, 2 com problemas.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..common import AnnotationError, atomic_json, digest, read_json, sha256
from .tasks import (
    CONTRACT, FORMAT, PLAN, TOKENIZER, build_plan, check_inputs, commit_path, label_path, load_plan, open_attempt,
)

log = logging.getLogger("educability_annotator.sidecar")


def prepare(args) -> dict:
    from transformers import AutoTokenizer

    from ..model import inspect_model
    from .runner import versions

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".prepare-lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise AnnotationError("Outra preparação está ativa nesta saída.") from exc
    try:
        request = {"input_root": str(Path(args.input_root).resolve()), "pattern": args.pattern,
                   "exclude": args.exclude, "include": args.include, "id_column": args.id_column,
                   "text_column": args.text_column, "model": args.model, "revision": args.revision,
                   "max_document_tokens": args.max_document_tokens, "chars_per_token": args.chars_per_token,
                   "sample_tasks": args.sample_tasks, "seed": args.seed}
        if (output / PLAN).exists():
            plan = load_plan(output)
            if plan["request"] != request:
                raise AnnotationError("A saída já tem um plano com outros parâmetros; use outra saída.")
        else:
            spec = inspect_model(args.model, args.revision, args.offline)
            options = {"revision": spec["revision"], "local_files_only": args.offline, "trust_remote_code": False}
            serialized = AutoTokenizer.from_pretrained(spec["id"], **options).backend_tokenizer.to_str()
            if digest(serialized) != spec["tokenizer_digest"]:
                raise AnnotationError("Tokenizer diverge do modelo inspecionado.")
            (output / TOKENIZER).write_text(serialized, encoding="utf-8")
            capacity = spec["chunk_limit"] - spec["special_tokens"]
            chunking = {"capacity": capacity, "special_tokens": spec["special_tokens"],
                        "max_document_tokens": args.max_document_tokens or 5 * capacity,
                        "chars_per_token": args.chars_per_token}
            body = build_plan(Path(args.input_root), args.pattern, args.exclude, args.include, args.id_column,
                              args.text_column, spec, chunking, threads=args.threads,
                              sample_tasks=args.sample_tasks, seed=args.seed)
            plan = body | {"request": request}
            plan = {key: value for key, value in plan.items() if key != "plan_id"}
            plan |= {"plan_id": digest(plan)}
            atomic_json(output / PLAN, plan)
        check_inputs(plan)
        contract = {"engine": FORMAT, "dtype": args.dtype, "attention": args.attention, "versions": versions()}
        attempt = open_attempt(output, plan, contract, settings={})
        return {"output": str(output), "plan_id": plan["plan_id"], "tasks": len(plan["tasks"]),
                "pending": len(attempt["pending"]), "attempt_id": attempt["attempt_id"],
                "rows_in_files": sum(task["rows"] for task in plan["tasks"])}
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def _check_task(output: Path, task: dict, plan_id: str, contract_id: str, full: bool, id_column: str) -> dict:
    marker = commit_path(output, task)
    if not marker.exists():
        return {"task": task["id"], "missing": True}
    commit = read_json(marker)
    labels = label_path(output, task)
    problems = []
    if commit.get("plan_id") != plan_id or commit.get("contract_id") != contract_id:
        problems.append("plano/contrato")
    if not labels.is_file() or labels.stat().st_size != commit["size"]:
        problems.append("arquivo ausente ou de outro tamanho")
    else:
        if full and sha256(labels) != commit["sha256"]:
            problems.append("sha256")
        rows = pq.ParquetFile(labels).metadata.num_rows
        if rows != commit["rows"] or rows != commit["rows_in_file"] - commit["rows_excluded"]:
            problems.append(f"linhas {rows} × commit {commit['rows']}")
        if full:
            ids = pq.read_table(labels, columns=[id_column]).column(0).to_numpy()
            if len(np.unique(ids)) != len(ids):
                problems.append("ID repetido")
            if len(ids) and task["id_min"] is not None and (ids.min() < task["id_min"] or ids.max() > task["id_max"]):
                problems.append("ID fora da faixa da entrada")
    return {"task": task["id"], "missing": False, "problems": problems, **{key: commit[key] for key in
            ("rows", "invalid", "chunks", "tokens", "rows_in_file", "rows_excluded", "retried")}}


def verify(args) -> dict:
    output = Path(args.output).resolve()
    plan = load_plan(output)
    contract = read_json(output / CONTRACT)
    check_inputs(plan)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        results = list(pool.map(lambda task: _check_task(output, task, plan["plan_id"], contract["contract_id"],
                                                         args.full, plan["id_column"]), plan["tasks"]))
    done = [result for result in results if not result["missing"]]
    problems = [result for result in done if result["problems"]]
    totals = {key: sum(result[key] for result in done)
              for key in ("rows", "invalid", "chunks", "tokens", "rows_in_file", "rows_excluded", "retried")}
    return {"valid": not problems, "complete": len(done) == len(results), "tasks": len(results),
            "committed": len(done), "full_checksums": args.full, **totals,
            "problems": [{"task": result["task"], "problems": result["problems"]} for result in problems[:50]]}


def status(args) -> dict:
    output = Path(args.output).resolve()
    plan = load_plan(output)
    commits = [read_json(commit_path(output, task)) for task in plan["tasks"] if commit_path(output, task).exists()]
    return {"tasks": len(plan["tasks"]), "committed": len(commits),
            "rows": sum(commit["rows"] for commit in commits), "tokens": sum(commit["tokens"] for commit in commits)}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m educability_annotator.sidecar", description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="plano imutável (se faltar) e nova tentativa")
    prep.add_argument("--input-root", required=True)
    prep.add_argument("--pattern", default="**/*.parquet", help="glob relativo à raiz de entrada")
    prep.add_argument("--exclude", nargs="*", default=[], help="Parquets com IDs a pular (globs)")
    prep.add_argument("--include", nargs="*", default=[], help="Parquets com os únicos IDs a anotar (globs)")
    prep.add_argument("--id-column", default="doc_uid")
    prep.add_argument("--text-column", default="text")
    prep.add_argument("--output", required=True)
    prep.add_argument("--model", required=True)
    prep.add_argument("--revision")
    prep.add_argument("--offline", action="store_true")
    prep.add_argument("--max-document-tokens", type=int, help="tokens de conteúdo; padrão 5 chunks")
    prep.add_argument("--chars-per-token", type=int, default=5, help="margem do prefixo tokenizado")
    prep.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    prep.add_argument("--attention", choices=("flash_attention_2", "sdpa", "eager"), default="flash_attention_2")
    prep.add_argument("--threads", type=int, default=16)
    prep.add_argument("--sample-tasks", type=int, help="testes: só N arquivos de entrada, sorteados")
    prep.add_argument("--seed", type=int, default=0, help="semente do --sample-tasks")
    run = sub.add_parser("run", help="anota as tarefas pendentes (um processo por GPU)")
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--loader-workers", type=int, default=8)
    run.add_argument("--read-rows", type=int, default=4096)
    run.add_argument("--batch-tokens", type=int, default=65536)
    run.add_argument("--prefetch", type=int, default=4)
    run.add_argument("--log-interval", type=float, default=30.0)
    run.add_argument("--torch-threads", type=int, default=1, help="threads de inferência por processo (CPU)")
    run.add_argument("--ranks-per-device", type=int, default=1, help="processos por GPU (rank local // k)")
    ver = sub.add_parser("verify", help="confere commits, tamanhos e contagens")
    ver.add_argument("--output", required=True)
    ver.add_argument("--full", action="store_true", help="também SHA-256 e IDs de cada arquivo")
    ver.add_argument("--threads", type=int, default=16)
    stat = sub.add_parser("status", help="tarefas e linhas confirmadas")
    stat.add_argument("--output", required=True)
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "prepare":
            result = prepare(args)
        elif args.command == "run":
            from .runner import run
            result = run(Path(args.output).resolve(), args.device, args.loader_workers, args.read_rows,
                         args.batch_tokens, args.prefetch, args.log_interval, args.torch_threads,
                         args.ranks_per_device)
        elif args.command == "verify":
            result = verify(args)
        else:
            result = status(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
        if args.command == "verify":
            return 0 if result["valid"] and result["complete"] else 3 if result["valid"] else 2
        return 0
    except KeyboardInterrupt:
        return 130
    except (AnnotationError, OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
