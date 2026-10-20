"""Public command line interface."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import __version__
from .common import MODEL_ID, AnnotationError


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("deve ser um inteiro positivo")
    return number


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="educability-annotator",
        description="Anotação incremental e distribuída de corpora JSONL/Parquet.",
    )
    root.add_argument("--version", action="version", version=__version__)
    sub = root.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="criar manifesto imutável em uma saída vazia")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="identificador do dataset no Hugging Face Hub")
    source.add_argument("--input", nargs="+", help="arquivos, diretórios ou globs entre aspas")
    prepare.add_argument("--dataset-config")
    prepare.add_argument("--dataset-split")
    prepare.add_argument("--dataset-revision")
    prepare.add_argument("--output-format", choices=("parquet", "jsonl"))
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--text-column", default="text", help="nome literal da coluna/campo de primeiro nível")
    prepare.add_argument("--model", default=MODEL_ID)
    prepare.add_argument("--revision")
    prepare.add_argument("--max-document-tokens", type=positive, help="tokens de conteúdo; padrão: um chunk")
    prepare.add_argument("--prefix", default="educability_")
    prepare.add_argument("--jsonl-task-bytes", type=positive, default=64 * 1024 * 1024)
    prepare.add_argument("--parquet-shard-rows", type=positive, help="regravar cópia intermediária com grupos menores")
    prepare.add_argument("--strict-fingerprints", action="store_true", help="calcular SHA-256 integral das entradas")
    prepare.add_argument("--offline", action="store_true")
    run = sub.add_parser("run", help="executar localmente ou sob torchrun")
    run.add_argument("--output", required=True)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    run.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    run.add_argument("--batch-size", type=positive, default=32)
    run.add_argument("--max-batch-tokens", type=positive, default=8192)
    run.add_argument("--read-rows", type=positive, default=256)
    run.add_argument("--window-bytes", type=positive, default=8 * 1024 * 1024)
    run.add_argument("--max-record-bytes", type=positive, default=64 * 1024 * 1024)
    run.add_argument("--checkpoint-rows", type=positive, default=4096)
    run.add_argument("--prefetch", type=positive, default=2)
    run.add_argument("--torch-threads", type=positive, default=1)
    run.add_argument("--arrow-threads", type=positive, default=1)
    run.add_argument("--log-interval", type=positive, default=10)
    run.add_argument("--join-timeout", type=positive, default=300)
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--recover-lock", action="store_true",
        help="recuperar bloqueio SOMENTE após parar todos os processos da execução anterior",
    )
    run.add_argument("--full-checksums", action="store_true")
    run.add_argument("--offline", action="store_true")
    for command, description in (("status", "consultar progresso"), ("verify", "verificar partes e cobertura")):
        cmd = sub.add_parser(command, help=description)
        cmd.add_argument("--output", required=True)
        if command == "verify":
            cmd.add_argument("--full-checksums", action="store_true")
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command == "prepare":
            from .planning import prepare
            manifest = prepare(
                args.input, Path(args.output), args.text_column, args.model, args.revision,
                args.max_document_tokens, args.prefix, args.jsonl_task_bytes,
                args.parquet_shard_rows, args.strict_fingerprints, args.offline,
                dataset=args.dataset, dataset_config=args.dataset_config,
                dataset_split=args.dataset_split, dataset_revision=args.dataset_revision,
                output_format=args.output_format,
            )
            result = {"manifest_id": manifest["manifest_id"], "files": len(manifest["sources"]),
                      "tasks": len(manifest["tasks"]), "output": str(Path(args.output).resolve())}
        elif args.command == "run":
            from .runner import execute
            result = execute(args)
        else:
            from .planning import load_manifest
            from .state import summarize, verify
            output = Path(args.output).resolve()
            manifest = load_manifest(output)
            result = (
                summarize(output, manifest) if args.command == "status"
                else verify(output, manifest, args.full_checksums)
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if args.command != "verify" or result["complete"] else 3
    except KeyboardInterrupt:
        logging.error("Interrompido; após parar os demais ranks, retome com --resume --recover-lock.")
        return 130
    except (AnnotationError, OSError, ValueError) as exc:
        logging.error("%s", exc)
        return 2
