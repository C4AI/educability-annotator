"""Amostra do piloto (PLANO_EDUCABILIDADE.md, FASE 3.1): N documentos uniformes por fonte, com o texto.

Por fonte, em quatro passos:
1. lê a coluna doc_uid de todas as partes do unified;
2. tira os removidos pelo dedup (decisões por fonte);
3. sorteia N entre os mantidos, com semente por fonte;
4. lê só as partes que contêm sorteados.

O sorteio é por documento, não por row group, porque há fontes com ordenação interna (na sonda de
21/10, dois row groups do HPLT tinham 95% de documentos longos contra 18% no total). Grava doc_uid,
texto, n_chars, source_config e source_meta, que servem para comparar com os scores da própria fonte
(GigaVerbo, ClassiCC).

Grava em partes de --rows-per-part documentos (tarefas pequenas para o motor). Com --split-existing,
só reparte uma amostra já gravada num arquivo por fonte, sem sortear de novo.

Uso: python tools/pilot_sample.py --data-root /data --version v1 --per-source 20000 --out /out/sample
     python tools/pilot_sample.py --split-existing --out /out/sample
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

COLUMNS = ["doc_uid", "text", "n_chars", "source_config", "source_meta"]


def id_range(path: Path) -> tuple[int, int] | None:
    metadata = pq.ParquetFile(path).metadata
    index = next(i for i in range(metadata.num_columns) if metadata.schema.column(i).path == "doc_uid")
    bounds = [(metadata.row_group(g).column(index).statistics.min, metadata.row_group(g).column(index).statistics.max)
              for g in range(metadata.num_row_groups) if metadata.row_group(g).num_rows]
    return (min(b[0] for b in bounds), max(b[1] for b in bounds)) if bounds else None


def sample_source(directory: Path, decisions: Path, per_source: int, seed: int, pool: ThreadPoolExecutor) -> tuple[pa.Table, dict]:
    source = directory.name.split("=", 1)[1]
    parts = sorted(directory.glob("part-*.parquet"))
    uids = np.concatenate(list(pool.map(lambda path: pq.read_table(path, columns=["doc_uid"]).column(0).to_numpy(), parts)))
    removed_file = decisions / directory.name / "decisions.parquet"
    removed = pq.read_table(removed_file, columns=["doc_uid"]).column(0).to_numpy() if removed_file.exists() else np.empty(0, np.int64)
    kept = uids[~np.isin(uids, removed)] if len(removed) else uids
    rng = np.random.default_rng([seed, zlib.crc32(source.encode())])
    chosen = np.sort(rng.choice(kept, size=min(per_source, len(kept)), replace=False)) if len(kept) else kept

    def read(path: Path) -> pa.Table | None:
        bounds = id_range(path)
        if bounds is None:
            return None
        wanted = chosen[(chosen >= bounds[0]) & (chosen <= bounds[1])]
        if not len(wanted):
            return None
        table = pq.read_table(path, columns=COLUMNS, filters=[("doc_uid", "in", wanted.tolist())])
        return table

    tables = [table for table in pool.map(read, parts) if table is not None]
    table = pa.concat_tables(tables) if tables else None
    if table is not None:
        table = table.sort_by("doc_uid")
        if table.num_rows != len(chosen) or not np.array_equal(table.column("doc_uid").to_numpy(), chosen):
            raise RuntimeError(f"{source}: amostra lida ({table.num_rows}) difere da sorteada ({len(chosen)})")
    stats = {"source": source, "parts": len(parts), "docs": int(len(uids)), "removed": int(len(uids) - len(kept)),
             "kept": int(len(kept)), "sample": int(len(chosen)),
             "sample_chars": int(pc.sum(table.column("n_chars")).as_py()) if table is not None else 0}
    return table, stats


def write_parts(table: pa.Table, target: Path, rows_per_part: int) -> int:
    """Partes de até rows_per_part linhas, na ordem de doc_uid, com rename atômico de cada uma."""
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, start in enumerate(range(0, table.num_rows, rows_per_part), start=1):
        temporary = target / f"part-{count - 1:06d}.parquet.tmp"
        pq.write_table(table.slice(start, rows_per_part), temporary, compression="zstd", row_group_size=rows_per_part)
        temporary.rename(target / f"part-{count - 1:06d}.parquet")
    return count


def split_existing(out: Path, rows_per_part: int, subsample: int | None = None, seed: int = 2026) -> int:
    """Reparte a amostra gravada; com subsample, antes sorteia uniformemente até N documentos por fonte."""
    for directory in sorted(out.glob("source=*")):
        parts = sorted(directory.glob("part-*.parquet"))
        table = pa.concat_tables([pq.read_table(path) for path in parts]).sort_by("doc_uid")
        if subsample is not None and table.num_rows > subsample:
            rng = np.random.default_rng([seed, zlib.crc32(directory.name.encode()), subsample])
            table = table.take(np.sort(rng.choice(table.num_rows, size=subsample, replace=False)))
        staging = directory.with_name(directory.name + ".split")
        written = write_parts(table, staging, rows_per_part)
        for path in parts:
            path.unlink()
        for path in sorted(staging.glob("part-*.parquet")):
            path.rename(directory / path.name)
        staging.rmdir()
        print(json.dumps({"source": directory.name.split("=", 1)[1], "docs": table.num_rows, "parts": written}),
              flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--decisions", default="dedup/{version}/exact/decisions", help="relativo à raiz de dados")
    parser.add_argument("--per-source", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--rows-per-part", type=int, default=1000)
    parser.add_argument("--split-existing", action="store_true")
    parser.add_argument("--subsample", type=int, help="com --split-existing: sorteia até N documentos por fonte")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.split_existing:
        result = split_existing(Path(args.out), args.rows_per_part, args.subsample, args.seed)
        if args.subsample:
            summary_path = Path(args.out) / "sample_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["subsample"] = {"per_source": args.subsample, "seed": args.seed}
            for item in summary["sources"]:
                item["sample"] = min(item["sample"], args.subsample)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    if not args.data_root:
        parser.error("--data-root é obrigatório para sortear a amostra")

    root = Path(args.data_root)
    decisions = root / args.decisions.format(version=args.version)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for directory in sorted((root / "unified" / args.version).glob("source=*")):
            start = time.monotonic()
            table, stats = sample_source(directory, decisions, args.per_source, args.seed, pool)
            if table is not None:
                stats["sample_parts"] = write_parts(table, out / directory.name, args.rows_per_part)
            stats["seconds"] = round(time.monotonic() - start, 1)
            summary.append(stats)
            print(json.dumps(stats, ensure_ascii=False), flush=True)
    (out / "sample_summary.json").write_text(json.dumps({"seed": args.seed, "per_source": args.per_source,
                                                         "decisions": str(decisions), "sources": summary},
                                                        indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
