"""Cobertura e distribuição dos rótulos sobre o corpus (PLANO_EDUCABILIDADE.md, 5.1 e 5.2).

Para cada fonte do unified:
- **Mantidos:** doc_uid de todas as partes, menos os IDs das decisões do dedup (--exclude).
- **Rotulados:** doc_uid de todos os diretórios de rótulos dados (--labels pode ter vários: a rodada
  base e as incrementais, como a do BDTD).
- **Cobertura:** mantidos sem rótulo (faltando, têm de ser zero) e rotulados fora dos mantidos
  (sobrando, ex.: removidos pelo fuzzy depois da rodada; o join por doc_uid resolve).
- **Distribuição exata** sobre mantidos ∩ rotulados: documentos e caracteres por classe e acima de
  cada limiar. Com --catalog/--checks/--before-after, também tokens Qwen3 pela razão
  tokens/caractere da fonte (a mesma conta do pilot_report.py).

Uso: python tools/labels_report.py --unified /data/unified/v1 --labels /labels/run1 [/labels/run2] \
         --exclude '/data/dedup/v1/*/decisions/source=*/decisions.parquet' --out-json r.json --out-md r.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PREFIX = "educability_"
THRESHOLDS = ("classe>=3", "classe=4", "score>=2.5", "score>=3.0")


def read_columns(paths: list[Path], columns: list[str], pool: ThreadPoolExecutor) -> dict[str, np.ndarray]:
    tables = list(pool.map(lambda path: pq.read_table(path, columns=columns), paths))
    return {name: np.concatenate([table.column(name).to_numpy(zero_copy_only=False) for table in tables])
            if tables else np.empty(0) for name in columns}


def source_report(directory: Path, label_roots: list[Path], excluded: list[str], pool: ThreadPoolExecutor) -> dict:
    source = directory.name.split("=", 1)[1]
    unified = read_columns(sorted(directory.glob("part-*.parquet")), ["doc_uid", "n_chars"], pool)
    order = np.argsort(unified["doc_uid"], kind="stable")
    uids, chars = unified["doc_uid"][order].astype(np.int64), unified["n_chars"][order].astype(np.int64)
    removed_files = [path for path in excluded if f"/{directory.name}/" in path]
    removed = (np.unique(read_columns([Path(p) for p in removed_files], ["doc_uid"], pool)["doc_uid"].astype(np.int64))
               if removed_files else np.empty(0, np.int64))
    kept_mask = ~np.isin(uids, removed)
    label_files = [path for root in label_roots for path in sorted((root / directory.name).glob("part-*.parquet"))]
    labels = read_columns(label_files, ["doc_uid", PREFIX + "class", PREFIX + "expected_score", PREFIX + "status"], pool)
    label_uids = labels["doc_uid"].astype(np.int64)
    if len(np.unique(label_uids)) != len(label_uids):
        raise RuntimeError(f"{source}: doc_uid rotulado mais de uma vez entre os diretórios de rótulos")
    position = np.searchsorted(uids, label_uids)
    known = (position < len(uids)) & (uids[np.minimum(position, len(uids) - 1)] == label_uids) if len(uids) else np.zeros(0, bool)
    if not known.all():
        raise RuntimeError(f"{source}: {int((~known).sum())} rótulos com doc_uid fora do unified")
    labeled_mask = np.zeros(len(uids), bool)
    labeled_mask[position] = True
    in_kept = kept_mask[position]
    status_ok = labels[PREFIX + "status"] == "ok"
    use = in_kept & status_ok
    classes = np.nan_to_num(labels[PREFIX + "class"].astype(float), nan=0).astype(int)
    scores = labels[PREFIX + "expected_score"].astype(float)
    doc_chars = chars[position]
    rules = {"classe>=3": classes >= 3, "classe=4": classes == 4, "score>=2.5": scores >= 2.5, "score>=3.0": scores >= 3.0}
    kept_chars = int(chars[kept_mask].sum())
    return {
        "source": source, "docs": int(len(uids)), "kept": int(kept_mask.sum()), "kept_chars": kept_chars,
        "labeled": int(len(label_uids)), "missing": int((kept_mask & ~labeled_mask).sum()),
        "extra": int((~in_kept).sum()), "invalid": int((in_kept & ~status_ok).sum()),
        "class_docs": {str(c): int((use & (classes == c)).sum()) for c in range(1, 5)},
        "class_chars": {str(c): int(doc_chars[use & (classes == c)].sum()) for c in range(1, 5)},
        "above": {name: {"docs": int((use & rule).sum()), "chars": int(doc_chars[use & rule].sum())}
                  for name, rule in rules.items()},
    }


def add_tokens(rows: list[dict], catalog: str, checks: list[str], before_after: str) -> None:
    """Tokens Qwen3 por razão tokens/caractere da fonte antes do dedup (conferência da ingestão ou catálogo)."""
    import yaml

    counted = {source["id"]: source.get("counted") or {} for source in yaml.safe_load(open(catalog))["sources"]}
    check = {}
    for path in checks:
        check.update(json.load(open(path))["sources"])
    chars_in = {json.loads(line)["s"]: json.loads(line)["chars"] for line in open(before_after)}
    for row in rows:
        info = check.get(row["source"], {})
        tokens = info.get("kept_tokens")
        if tokens is None:
            tokens = (counted.get(row["source"], {}).get("tokens") or 0) - sum((info.get("excluded_tokens") or {}).values())
        ratio = tokens / chars_in[row["source"]] if chars_in.get(row["source"]) else None
        row["tokens_per_char"] = ratio
        if ratio:
            row["kept_tokens"] = round(row["kept_chars"] * ratio)
            for entry in row["above"].values():
                entry["tokens"] = round(entry["chars"] * ratio)


def markdown(rows: list[dict]) -> str:
    lines = ["# Rótulos de educabilidade no corpus", "",
             "| Fonte | Mantidos | Faltando | Sobrando | Classe 1/2/3/4 (% docs) | "
             + " | ".join(f"{name} (% texto)" for name in THRESHOLDS) + " |",
             "|---|---:|---:|---:|---|" + "---:|" * len(THRESHOLDS)]
    for row in sorted(rows, key=lambda r: -r["kept_chars"]):
        total = max(sum(row["class_docs"].values()), 1)
        chars = max(sum(row["class_chars"].values()), 1)
        lines.append(f"| {row['source']} | {row['kept']:,} | {row['missing']:,} | {row['extra']:,} | "
                     + "/".join(f"{100 * row['class_docs'][str(c)] / total:.1f}" for c in range(1, 5)) + " | "
                     + " | ".join(f"{100 * row['above'][name]['chars'] / chars:.1f}" for name in THRESHOLDS) + " |")
    if all("kept_tokens" in row for row in rows):
        total = sum(row["kept_tokens"] for row in rows)
        lines += ["", f"Tokens Qwen3 mantidos: {total / 1e9:,.1f} B. Acima de cada limiar: "
                  + "; ".join(f"{name} = {sum(row['above'][name]['tokens'] for row in rows) / 1e9:,.1f} B"
                              for name in THRESHOLDS) + "."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--sources", nargs="*", help="só estas fontes (testes)")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--catalog")
    parser.add_argument("--checks", nargs="*", default=[])
    parser.add_argument("--before-after")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    excluded = sorted({path for pattern in args.exclude for path in glob.glob(pattern, recursive=True)})
    rows = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for directory in sorted(Path(args.unified).glob("source=*")):
            if args.sources and directory.name.split("=", 1)[1] not in args.sources:
                continue
            rows.append(source_report(directory, [Path(root) for root in args.labels], excluded, pool))
            print(json.dumps({key: rows[-1][key] for key in ("source", "kept", "labeled", "missing", "extra")}),
                  flush=True)
    if args.catalog and args.before_after:
        add_tokens(rows, args.catalog, args.checks, args.before_after)
    summary = {key: sum(row[key] for row in rows) for key in ("docs", "kept", "labeled", "missing", "extra", "invalid")}
    Path(args.out_json).write_text(json.dumps({"summary": summary, "sources": rows}, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(markdown(rows), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0 if summary["missing"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
