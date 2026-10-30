"""Paridade entre duas rodadas de rótulos sobre os mesmos documentos (PLANO_EDUCABILIDADE.md, 2.5).

Exemplo: piloto em CPU FP32 (referência) × piloto em GPU com FlashAttention-2 e FP16 (candidato). Junta
por doc_uid os arquivos de mesmo caminho relativo e mede, no total e por fonte:
- documentos comparados e trocas de classe;
- |Δ score esperado| (p50, p99, máx.) e |Δp| máx.;
- divergências de status, tokens ou chunks, que têm de ser zero porque a tokenização é a mesma.

Uso: python tools/compare_labels.py --reference A --candidate B --out-json c.json --out-md c.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PREFIX = "educability_"
PROBS = [f"{PREFIX}p{i}" for i in range(1, 5)]


def metrics(frame: pd.DataFrame) -> dict:
    ok = frame[(frame[PREFIX + "status_ref"] == "ok") & (frame[PREFIX + "status_cand"] == "ok")]
    delta = (ok[PREFIX + "expected_score_cand"] - ok[PREFIX + "expected_score_ref"]).abs()
    probs = np.abs(ok[[p + "_cand" for p in PROBS]].to_numpy() - ok[[p + "_ref" for p in PROBS]].to_numpy())
    changed = ok[PREFIX + "class_cand"] != ok[PREFIX + "class_ref"]
    return {
        "documents": len(frame), "compared_ok": len(ok),
        "status_mismatch": int((frame[PREFIX + "status_ref"] != frame[PREFIX + "status_cand"]).sum()),
        "tokens_mismatch": int((ok[PREFIX + "tokens_used_ref"] != ok[PREFIX + "tokens_used_cand"]).sum()),
        "chunks_mismatch": int((ok[PREFIX + "chunk_count_ref"] != ok[PREFIX + "chunk_count_cand"]).sum()),
        "class_changes": int(changed.sum()),
        "class_change_pct": round(100 * float(changed.mean()), 4) if len(ok) else None,
        "expected_abs_p50": float(delta.quantile(0.5)) if len(ok) else None,
        "expected_abs_p99": float(delta.quantile(0.99)) if len(ok) else None,
        "expected_abs_max": float(delta.max()) if len(ok) else None,
        "prob_abs_max": float(probs.max()) if len(ok) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    reference, candidate = Path(args.reference), Path(args.candidate)
    per_source, frames = {}, []
    for directory in sorted(reference.glob("source=*")):
        source = directory.name.split("=", 1)[1]
        joined = []
        for path in sorted(directory.glob("part-*.parquet")):
            other = candidate / path.relative_to(reference)
            if not other.exists():
                raise FileNotFoundError(f"rótulos do candidato ausentes: {other}")
            joined.append(pq.read_table(path).to_pandas().merge(pq.read_table(other).to_pandas(), on="doc_uid",
                                                               suffixes=("_ref", "_cand"), validate="one_to_one"))
        frame = pd.concat(joined, ignore_index=True)
        per_source[source] = metrics(frame)
        frames.append(frame)
    total = metrics(pd.concat(frames, ignore_index=True))
    report = {"reference": str(reference), "candidate": str(candidate), "total": total, "sources": per_source}
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Paridade de rótulos", "", f"Referência: `{reference}`  ", f"Candidato: `{candidate}`", "",
             "| Fonte | Docs | Trocas de classe (%) | |Δ score| p50 | p99 | máx. | |Δp| máx. | Divergências status/tokens/chunks |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for source, row in [("**Total**", total)] + sorted(per_source.items()):
        lines.append(f"| {source} | {row['compared_ok']:,} | {row['class_changes']} ({row['class_change_pct']}) | "
                     f"{row['expected_abs_p50']:.5f} | {row['expected_abs_p99']:.5f} | {row['expected_abs_max']:.4f} | "
                     f"{row['prob_abs_max']:.4f} | {row['status_mismatch']}/{row['tokens_mismatch']}/{row['chunks_mismatch']} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(total), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
