"""Amostra B2 da Deduplicação (regra "fica a cópia mais recente"): entradas e controle para o motor.

prepare: grava os removidos (removed_sample.parquet) em partes de 1.000 em <out>/removed/source=b2/ e
         ~N mantidos sorteados de kept_labels.parquet (final_kept_doc_uid), com o texto buscado em
         release/v1/kept por doc_uid, em <out>/control/source=control/. Imprime o README.txt e os esquemas.
compare: confere os rótulos do controle contra os gravados na rodada completa (mesmo doc_uid):
         taxa de classe igual e maior |Δ expected_score|.

Uso: python tools/b2_inputs.py prepare --b2 /b2 --kept /kept --out /out [--control 500 --seed 7]
     python tools/b2_inputs.py compare --control-labels /cl --full-labels /full --out c.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PREFIX = "educability_"


def write_parts(table: pa.Table, target: Path, rows: int = 1000) -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, start in enumerate(range(0, table.num_rows, rows), start=1):
        pq.write_table(table.slice(start, rows), target / f"part-{count - 1:06d}.parquet", compression="zstd")
    return count


def by_uid(directory: Path, uids: np.ndarray, columns: list[str]) -> pa.Table:
    dataset = ds.dataset(str(directory), format="parquet", exclude_invalid_files=True)
    table = dataset.to_table(columns=columns, filter=ds.field("doc_uid").isin(pa.array(uids, pa.int64())))
    return table.sort_by("doc_uid")


def prepare(args) -> int:
    b2, out = Path(args.b2), Path(args.out)
    readme = b2 / "README.txt"
    if readme.exists():
        print(readme.read_text(encoding="utf-8"), flush=True)
    removed = pq.read_table(b2 / "removed_sample.parquet")
    print("removed:", removed.num_rows, removed.schema.names, flush=True)
    removed = removed.select(["doc_uid", "text"]).sort_by("doc_uid")
    if len(pc.unique(removed.column("doc_uid"))) != removed.num_rows:
        raise RuntimeError("doc_uid repetido na amostra de removidos")
    print("partes removed:", write_parts(removed, out / "removed" / "source=b2"), flush=True)
    kept = pq.read_table(b2 / "kept_labels.parquet")
    print("kept_labels:", kept.num_rows, kept.schema.names, flush=True)
    pool = np.unique(kept.column("final_kept_doc_uid").to_numpy())
    chosen = np.sort(np.random.default_rng(args.seed).choice(pool, size=min(args.control, len(pool)), replace=False))
    control = by_uid(Path(args.kept), chosen, ["doc_uid", "text"])
    if control.num_rows != len(chosen):
        raise RuntimeError(f"controle: {control.num_rows} textos para {len(chosen)} ids")
    print("partes control:", write_parts(control, out / "control" / "source=control"), flush=True)
    return 0


def compare(args) -> int:
    control = pa.concat_tables([pq.read_table(p) for p in sorted(Path(args.control_labels).glob("source=*/part-*.parquet"))])
    uids = control.column("doc_uid").to_numpy()
    full = by_uid(Path(args.full_labels), uids, ["doc_uid", PREFIX + "class", PREFIX + "expected_score"])
    a = control.select(["doc_uid", PREFIX + "class", PREFIX + "expected_score"]).sort_by("doc_uid").to_pandas()
    b = full.to_pandas()
    merged = a.merge(b, on="doc_uid", suffixes=("_ctl", "_full"), validate="one_to_one")
    same = merged[PREFIX + "class_ctl"] == merged[PREFIX + "class_full"]
    delta = (merged[PREFIX + "expected_score_ctl"] - merged[PREFIX + "expected_score_full"]).abs()
    report = {"control": len(a), "matched": len(merged), "class_equal_pct": round(100 * float(same.mean()), 3),
              "class_differences": int((~same).sum()), "expected_abs_max": float(delta.max()),
              "expected_abs_mean": float(delta.mean())}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if len(merged) == len(a) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--b2", required=True)
    prep.add_argument("--kept", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--control", type=int, default=500)
    prep.add_argument("--seed", type=int, default=7)
    comp = sub.add_parser("compare")
    comp.add_argument("--control-labels", required=True)
    comp.add_argument("--full-labels", required=True)
    comp.add_argument("--out", required=True)
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
