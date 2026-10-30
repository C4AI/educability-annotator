"""Relatório do piloto por fonte (PLANO_EDUCABILIDADE.md, FASE 3.2).

Junta os rótulos do mini com a amostra da FASE 3.1 e, por fonte, mostra:
- distribuição de classes e do score esperado;
- documentos com mais de um chunk: diferença entre a média, o máximo e o 1º chunk;
- efeito do tamanho (Spearman com n_chars);
- comparação com os scores da própria fonte, sem supor o formato. Todo campo numérico do
  source_meta com "score" no nome entra num Spearman; source_config e campos categóricos curtos
  (label, register…) dão a média do nosso score por categoria.

Com os arquivos do cpt-corpus (--catalog, --checks, --before-after, --sample-summary), também extrapola
a amostra para o corpus mantido pelo dedup exato: documentos e tokens Qwen3 acima de cada limiar.
- Tokens Qwen3 de uma fonte: kept_tokens da conferência da ingestão, ou contagem do catálogo menos os
  excluídos, multiplicados pela fração de caracteres que o exato manteve (a mesma conta do
  cpt-corpus/logs/dev/dedup_tokens_est.py).
- Tokens acima do limiar: essa conta × a fração de caracteres da amostra acima dele. A amostra é
  uniforme por documento entre os mantidos, então pesar por caracteres estima a fração de texto.

Uso: python tools/pilot_report.py --sample /sample --labels /labels --out-json r.json --out-md r.md \
         [--catalog sources.yaml --checks c1.json c2.json --before-after ba.jsonl --sample-summary s.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PREFIX = "educability_"
THRESHOLDS = {
    "classe>=3": lambda frame: frame[PREFIX + "class"] >= 3,
    "classe=4": lambda frame: frame[PREFIX + "class"] == 4,
    "score>=2.5": lambda frame: frame[PREFIX + "expected_score"] >= 2.5,
    "score>=3.0": lambda frame: frame[PREFIX + "expected_score"] >= 3.0,
}


def corpus_volumes(catalog: str, checks: list[str], before_after: str, sample_summary: str) -> dict:
    """Por fonte: documentos e tokens Qwen3 mantidos pelo exato (mesma conta do dedup_tokens_est.py)."""
    import yaml

    counted = {source["id"]: source.get("counted") or {} for source in yaml.safe_load(open(catalog))["sources"]}
    check = {}
    for path in checks:
        check.update(json.load(open(path))["sources"])
    kept_docs = {item["source"]: item["kept"] for item in json.load(open(sample_summary))["sources"]}
    volumes = {}
    for line in open(before_after):
        row = json.loads(line)
        source, info = row["s"], check.get(row["s"], {})
        excluded = sum((info.get("excluded_tokens") or {}).values())
        tokens = info.get("kept_tokens")
        if tokens is None:
            tokens = (counted.get(source, {}).get("tokens") or 0) - excluded
        kept_chars = row["chars"] - row["rm_chars"]
        volumes[source] = {"docs": kept_docs.get(source, row["docs"] - row["rm_docs"]),
                           "tokens": round(tokens * kept_chars / row["chars"]) if row["chars"] else 0}
    return volumes


def above_thresholds(frame: pd.DataFrame, volume: dict | None) -> dict:
    ok = frame[PREFIX + "status"] == "ok"
    chars = frame["n_chars"].astype(float)
    result = {}
    for name, rule in THRESHOLDS.items():
        passing = ok & rule(frame).fillna(False)
        doc_share = float(passing.mean())
        char_share = float(chars[passing].sum() / chars.sum()) if chars.sum() else 0.0
        entry = {"docs_pct": round(100 * doc_share, 2), "chars_pct": round(100 * char_share, 2)}
        if volume:
            entry |= {"docs_est": round(doc_share * volume["docs"]), "tokens_est": round(char_share * volume["tokens"])}
        result[name] = entry
    return result


def spearman(x: pd.Series, y: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 30 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    return float(frame["x"].rank().corr(frame["y"].rank()))


def flatten(meta) -> dict:
    if not isinstance(meta, str) or not meta:
        return {}
    try:
        value = json.loads(meta)
    except ValueError:
        return {}
    out = {}
    stack = [("", value)]
    while stack:
        prefix, item = stack.pop()
        if isinstance(item, dict):
            stack.extend((f"{prefix}{key}.", child) for key, child in item.items())
        else:
            out[prefix.rstrip(".")] = item
    return out


def source_fields(frame: pd.DataFrame) -> dict:
    metas = pd.DataFrame([flatten(meta) for meta in frame["source_meta"]], index=frame.index)
    numeric, categorical = {}, {}
    for column in metas.columns:
        values = metas[column]
        if "score" in column.lower() or "prob" in column.lower():
            number = pd.to_numeric(values, errors="coerce")
            if number.notna().mean() > 0.5:
                numeric[column] = spearman(number, frame[PREFIX + "expected_score"])
                continue
        if values.map(lambda v: isinstance(v, str) and len(v) <= 40).mean() > 0.9 and 1 < values.nunique() <= 30:
            categorical[column] = values
    if "source_config" in frame and frame["source_config"].nunique() > 1:
        categorical["source_config"] = frame["source_config"]
    groups = {}
    for name, values in categorical.items():
        stats = frame.groupby(values)[PREFIX + "expected_score"].agg(["count", "mean"]).sort_values("mean")
        groups[name] = {str(key): {"docs": int(row["count"]), "mean_score": round(float(row["mean"]), 3)}
                        for key, row in stats.iterrows() if row["count"] >= 20}
    return {"spearman": {k: (None if v is None else round(v, 3)) for k, v in numeric.items()}, "groups": groups}


def summarize(source: str, frame: pd.DataFrame, volume: dict | None = None) -> dict:
    ok = frame[frame[PREFIX + "status"] == "ok"]
    scores = ok[PREFIX + "expected_score"]
    classes = Counter(ok[PREFIX + "class"].astype(int))
    multi = ok[ok[PREFIX + "chunk_count"] > 1]
    return {
        "source": source, "docs": len(frame), "ok": len(ok),
        "class_pct": {str(c): round(100 * classes.get(c, 0) / max(len(ok), 1), 1) for c in range(1, 5)},
        "score_mean": round(float(scores.mean()), 3),
        "score_p10_p50_p90": [round(float(q), 3) for q in scores.quantile([0.1, 0.5, 0.9])],
        "score_std": round(float(scores.std()), 3),
        "multi_chunk_pct": round(100 * len(multi) / max(len(ok), 1), 1),
        "truncated_pct": round(100 * float(ok[PREFIX + "truncated"].mean()), 1),
        "mean_tokens": round(float(ok[PREFIX + "tokens_used"].mean()), 1),
        "multi_chunk": None if len(multi) < 30 else {
            "max_minus_mean": round(float((multi[PREFIX + "max_chunk_score"] - multi[PREFIX + "expected_score"]).mean()), 3),
            "first_minus_mean": round(float((multi[PREFIX + "first_chunk_score"] - multi[PREFIX + "expected_score"]).mean()), 3),
            "first_vs_mean_spearman": spearman(multi[PREFIX + "first_chunk_score"], multi[PREFIX + "expected_score"]),
        },
        "length_spearman": spearman(ok["n_chars"].astype(float), scores),
        "source_scores": source_fields(ok),
        "corpus": volume,
        "above": above_thresholds(frame, volume),
    }


def markdown(rows: list[dict]) -> str:
    lines = ["# Piloto do educability-mini por fonte", "",
             "| Fonte | Docs | Classe 1/2/3/4 (%) | Score médio | p10/p50/p90 | >1 chunk | Spearman tamanho | Scores da fonte |",
             "|---|---:|---|---:|---|---:|---:|---|"]
    for row in sorted(rows, key=lambda r: r["score_mean"]):
        own = "; ".join(f"{k}: {v}" for k, v in row["source_scores"]["spearman"].items() if v is not None)
        lines.append(f"| {row['source']} | {row['ok']} | {'/'.join(str(v) for v in row['class_pct'].values())} | "
                     f"{row['score_mean']} | {'/'.join(map(str, row['score_p10_p50_p90']))} | {row['multi_chunk_pct']}% | "
                     f"{row['length_spearman']} | {own} |")
    if all(row["corpus"] for row in rows):
        lines += ["", "## Volume estimado acima de cada limiar (corpus mantido pelo exato, tokens Qwen3)", "",
                  "| Fonte | Docs mantidos | Tokens mantidos | " + " | ".join(f"Tokens {name} (%)" for name in THRESHOLDS) + " |",
                  "|---|---:|---:|" + "---:|" * len(THRESHOLDS)]
        totals = {name: 0 for name in THRESHOLDS}
        for row in sorted(rows, key=lambda r: -r["corpus"]["tokens"]):
            cells = []
            for name in THRESHOLDS:
                totals[name] += row["above"][name]["tokens_est"]
                cells.append(f"{row['above'][name]['tokens_est'] / 1e9:,.2f} B ({row['above'][name]['chars_pct']}%)")
            lines.append(f"| {row['source']} | {row['corpus']['docs']:,} | {row['corpus']['tokens'] / 1e9:,.2f} B | "
                         + " | ".join(cells) + " |")
        total = sum(row["corpus"]["tokens"] for row in rows)
        lines.append(f"| **Total** | {sum(row['corpus']['docs'] for row in rows):,} | {total / 1e9:,.1f} B | "
                     + " | ".join(f"**{value / 1e9:,.1f} B ({100 * value / total:.1f}%)**" for value in totals.values()) + " |")
    lines += ["", "## Médias do score por categoria da fonte", ""]
    for row in rows:
        for name, groups in row["source_scores"]["groups"].items():
            if groups:
                values = ", ".join(f"{key}: {value['mean_score']} ({value['docs']})" for key, value in groups.items())
                lines.append(f"- **{row['source']}** · `{name}`: {values}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--catalog")
    parser.add_argument("--checks", nargs="*", default=[])
    parser.add_argument("--before-after")
    parser.add_argument("--sample-summary")
    args = parser.parse_args()
    volumes = (corpus_volumes(args.catalog, args.checks, args.before_after, args.sample_summary)
               if args.catalog and args.before_after and args.sample_summary else {})
    rows = []
    for directory in sorted(Path(args.sample).glob("source=*")):
        source = directory.name.split("=", 1)[1]
        parts = sorted(directory.glob("part-*.parquet"))
        sample = pd.concat([pq.read_table(path, columns=["doc_uid", "n_chars", "source_config", "source_meta"]).to_pandas()
                            for path in parts], ignore_index=True)
        labels = pd.concat([pq.read_table(Path(args.labels) / path.relative_to(args.sample)).to_pandas()
                            for path in parts], ignore_index=True)
        frame = sample.merge(labels, on="doc_uid", how="inner", validate="one_to_one")
        if len(frame) != len(sample):
            raise RuntimeError(f"{source}: {len(frame)} rótulos para {len(sample)} documentos da amostra")
        rows.append(summarize(source, frame, volumes.get(source)))
        print(json.dumps({key: rows[-1][key] for key in ("source", "ok", "class_pct", "score_mean")}), flush=True)
    Path(args.out_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    Path(args.out_md).write_text(markdown(rows), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
