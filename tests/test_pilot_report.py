import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pilot_report  # noqa: E402


def write_source(root: Path, labels: Path, source: str, scores: np.ndarray, start_uid: int) -> None:
    rng = np.random.default_rng(start_uid)
    for part, chunk in enumerate(np.array_split(np.arange(len(scores)), 2)):
        uids = start_uid + chunk
        expected = scores[chunk]
        classes = np.clip(np.rint(expected), 1, 4).astype(np.int8)
        sample_dir, label_dir = root / f"source={source}", labels / f"source={source}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        metas = [json.dumps({"edu_score": float(s + rng.normal(0, 0.3)), "register": "IN" if s > 2.5 else "OP"})
                 for s in expected]
        pq.write_table(pa.table({"doc_uid": uids, "n_chars": rng.integers(100, 9000, len(uids)),
                                 "source_config": ["edu_high" if s > 2.5 else "edu_low" for s in expected],
                                 "source_meta": metas}), sample_dir / f"part-{part:06d}.parquet")
        status = np.where(np.arange(len(uids)) == 0, "empty_text", "ok")
        invalid = status != "ok"
        pq.write_table(pa.table({
            "doc_uid": uids,
            "educability_class": pa.array(classes, mask=invalid),
            "educability_expected_score": pa.array(expected.astype(np.float32), mask=invalid),
            "educability_max_chunk_score": pa.array((expected + 0.2).astype(np.float32), mask=invalid),
            "educability_first_chunk_score": pa.array((expected - 0.1).astype(np.float32), mask=invalid),
            "educability_tokens_used": pa.array(rng.integers(10, 5000, len(uids)).astype(np.int32), mask=invalid),
            "educability_chunk_count": pa.array(rng.integers(1, 6, len(uids)).astype(np.int16), mask=invalid),
            "educability_truncated": pa.array(rng.random(len(uids)) < 0.1, mask=invalid),
            "educability_status": status,
        }), label_dir / f"part-{part:06d}.parquet")


def test_report_with_corpus_volumes(tmp_path):
    sample, labels = tmp_path / "sample", tmp_path / "labels"
    write_source(sample, labels, "alpha", np.linspace(1.0, 4.0, 400), 1000)
    write_source(sample, labels, "beta", np.full(300, 1.5), 5000)
    (tmp_path / "sources.yaml").write_text(
        "sources:\n  - id: alpha\n    counted: {docs: 1000, tokens: 2000000}\n  - id: beta\n    counted: {docs: 600, tokens: 900000}\n")
    (tmp_path / "check.json").write_text(json.dumps({"sources": {"alpha": {"kept_tokens": None, "excluded_tokens": {"x": 100000}},
                                                                 "beta": {"kept_tokens": 800000}}}))
    (tmp_path / "ba.jsonl").write_text(
        json.dumps({"s": "alpha", "docs": 1000, "chars": 1000, "rm_docs": 100, "rm_chars": 250}) + "\n"
        + json.dumps({"s": "beta", "docs": 600, "chars": 500, "rm_docs": 0, "rm_chars": 0}) + "\n")
    (tmp_path / "summary.json").write_text(json.dumps({"sources": [{"source": "alpha", "kept": 900}, {"source": "beta", "kept": 600}]}))
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    sys.argv = ["pilot_report.py", "--sample", str(sample), "--labels", str(labels), "--out-json", str(out_json),
                "--out-md", str(out_md), "--catalog", str(tmp_path / "sources.yaml"), "--checks", str(tmp_path / "check.json"),
                "--before-after", str(tmp_path / "ba.jsonl"), "--sample-summary", str(tmp_path / "summary.json")]
    assert pilot_report.main() == 0
    rows = {row["source"]: row for row in json.loads(out_json.read_text())}
    assert rows["alpha"]["corpus"] == {"docs": 900, "tokens": round((2000000 - 100000) * 750 / 1000)}
    assert rows["beta"]["corpus"] == {"docs": 600, "tokens": 800000}
    assert rows["beta"]["above"]["classe>=3"]["tokens_est"] == 0
    assert 0 < rows["alpha"]["above"]["classe=4"]["docs_pct"] < rows["alpha"]["above"]["classe>=3"]["docs_pct"] < 100
    assert rows["alpha"]["source_scores"]["spearman"]["edu_score"] == pytest.approx(0.9, abs=0.1)
    assert set(rows["alpha"]["source_scores"]["groups"]["source_config"]) == {"edu_low", "edu_high"}
    assert "Total" in out_md.read_text()
