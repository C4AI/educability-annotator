import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from educability_annotator.common import MODEL_REVISION
from educability_annotator.model import Classifier, inspect_model
from educability_annotator.sidecar.__main__ import main as sidecar
from educability_annotator.sidecar.chunking import OK, STATUSES, Chunker
from educability_annotator.sidecar.packing import pack

WORDS = ["a", "b", "c", "educação", "ciência"]
SEPARATORS = [" ", " ", "\n", "  ", "\t"]


def words(count: int, seed: int = 0) -> str:
    return "".join(WORDS[(i * 7 + seed) % 5] + SEPARATORS[(i + seed) % 5] for i in range(count)).rstrip()


def tokenizer_json(model) -> str:
    return (Path(model) / "tokenizer.json").read_text(encoding="utf-8")


def reference_chunks(prepared, documents: int) -> list[list[list[int]]]:
    chunks = [[] for _ in range(documents)]
    for chunk in prepared.chunks:
        chunks[chunk.document].append(chunk.inputs["input_ids"])
    return chunks


def sidecar_chunks(chunker: Chunker, tokenized, documents: int, budget: int = 10**6) -> list[list[list[int]]]:
    chunks = [[] for _ in range(documents)]
    offset = 0
    for batch in pack(chunker, tokenized, np.arange(documents), budget):
        bounds = batch["cu_seqlens"]
        for index, document in enumerate(batch["chunk_doc"]):
            chunks[offset + document].append(batch["input_ids"][bounds[index]:bounds[index + 1]].tolist())
        offset += len(batch["doc_id"])
    return chunks


def compare_with_classifier(classifier: Classifier, chunker: Chunker, texts: list) -> int:
    expected = classifier.prepare([{"text": text} for text in texts], "text")
    sliced = [text[:chunker.prefix_chars] if isinstance(text, str) else text for text in texts]
    longer = [isinstance(text, str) and len(text) > chunker.prefix_chars for text in texts]
    tokenized = chunker.tokenize(sliced, longer, full_text=lambda index: texts[index])
    assert [STATUSES[status] for status in tokenized.status] == [row["status"] for row in expected.results]
    for document, row in enumerate(expected.results):
        if row["status"] == "ok":
            assert tokenized.tokens_used[document] == row["tokens_used"]
            assert bool(tokenized.truncated[document]) == row["truncated"]
    assert sidecar_chunks(chunker, tokenized, len(texts)) == reference_chunks(expected, len(texts))
    return tokenized.retried


@pytest.mark.parametrize("chars_per_token", [1, 5])
def test_prefix_tokens_match_classifier(tiny_model, chars_per_token):
    classifier = Classifier(inspect_model(str(tiny_model)), 5110, "cpu", "float32")
    chunker = Chunker(tokenizer_json(tiny_model), 1022, 5110, chars_per_token)
    texts = [words(count, count) for count in (1, 2, 1021, 1022, 1023, 2044, 5109, 5110, 5111, 7000)]
    texts += ["", " \n\t ", None, " " * 30000 + "a b", "educação" * 9000 + " a"]
    retried = compare_with_classifier(classifier, chunker, texts)
    assert retried > 0  # espaço inicial, palavra gigante e, com 1 caractere/token, prefixos curtos


def test_pack_keeps_documents_whole(tiny_model):
    chunker = Chunker(tokenizer_json(tiny_model), 1022, 3066, 5)
    texts = [words(count, count) for count in (5, 3000, 1, 2100, 0, 900, 4000)]
    tokenized = chunker.tokenize(texts)
    budget = 3 * 1024
    batches = list(pack(chunker, tokenized, np.arange(100, 100 + len(texts)), budget))
    assert np.concatenate([batch["doc_id"] for batch in batches]).tolist() == list(range(100, 107))
    for batch in batches:
        assert batch["cu_seqlens"][-1] == len(batch["input_ids"]) <= budget
        weights = np.bincount(batch["chunk_doc"], weights=batch["chunk_weight"], minlength=len(batch["doc_id"]))
        assert weights.tolist() == batch["tokens_used"].tolist()
        firsts = np.bincount(batch["chunk_doc"][batch["chunk_first"]], minlength=len(batch["doc_id"]))
        assert firsts.tolist() == (batch["status"] == OK).astype(int).tolist()
    with pytest.raises(ValueError):
        next(pack(chunker, tokenized, np.arange(len(texts)), 1000))


def write_corpus(root: Path, documents: int = 40) -> dict[int, object]:
    texts = {}
    for source_index, source in enumerate(("alpha", "beta")):
        directory = root / f"source={source}"
        directory.mkdir(parents=True)
        for part in range(2):
            uids, values = [], []
            for row in range(documents // 4):
                uid = (source_index << 40) + (part << 20) + row
                count = [0, 3, 40, 1100, 2500, 3500][(row + part) % 6]
                value = None if row == 5 else "" if row == 6 else words(count, uid % 13)
                uids.append(uid)
                values.append(value)
                texts[uid] = value
            pq.write_table(pa.table({"doc_uid": pa.array(uids, pa.int64()), "text": pa.array(values, pa.string()),
                                     "outra": pa.array(range(len(uids)))}),
                           directory / f"part-{part:06d}-0.parquet", row_group_size=3)
    return texts


def prepare_args(corpus: Path, model, output: Path, *extra) -> list[str]:
    return ["prepare", "--input-root", str(corpus), "--pattern", "source=*/part-*.parquet", "--model", str(model),
            "--offline", "--output", str(output), "--dtype", "float32", "--attention", "sdpa",
            "--max-document-tokens", "3066", "--threads", "2", *extra]


def run_args(output: Path, workers: int) -> list[str]:
    return ["run", "--output", str(output), "--device", "cpu", "--loader-workers", str(workers),
            "--read-rows", "4", "--batch-tokens", "8192", "--log-interval", "1000"]


def labels(output: Path) -> dict[int, dict]:
    table = pa.concat_tables([pq.read_table(path) for path in sorted(output.rglob("part-*.parquet"))])
    return {row["doc_uid"]: row for row in table.to_pylist()}


def test_end_to_end_matches_classifier(tmp_path, tiny_model):
    corpus, output = tmp_path / "unified", tmp_path / "labels"
    texts = write_corpus(corpus)
    excluded = sorted(uid for uid in texts if uid % 3 == 0)
    decisions = tmp_path / "decisions" / "source=alpha"
    decisions.mkdir(parents=True)
    pq.write_table(pa.table({"doc_uid": pa.array(excluded, pa.int64())}), decisions / "decisions.parquet")
    exclude = str(tmp_path / "decisions" / "*" / "decisions.parquet")
    assert sidecar(prepare_args(corpus, tiny_model, output, "--exclude", exclude)) == 0
    assert sidecar(run_args(output, workers=2)) == 0
    assert sidecar(["verify", "--output", str(output), "--full"]) == 0

    found = labels(output)
    assert set(found) == set(texts) - set(excluded)
    classifier = Classifier(inspect_model(str(tiny_model)), 3066, "cpu", "float32")
    uids = sorted(found)
    prepared = classifier.prepare([{"text": texts[uid]} for uid in uids], "text")
    chunk_scores = {}
    for chunk, probs in zip(prepared.chunks, classifier._forward(prepared.chunks)):
        chunk_scores.setdefault(chunk.document, []).append(sum((i + 1) * p for i, p in enumerate(probs)))
    expected, _ = classifier.predict(prepared)
    for document, (uid, row) in enumerate(zip(uids, expected)):
        got = found[uid]
        assert got["educability_status"] == row["status"]
        if row["status"] != "ok":
            assert got["educability_class"] is None and got["educability_p1"] is None
            continue
        assert got["educability_class"] == row["class"] and got["educability_label"] == row["label"]
        assert [got[f"educability_p{i}"] for i in range(1, 5)] == pytest.approx(
            [row[f"p{i}"] for i in range(1, 5)], abs=1e-6)
        assert got["educability_expected_score"] == pytest.approx(row["expected_score"], abs=1e-5)
        assert got["educability_max_chunk_score"] == pytest.approx(max(chunk_scores[document]), abs=1e-5)
        assert got["educability_first_chunk_score"] == pytest.approx(chunk_scores[document][0], abs=1e-5)
        assert (got["educability_tokens_used"], got["educability_chunk_count"], got["educability_truncated"]) == (
            row["tokens_used"], row["chunk_count"], row["truncated"])


def test_resume_redoes_only_missing_tasks(tmp_path, tiny_model, capsys):
    corpus, output = tmp_path / "unified", tmp_path / "labels"
    write_corpus(corpus, documents=16)
    assert sidecar(prepare_args(corpus, tiny_model, output)) == 0
    assert sidecar(run_args(output, workers=0)) == 0
    files = sorted(output.rglob("part-*.parquet"))
    removed = files[0]
    removed.unlink()
    removed.with_name(removed.name + ".commit.json").unlink()
    kept = {path: path.stat().st_mtime_ns for path in files[1:]}
    capsys.readouterr()
    assert sidecar(prepare_args(corpus, tiny_model, output)) == 0
    assert json.loads(capsys.readouterr().out)["pending"] == 1
    assert sidecar(run_args(output, workers=1)) == 0
    assert sidecar(["verify", "--output", str(output)]) == 0
    assert {path: path.stat().st_mtime_ns for path in files[1:]} == kept and removed.exists()


def test_sample_tasks_picks_a_fixed_subset(tmp_path, tiny_model, capsys):
    corpus = tmp_path / "unified"
    write_corpus(corpus, documents=8)
    found = []
    for name in ("a", "b"):
        capsys.readouterr()
        assert sidecar(prepare_args(corpus, tiny_model, tmp_path / name, "--sample-tasks", "2", "--seed", "3")) == 0
        assert json.loads(capsys.readouterr().out)["tasks"] == 2
        found.append([task["path"] for task in json.loads((tmp_path / name / "plan.json").read_text())["tasks"]])
    assert sorted(found[0]) == sorted(found[1])


def test_contract_change_needs_new_output(tmp_path, tiny_model):
    corpus, output = tmp_path / "unified", tmp_path / "labels"
    write_corpus(corpus, documents=8)
    assert sidecar(prepare_args(corpus, tiny_model, output)) == 0
    changed = prepare_args(corpus, tiny_model, output)
    changed[changed.index("float32")] = "bfloat16"
    assert sidecar(changed) == 2
    assert sidecar(prepare_args(corpus, tiny_model, output, "--chars-per-token", "4")) == 2


@pytest.mark.hub
@pytest.mark.skipif(os.environ.get("EDUCABILITY_TEST_HUB") != "1", reason="ative EDUCABILITY_TEST_HUB=1")
@pytest.mark.parametrize("chars_per_token", [1, 2, 5])
def test_prefix_tokens_match_published_tokenizer(chars_per_token):
    model = os.environ.get("EDUCABILITY_TEST_MODEL", "gregassagraf/educability-norberto-mini-4class-v1")
    spec = inspect_model(model, MODEL_REVISION)
    classifier = Classifier(spec, 5110, "cpu", "float32")
    chunker = Chunker(classifier.backend.to_str(), 1022, 5110, chars_per_token)
    base = "A educação é um direito de todos e dever do Estado e da família, promovida com a sociedade. "
    texts = [
        base * 600,
        "Ação, coração e informação. " * 2000,
        "Ação decomposta em NFD, néctar e pão. " * 1500,
        "=" * 30000 + " fim",
        "https://exemplo.gov.br/consulta?processo=" + "0123456789" * 3000 + " texto depois",
        "linha\n\n\n" * 5000 + "\t\tcom tabulações   e    espaços   " * 800,
        "palavra" * 6000,
        "简体中文 e português juntos 🙂 " * 2500,
        "1.234,56; 7.890,12; " * 3000,
        base,
    ]
    compare_with_classifier(classifier, chunker, texts)
