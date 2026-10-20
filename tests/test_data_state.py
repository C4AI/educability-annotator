import datetime
import json
import os
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from educability_annotator.common import AnnotationError, annotation_names, read_json
from educability_annotator.data import parse_record, read_windows
from educability_annotator.model import Classifier
from educability_annotator.planning import discover, jsonl_ranges, load_manifest, prepare
from educability_annotator.runner import execute
from educability_annotator.state import metadata_path, part_path, recover_state, verify
from conftest import output_rows, run_args


@pytest.mark.parametrize("block", [1, 2, 3, 10, 29, 1000])
def test_jsonl_boundaries(tmp_path, block):
    path = tmp_path / "data.jsonl"
    rows = [{"id": 0, "text": "educação 😀"}, {"id": 1, "text": "b" * 91}, {"id": 2}]
    path.write_bytes(b"\r\n".join(json.dumps(row, ensure_ascii=False).encode() for row in rows))
    ranges = jsonl_ranges(path, block)
    actual = []
    for i, (start, end) in enumerate(ranges):
        task = {"id": str(i), "format": "jsonl", "path": str(path), "start": start, "end": end}
        for window in read_windows(task, start, "text", "educability_", read_rows=1):
            actual.extend(parse_record(raw, "test") for raw in window.payload)
    assert actual == rows


@pytest.mark.parametrize("raw", [b"", b"\n", b"null\n", b"[]\n", b"{bad}", b'{"x":NaN}',
                                 b'{"x":1,"x":2}', b'{"x":"\xff"}'])
def test_malformed_json(raw):
    with pytest.raises(AnnotationError):
        parse_record(raw, "test")


def test_preservation_invalid_status_and_resume(make_job, monkeypatch):
    rows = [{"id": i, "text": text, "nested": {"a": [1, None, "😀"]}}
            for i, text in enumerate(["ciência", "", None, 20, "a " * 1200])]
    source, output, manifest = make_job(rows)
    before = source.read_bytes()
    execute(run_args(output))
    actual = output_rows(output)
    for original, annotated in zip(rows, actual, strict=True):
        assert {key: annotated[key] for key in original} == original
    assert actual[4]["educability_truncated"] is True
    assert [row["educability_status"] for row in actual] == ["ok", "empty_text", "null_text", "non_string_text", "ok"]
    assert source.read_bytes() == before
    assert verify(output, manifest, full=True)["complete"]
    monkeypatch.setattr(Classifier, "__init__", lambda *a, **k: pytest.fail("retomada recarregou o modelo"))
    result = execute(run_args(output, "--resume"))
    assert result["new_rows"] == 0


def test_json_numeric_lexemes(tmp_path, tiny_model):
    source = tmp_path / "numbers.jsonl"
    source.write_bytes(b'  {"text":"a","big":123456789123456789123456789,"n":0.123456789123456789123456789}\n   {}  \n')
    output = tmp_path / "out"
    manifest = prepare([str(source)], output, "text", str(tiny_model))
    execute(run_args(output))
    raw = part_path(output, manifest["tasks"][0], 0).read_bytes()
    assert b"0.123456789123456789123456789" in raw
    assert len(output_rows(output)) == 2


def test_parquet_types_metadata_and_resharding(tmp_path, tiny_model):
    schema = pa.schema([
        pa.field("id", pa.int32(), metadata={b"unit": b"id"}),
        pa.field("text", pa.string()), pa.field("price", pa.decimal128(12, 3)),
        pa.field("at", pa.timestamp("us", tz="UTC")), pa.field("nested", pa.list_(pa.int16())),
        pa.field("raw", pa.binary()),
    ], metadata={b"owner": b"research"})
    table = pa.Table.from_pylist([
        {"id": i, "text": "a" if i % 2 else None, "price": Decimal("12.345"),
         "at": datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
         "nested": [1, None, 3], "raw": bytes([i])} for i in range(7)
    ], schema=schema)
    source, output = tmp_path / "input.parquet", tmp_path / "out"
    pq.write_table(table, source, use_compliant_nested_type=False)
    original_bytes = source.read_bytes()
    manifest = prepare([str(source)], output, "text", str(tiny_model), parquet_shard_rows=2)
    assert len(manifest["tasks"]) == 4
    execute(run_args(output))
    parts = [pq.read_table(part_path(output, task, 0)) for task in manifest["tasks"]]
    actual = pa.concat_tables(parts).select(schema.names)
    assert actual.equals(table, check_metadata=True)
    assert source.read_bytes() == original_bytes
    assert verify(output, manifest, True)["rows"] == 7


@pytest.mark.parametrize("file_format", ["parquet", "jsonl"])
def test_empty_input(tmp_path, tiny_model, file_format):
    source = tmp_path / f"empty.{file_format}"
    if file_format == "jsonl":
        source.write_bytes(b"")
    else:
        pq.write_table(pa.table({"text": pa.array([], type=pa.string())}), source)
    output = tmp_path / "out"
    manifest = prepare([str(source)], output, "text", str(tiny_model))
    execute(run_args(output))
    assert output_rows(output) == []
    assert verify(output, manifest, full=True)["complete"]


def test_collision_fails_without_discard(make_job):
    _, output, _ = make_job([{"text": "a", "educability_class": 3}])
    with pytest.raises(AnnotationError, match="já existem"):
        execute(run_args(output))
    assert not (output / "COMPLETE.json").exists()


def test_corrupt_committed_part(make_job):
    _, output, manifest = make_job()
    execute(run_args(output))
    path = part_path(output, manifest["tasks"][0], 0)
    data = path.read_bytes()
    path.write_bytes(data.replace(b'"id": 0', b'"id": 9', 1))
    with pytest.raises(AnnotationError, match="Checksum"):
        verify(output, manifest, full=True)


def test_missing_part(make_job):
    _, output, manifest = make_job()
    execute(run_args(output))
    part_path(output, manifest["tasks"][0], 0).unlink()
    with pytest.raises(AnnotationError, match="ausente"):
        execute(run_args(output, "--resume"))


def test_modified_input_and_contract(make_job):
    source, output, _ = make_job()
    execute(run_args(output))
    with pytest.raises(AnnotationError, match="incompatíveis"):
        execute(run_args(output, "--resume", "--attention", "eager"))
    source.write_text(source.read_text() + '{"text":"new"}\n')
    with pytest.raises(AnnotationError, match="alterada"):
        execute(run_args(output, "--resume", "--recover-lock"))


def test_strict_fingerprint_detects_same_size_mtime(make_job):
    source, output, manifest = make_job(strict=True)
    execute(run_args(output))
    stat = source.stat()
    source.write_bytes(source.read_bytes().replace(b'"id": 0', b'"id": 9', 1))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(AnnotationError, match="sha256"):
        verify(output, manifest, full=True)


def test_manifest_tampering(make_job):
    _, output, _ = make_job()
    path = output / "manifest.json"
    manifest = read_json(path)
    manifest["config"]["max_document_tokens"] = 5000
    path.write_text(json.dumps(manifest))
    with pytest.raises(AnnotationError, match="alterado"):
        load_manifest(output)


def test_globs_directories_duplicates(tmp_path):
    for name in ["a.jsonl", "b.parquet"]:
        (tmp_path / name).write_bytes(b"")
    paths = discover([str(tmp_path), str(tmp_path / "*.jsonl")], tmp_path / "out")
    assert [p.name for p in paths] == ["a.jsonl", "b.parquet"]


def test_literal_filename_with_glob_characters(tmp_path):
    source = tmp_path / "corpus [2026].jsonl"
    source.write_bytes(b"")
    assert discover([str(source)], tmp_path / "out") == [source.resolve()]
