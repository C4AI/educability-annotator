import pytest

from educability_annotator import pipeline, state
from educability_annotator.common import AnnotationError
from educability_annotator.data import PartWriter
from educability_annotator.model import Classifier
from educability_annotator.runner import execute
from educability_annotator.state import metadata_path, part_path, verify
from conftest import output_rows, run_args


@pytest.mark.parametrize("phase", ["during_write", "after_rename", "after_commit"])
def test_interruption_at_publication_boundaries(make_job, monkeypatch, phase):
    _, output, manifest = make_job(jsonl_task_bytes=100000)
    task = manifest["tasks"][0]
    if phase == "during_write":
        original = PartWriter.write
        def fail(writer, *args):
            original(writer, *args)
            raise OSError("simulated write interruption")
        monkeypatch.setattr(PartWriter, "write", fail)
    elif phase == "after_rename":
        def fail(*args):
            raise OSError("simulated interruption before commit")
        monkeypatch.setattr(pipeline, "publish_commit", fail)
    else:
        original = state.atomic_json
        def fail(path, value):
            original(path, value)
            if path.name.endswith(".commit.json"):
                raise OSError("simulated interruption after commit")
        monkeypatch.setattr(state, "atomic_json", fail)
    with pytest.raises(OSError, match="simulated"):
        execute(run_args(output))
    assert not (output / "COMPLETE.json").exists()
    confirmed = metadata_path(output, task, 0).exists()
    assert confirmed == (phase == "after_commit")
    saved_mtime = part_path(output, task, 0).stat().st_mtime_ns if confirmed else None
    monkeypatch.undo()
    seen = []
    original_prepare = Classifier.prepare
    def tracked(classifier, rows, column):
        seen.extend(row[column] for row in rows)
        return original_prepare(classifier, rows, column)
    monkeypatch.setattr(Classifier, "prepare", tracked)
    execute(run_args(output, "--resume", "--recover-lock", "--read-rows", "3"))
    rows = output_rows(output)
    assert [row["id"] for row in rows] == list(range(15))
    assert verify(output, manifest, full=True)["complete"]
    if confirmed:
        assert part_path(output, task, 0).stat().st_mtime_ns == saved_mtime
        assert len(seen) == 11
    else:
        assert len(seen) == 15


def test_unconfirmed_orphan_not_reused(make_job):
    _, output, manifest = make_job()
    orphan = part_path(output, manifest["tasks"][0], 0)
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"not a result")
    execute(run_args(output))
    assert verify(output, manifest, full=True)["complete"]


def test_existing_run_requires_resume_and_explicit_lock_recovery(make_job):
    _, output, _ = make_job()
    execute(run_args(output))
    with pytest.raises(AnnotationError, match="resume"):
        execute(run_args(output))
    with pytest.raises(AnnotationError, match="bloqueada"):
        execute(run_args(output, "--resume"))
    execute(run_args(output, "--resume", "--recover-lock"))


def test_hole_in_commit_sequence(make_job):
    _, output, manifest = make_job(jsonl_task_bytes=100000)
    execute(run_args(output))
    metadata_path(output, manifest["tasks"][0], 0).unlink()
    with pytest.raises(AnnotationError, match="incompleta"):
        verify(output, manifest)


def test_parquet_resume_inside_row_group_with_different_read_window(tmp_path, tiny_model, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from educability_annotator.planning import prepare

    source, output = tmp_path / "input.parquet", tmp_path / "out"
    rows = [{"id": i, "text": "a b"} for i in range(15)]
    pq.write_table(pa.Table.from_pylist(rows), source)
    manifest = prepare([str(source)], output, "text", str(tiny_model))
    original = state.atomic_json
    def fail_after_commit(path, value):
        original(path, value)
        if path.name.endswith(".commit.json"):
            raise OSError("simulated")
    monkeypatch.setattr(state, "atomic_json", fail_after_commit)
    with pytest.raises(OSError, match="simulated"):
        execute(run_args(output))
    monkeypatch.undo()
    seen = []
    original_prepare = Classifier.prepare
    def track(classifier, records, column):
        seen.extend(records)
        return original_prepare(classifier, records, column)
    monkeypatch.setattr(Classifier, "prepare", track)
    execute(run_args(output, "--resume", "--recover-lock", "--read-rows", "3"))
    assert len(seen) == 11
    assert [row["id"] for row in output_rows(output)] == list(range(15))
    assert verify(output, manifest, full=True)["complete"]
