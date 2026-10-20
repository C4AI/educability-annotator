import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

import pytest

from educability_annotator.common import AnnotationError, read_json
from educability_annotator.distributed import Topology
from educability_annotator.planning import assignments
from educability_annotator.runner import execute
from educability_annotator.state import verify
from conftest import output_rows, run_args


def launch(output, ranks, extra=(), launcher_extra=()):
    command = [
        sys.executable, "-m", "torch.distributed.run",
        *(launcher_extra or (
            "--nnodes=1", "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0",
            f"--rdzv-id={uuid.uuid4().hex}",
        )),
        "--local-addr=127.0.0.1",
        f"--nproc-per-node={ranks}", "--max-restarts=0",
        "-m", "educability_annotator", "run", "--output", str(output),
        "--device", "cpu", "--offline", "--read-rows", "2", "--checkpoint-rows", "4",
        "--batch-size", "2", "--join-timeout", "90", *extra,
    ]
    logfile = output.parent / f"torchrun-{uuid.uuid4().hex}.log"
    env = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "TOKENIZERS_PARALLELISM": "false"}
    with logfile.open("wb") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True,
        )
    return process, logfile


def finish(process, logfile, timeout=120):
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        pytest.fail(f"Processo excedeu timeout:\n{logfile.read_text()}")
    assert code == 0, logfile.read_text()


def compare(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if isinstance(a[key], float):
                assert a[key] == pytest.approx(b[key], abs=1e-6)
            else:
                assert a[key] == b[key]


@pytest.mark.parametrize("world", [1, 2, 3, 9])
def test_assignment_complete_disjoint_balanced(world):
    tasks = [{"id": str(i), "cost": value} for i, value in enumerate([1000, 500, 250, 100, 50, 1])]
    result = assignments(tasks, world)
    assert result == assignments(list(reversed(tasks)), world)
    flat = [task for rank in result for task in rank]
    assert len(flat) == len(set(flat)) == len(tasks)
    assert set(flat) == {task["id"] for task in tasks}


@pytest.mark.integration
@pytest.mark.parametrize("world", [2, 3])
def test_real_torchrun_equivalence(make_job, world):
    _, sequential, _ = make_job()
    _, distributed, manifest = make_job()
    execute(run_args(sequential))
    process, logfile = launch(distributed, world)
    finish(process, logfile)
    compare(output_rows(sequential), output_rows(distributed))
    assert verify(distributed, manifest, full=True)["complete"]
    assert not (distributed / ".run-lock").exists()


@pytest.mark.integration
def test_more_ranks_than_tasks(make_job):
    _, output, manifest = make_job([{"id": 0, "text": "a"}])
    process, logfile = launch(output, 3)
    finish(process, logfile)
    assert len(output_rows(output)) == 1
    assert verify(output, manifest)["complete"]


@pytest.mark.integration
def test_two_simulated_nodes(make_job):
    _, output, manifest = make_job()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    run_id = uuid.uuid4().hex
    common = [
        "--nnodes=2", "--rdzv-backend=c10d", f"--rdzv-endpoint=127.0.0.1:{port}",
        f"--rdzv-id={run_id}",
    ]
    processes = [
        launch(output, 1, launcher_extra=(*common, f"--node-rank={rank}"))
        for rank in range(2)
    ]
    try:
        for process, logfile in processes:
            finish(process, logfile)
    finally:
        for process, _ in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert [row["id"] for row in output_rows(output)] == list(range(15))
    assert verify(output, manifest, full=True)["complete"]


@pytest.mark.integration
def test_kill_and_resume_with_different_world_size(make_job):
    rows = [{"id": i, "text": "a b ciência " * 60} for i in range(300)]
    _, output, manifest = make_job(rows, jsonl_task_bytes=25000)
    process, logfile = launch(output, 2)
    deadline = time.monotonic() + 90
    commits = []
    try:
        while time.monotonic() < deadline:
            commits = list((output / "parts").glob("*/*.commit.json"))
            if commits:
                break
            if process.poll() is not None:
                pytest.fail(logfile.read_text())
            time.sleep(0.01)
        assert commits, logfile.read_text()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    saved = {path: path.stat().st_mtime_ns for path in (output / "parts").glob("*/*.commit.json")}
    assert not (output / "COMPLETE.json").exists()
    process, logfile = launch(output, 3, ("--resume", "--recover-lock", "--full-checksums"))
    finish(process, logfile)
    assert all(path.stat().st_mtime_ns == mtime for path, mtime in saved.items())
    assert [row["id"] for row in output_rows(output)] == list(range(300))
    assert verify(output, manifest, full=True)["complete"]


def test_partial_rank_environment(monkeypatch):
    for name in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(AnnotationError, match="juntos"):
        Topology.from_environment()


def test_lock_conflict_without_mutating_owner(make_job):
    from educability_annotator.distributed import Job
    _, output, _ = make_job()
    owner = Job(output, Topology(0, 1, 0, "first"))
    owner.join({"dtype": "float32"}, False)
    original = read_json(output / ".run-lock" / "owner.json")
    competitor = Job(output, Topology(0, 1, 0, "second"))
    with pytest.raises(AnnotationError, match="bloqueada"):
        competitor.join({"dtype": "float32"}, False)
    assert read_json(output / ".run-lock" / "owner.json") == original
    owner.release()


def test_incompatible_rank_signatures_fail_before_processing(make_job):
    from concurrent.futures import ThreadPoolExecutor
    from educability_annotator.distributed import Job

    _, output, _ = make_job()
    first = Job(output, Topology(0, 2, 0, "same-job"), timeout=3)
    second = Job(output, Topology(1, 2, 1, "same-job"), timeout=3)
    with ThreadPoolExecutor(max_workers=2) as executor:
        rank0 = executor.submit(first.join, {"dtype": "float32"}, False)
        rank1 = executor.submit(second.join, {"dtype": "bfloat16"}, False)
        rank1.result(timeout=5)
        with pytest.raises(AnnotationError, match="incompatíveis") as error:
            rank0.result(timeout=5)
        first.fail(error.value)
    with pytest.raises(AnnotationError, match="incompatíveis"):
        second.ready()
    assert not (output / "COMPLETE.json").exists()
