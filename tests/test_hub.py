import functools
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip('datasets')
from datasets import load_dataset
from huggingface_hub import HfApi

from educability_annotator import hub, pipeline, state
from educability_annotator.cli import parser
from educability_annotator.common import AnnotationError
from educability_annotator.data import PartWriter, read_windows
from educability_annotator.model import Classifier
from educability_annotator.planning import prepare
from educability_annotator.runner import execute
from educability_annotator.state import verify, metadata_path, part_path
from conftest import output_rows, run_args
from test_distributed import compare, finish


@pytest.fixture
def streamed_job(tmp_path, tiny_model, monkeypatch):
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(HfApi, 'dataset_info', lambda *a, **kw: SimpleNamespace(sha='a' * 40))
    counter = 0

    def make(count=15, shards=3, output_format='parquet', input_format='parquet'):
        nonlocal counter
        counter += 1
        files = []
        rows = [{'id': i, 'text': 'a b ciência ' * (i % 4 + 1),
                 'nested': {'name': 'São Paulo', 'values': [i, None]}}
                for i in range(count)]
        for index in range(shards):
            path = tmp_path / f'input-{counter}-{index}.{input_format}'
            subset = rows[index * count // shards:(index + 1) * count // shards]
            schema = pa.schema([('id', pa.int64()), ('text', pa.string()),
                                ('nested', pa.struct([('name', pa.string()), ('values', pa.list_(pa.int64()))]))])
            if input_format == 'parquet':
                pq.write_table(pa.Table.from_pylist(subset, schema=schema), path)
            else:
                path.write_text(''.join(json.dumps(row) + '\n' for row in subset), encoding='utf-8')
            files.append(f'http://127.0.0.1:{server.server_port}/{path.name}')

        def open_fixture(source):
            options = {'batch_size': 256} if input_format == 'parquet' else {}
            stream = load_dataset(input_format, data_files={'train': files}, split='train', streaming=True, **options)
            if stream.features:
                stream = stream.decode(False)
            return stream, 'default', files
        monkeypatch.setattr(hub, 'open_stream', open_fixture)
        output = tmp_path / f'out-{counter}'
        manifest = prepare(None, output, 'text', str(tiny_model), dataset='test/corpus',
                           dataset_split='train', output_format=output_format)
        return output, manifest, rows, files
    yield make
    server.shutdown()
    server.server_close()
    thread.join()


def args(output, *extra):
    result = run_args(output, *extra)
    result.offline = False
    return result


@pytest.mark.parametrize('output_format', ['jsonl', 'parquet'])
@pytest.mark.parametrize('count', [0, 4, 15])
def test_stream_roundtrip(streamed_job, output_format, count):
    output, manifest, originals, _ = streamed_job(count=count, output_format=output_format)
    execute(args(output))
    rows = output_rows(output)
    assert [{k: row[k] for k in ('id', 'text', 'nested')} for row in rows] == originals
    assert verify(output, manifest, full=True)['complete']
    execute(args(output, '--resume'))
    assert output_rows(output) == rows


@pytest.mark.parametrize('phase', ['during_write', 'after_rename', 'after_commit'])
def test_stream_resume(streamed_job, monkeypatch, phase):
    output, manifest, originals, _ = streamed_job(shards=1)
    if phase == 'during_write':
        original = PartWriter.write
        def fail(self, *values):
            original(self, *values)
            raise OSError('simulated')
        target, name = PartWriter, 'write'
    elif phase == 'after_rename':
        def fail(*values):
            raise OSError('simulated')
        target, name = pipeline, 'publish_commit'
    else:
        original = state.atomic_json
        def fail(path, value):
            original(path, value)
            if path.name.endswith('.commit.json'):
                raise OSError('simulated')
        target, name = state, 'atomic_json'
    with monkeypatch.context() as patch:
        patch.setattr(target, name, fail)
        with pytest.raises(OSError, match='simulated'):
            execute(args(output))
    seen = []
    original_predict = Classifier.predict
    def track(self, prepared):
        result = original_predict(self, prepared)
        seen.extend(result[0])
        return result
    monkeypatch.setattr(Classifier, 'predict', track)
    task = manifest['tasks'][0]
    saved = part_path(output, task, 0).stat().st_mtime_ns if phase == 'after_commit' else None
    execute(args(output, '--resume', '--recover-lock', '--read-rows', '3'))
    assert [r['id'] for r in output_rows(output)] == [r['id'] for r in originals]
    assert len(seen) == (11 if phase == 'after_commit' else 15)
    assert verify(output, manifest, full=True)['complete']
    if saved:
        assert part_path(output, task, 0).stat().st_mtime_ns == saved


def test_checkpoint_snapshot_and_corruption(streamed_job):
    output, manifest, _, _ = streamed_job(shards=1)
    task = manifest['tasks'][0]
    windows = list(read_windows(task, 0, 'text', 'educability_', read_rows=2))
    resumed = list(read_windows(task, 2, 'text', 'educability_', read_rows=3,
                               stream_state=windows[0].stream_state))
    assert resumed[0].payload['id'].to_pylist() == [2, 3, 4]
    execute(args(output))
    part = part_path(output, task, 0)
    raw = bytearray(part.read_bytes()); raw[10] ^= 1; part.write_bytes(raw)
    with pytest.raises(AnnotationError, match='Checksum'):
        verify(output, manifest, full=True)


def test_cli_and_constraints(streamed_job):
    parsed = parser().parse_args(['prepare', '--dataset', 'x/y', '--dataset-split', 'train', '--output', 'out'])
    assert parsed.input is None and parsed.dataset == 'x/y'
    with pytest.raises(SystemExit):
        parser().parse_args(['prepare', '--dataset', 'x/y', '--input', 'file', '--output', 'out'])
    output, _, _, _ = streamed_job()
    with pytest.raises(AnnotationError, match='conectividade'):
        execute(run_args(output))


def test_pinned_files_and_auth(monkeypatch):
    sha = 'a' * 40
    url = f'hf://datasets/private/corpus@{sha}/train.parquet'
    assert hub.validate_files([url], 'private/corpus', sha) == [url]
    for invalid in ['https://external.example/data', 'hf://datasets/private/corpus@main/a']:
        with pytest.raises(AnnotationError, match='externos'):
            hub.validate_files([invalid], 'private/corpus', sha)
    calls = []
    sentinel = SimpleNamespace(features={'text': 'string'}, decode=lambda enabled: ('stream', enabled))
    builder = SimpleNamespace(config=SimpleNamespace(name='pt', data_files={'train': [url]}),
                              as_streaming_dataset=lambda split: sentinel)
    def load(repo, **kw):
        calls.append((repo, kw))
        return builder
    monkeypatch.setenv('HF_TOKEN', 'test-secret-never-persist')
    monkeypatch.setattr('datasets.load_dataset_builder', load)
    source = dict(repo='private/corpus', config='pt', split='train', revision=sha)
    result, config, _ = hub.open_stream(source)
    assert result == ('stream', False) and config == 'pt'
    assert calls == [('private/corpus', {'name': 'pt', 'revision': sha})]
    assert 'test-secret' not in json.dumps(source)
    with pytest.raises(AnnotationError, match='Split'):
        hub.open_stream(source | {'split': 'missing'})


def launch_hub(output, files, ranks, extra=()):
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
               '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:0',
               f'--rdzv-id={uuid.uuid4().hex}', '--local-addr=127.0.0.1',
               f'--nproc-per-node={ranks}', '--max-restarts=0', str(Path(__file__).with_name('hub_worker.py')),
               'run', '--output', str(output), '--device', 'cpu', '--read-rows', '2',
               '--checkpoint-rows', '4', '--batch-size', '2', *extra]
    logfile = output.parent / f'worker-{uuid.uuid4().hex}.log'
    env = os.environ | {'EDUCABILITY_TEST_STREAM_FILES': json.dumps(files), 'OMP_NUM_THREADS': '1'}
    with logfile.open('wb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return process, logfile


@pytest.mark.integration
@pytest.mark.parametrize('world', [2, 3])
def test_stream_distributed(streamed_job, world):
    sequential, _, _, _ = streamed_job(shards=2)
    execute(args(sequential))
    output, manifest, _, files = streamed_job(shards=2)
    process, logfile = launch_hub(output, files, world)
    finish(process, logfile)
    compare(output_rows(sequential), output_rows(output))
    assert verify(output, manifest, full=True)['complete']


@pytest.mark.integration
def test_stream_kill_resume(streamed_job):
    output, manifest, originals, files = streamed_job(count=600, shards=3)
    process, logfile = launch_hub(output, files, 2)
    deadline = time.monotonic() + 90
    try:
        while not list((output / 'parts').glob('*/*.commit.json')):
            if process.poll() is not None or time.monotonic() > deadline:
                pytest.fail(logfile.read_text())
            time.sleep(.02)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    confirmed = {p: p.stat().st_mtime_ns for p in (output / 'parts').glob('*/*.commit.json')}
    process, logfile = launch_hub(output, files, 3, ('--resume', '--recover-lock'))
    finish(process, logfile)
    assert [r['id'] for r in output_rows(output)] == [r['id'] for r in originals]
    assert all(p.stat().st_mtime_ns == mtime for p, mtime in confirmed.items())
    assert verify(output, manifest, full=True)['complete']


@pytest.mark.hub
@pytest.mark.skipif(os.getenv('EDUCABILITY_TEST_DATASET_HUB') != '1', reason='opt-in: acesso real ao Hub')
def test_public_dataset(tmp_path, tiny_model):
    output = tmp_path / 'real-hub'
    manifest = prepare(None, output, 'text', str(tiny_model),
                       dataset='cornell-movie-review-data/rotten_tomatoes', dataset_split='test')
    execute(args(output, '--read-rows', '32', '--checkpoint-rows', '256'))
    assert verify(output, manifest, full=True)['complete']
    assert len(output_rows(output)) == 1066


def test_hub_access_error_is_actionable(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError('secret-token')
    monkeypatch.setattr('datasets.load_dataset_builder', denied)
    with pytest.raises(AnnotationError, match='acesso') as error:
        hub.open_stream(dict(repo='private/data', config=None, revision='a' * 40, split='train'))
    assert 'secret-token' not in str(error.value)


@pytest.mark.parametrize('case', ['collision', 'json_binary', 'missing', 'null', 'non_string'])
def test_hub_field_contract(tmp_path, tiny_model, monkeypatch, case):
    from datasets import Dataset
    record = {'id': 1, 'text': 'ciência', 'nested': {'items': [1, None]}}
    if case == 'collision':
        record['educability_class'] = 3
    elif case == 'json_binary':
        record['blob'] = b'abc'
    elif case == 'missing':
        record.pop('text')
    elif case == 'null':
        record['text'] = None
    else:
        record['text'] = 123
    dataset = Dataset.from_list([record])
    monkeypatch.setattr(HfApi, 'dataset_info', lambda *a, **kw: SimpleNamespace(sha='a' * 40))
    monkeypatch.setattr(hub, 'open_stream', lambda source: (dataset.to_iterable_dataset(), 'default', ['fixture']))
    output = tmp_path / 'out'
    if case == 'collision':
        with pytest.raises(AnnotationError, match='Colunas'):
            prepare(None, output, 'text', str(tiny_model), dataset='x/y', dataset_split='train')
        return
    manifest = prepare(None, output, 'text', str(tiny_model), dataset='x/y', dataset_split='train', output_format='jsonl')
    if case == 'json_binary':
        with pytest.raises(AnnotationError, match='representável'):
            execute(args(output))
        return
    execute(args(output))
    row = output_rows(output)[0]
    assert row['educability_class'] is None
    assert row['educability_status'] != 'ok'
    assert all(row[k] == v for k, v in record.items())
    assert verify(output, manifest)['complete']


def test_changed_source_and_missing_split(tmp_path, tiny_model, monkeypatch):
    with pytest.raises(AnnotationError, match='obrigatório'):
        prepare(None, tmp_path / 'out', 'text', str(tiny_model), dataset='x/y')
    stream = SimpleNamespace(num_shards=2)
    monkeypatch.setattr(hub, 'open_stream', lambda source: (stream, 'default', []))
    with pytest.raises(AnnotationError, match='shards'):
        hub.check_source({'num_shards': 1})


def test_stream_matches_local_pipeline(streamed_job, make_job):
    remote, _, originals, _ = streamed_job()
    _, local, _ = make_job(originals)
    execute(run_args(local))
    execute(args(remote))
    compare(output_rows(local), output_rows(remote))


def test_nested_schema_drift_is_not_silently_discarded():
    with pytest.raises(AnnotationError, match='aninhado'):
        hub.validate_shape([{'known': 1, 'new': 2}], pa.list_(pa.struct([('known', pa.int64())])))


@pytest.mark.parametrize('output_format', ['parquet', 'jsonl'])
def test_http_json_source_resume(streamed_job, output_format):
    output, manifest, originals, _ = streamed_job(input_format='json', output_format=output_format, shards=1)
    task = manifest['tasks'][0]
    first = next(read_windows(task, 0, 'text', 'educability_', read_rows=2))
    rest = list(read_windows(task, 2, 'text', 'educability_', read_rows=5, stream_state=first.stream_state))
    assert sum(window.rows for window in rest) == len(originals) - 2
    assert rest[0].texts[0] == {'text': originals[2]['text']}
    execute(args(output))
    assert [row['id'] for row in output_rows(output)] == list(range(15))
    assert verify(output, manifest, full=True)['complete']
