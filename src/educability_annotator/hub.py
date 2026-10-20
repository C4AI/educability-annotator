"""Hub streaming adapter. No credentials or downloaded corpus enter the manifest."""
from __future__ import annotations

import base64
import copy
import json
import re
from itertools import islice
from urllib.parse import unquote, urlparse

import pyarrow as pa

from .common import AnnotationError, digest
from .data import Window, check_columns

def dependencies():
    try:
        import datasets
    except ImportError as exc:
        raise AnnotationError('Instale o suporte Hub: pip install "educability-annotator[hub]"') from exc
    if datasets.__version__ != '4.4.1':
        raise AnnotationError('O leitor Hub requer datasets==4.4.1 para checkpoints compatíveis.')
    return datasets

def validate_files(files, repo, revision):
    """Only repository-owned, immutable data files may back a resumable stream."""
    paths = [str(path) for path in files]
    for path in paths:
        decoded = unquote(path)
        parsed = urlparse(decoded)
        valid = decoded.startswith(f'hf://datasets/{repo}@{revision}/') or (
            parsed.scheme == 'https' and parsed.netloc == 'huggingface.co'
            and parsed.path.startswith(f'/datasets/{repo}/resolve/{revision}/')
        )
        if not valid:
            raise AnnotationError('Dataset referencia dados externos ou revisão não imutável; origem não suportada.')
    return paths

def _open_stream(source):
    ds = dependencies()
    builder = ds.load_dataset_builder(
        source['repo'], name=source.get('config'), revision=source['revision'],
    )
    split = source['split']
    files = builder.config.data_files
    if files is None or split not in files:
        raise AnnotationError(f'Split indisponível: {split}')
    paths = validate_files(files[split], source['repo'], source['revision'])
    if source.get('files_digest') and digest(paths) != source['files_digest']:
        raise AnnotationError('Lista de arquivos Hub incompatível com o manifesto.')
    # A positive batch size also handles empty Parquet row groups in datasets 4.4.1.
    if builder.__class__.__module__ == 'datasets.packaged_modules.parquet.parquet':
        builder.config.batch_size = 256
    stream = builder.as_streaming_dataset(split=split)
    # Unknown features have no media decoder; decode(False) requires known features.
    if stream.features:
        stream = stream.decode(False)
    return stream, builder.config.name, paths


def open_stream(source):
    try:
        return _open_stream(source)
    except AnnotationError:
        raise
    except Exception as exc:
        raise AnnotationError(
            f"Falha ao abrir dataset Hub ({type(exc).__name__}); confira acesso, config, split e conectividade."
        ) from None

def schema_from_source(source):
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(source['schema'])))

def discover(repo, config, split, revision, prefix, output_format):
    from huggingface_hub import HfApi
    dependencies()
    if not split:
        raise AnnotationError('--dataset-split é obrigatório para entradas Hub.')
    try:
        sha = HfApi().dataset_info(repo, revision=revision or 'main').sha
    except Exception as exc:
        raise AnnotationError(f'Não foi possível resolver a revisão Hub ({type(exc).__name__}); confira acesso e conectividade.') from None
    if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{40}', sha):
        raise AnnotationError('Hub não retornou um commit imutável válido.')
    source = {'kind': 'hub', 'repo': repo, 'config': config, 'split': split,
              'revision': sha, 'datasets_version': '4.4.1'}
    stream, resolved_config, paths = open_stream(source)
    source['config'] = resolved_config
    source['files_digest'] = digest(paths)
    if stream.features is not None:
        schema = stream.features.arrow_schema
    else:
        sample = list(islice(iter(stream), 256))
        schema = pa.Table.from_pylist(sample).schema
    check_columns(schema.names, prefix)
    source['schema'] = base64.b64encode(schema.serialize().to_pybytes()).decode('ascii')
    source['num_shards'] = stream.num_shards
    tasks = [dict(id=f'hub-t{i:08d}', source_kind='hub', source=source,
                  shard_index=i, format=output_format, cost=1)
             for i in range(stream.num_shards)]
    return source, tasks

def check_source(source):
    stream, _, _ = open_stream(source)
    if stream.num_shards != source['num_shards']:
        raise AnnotationError('Quantidade de shards Hub mudou.')
    if stream.features is not None and not stream.features.arrow_schema.equals(schema_from_source(source)):
        raise AnnotationError('Schema Hub mudou.')


def validate_shape(value, field_type):
    """Arrow's schema constructor otherwise silently drops unknown struct fields."""
    if value is None:
        return
    if pa.types.is_struct(field_type) and isinstance(value, dict):
        if set(value) - set(field_type.names):
            raise AnnotationError('Campo aninhado não previsto no schema Hub.')
        for field in field_type:
            validate_shape(value.get(field.name), field.type)
    elif (pa.types.is_list(field_type) or pa.types.is_large_list(field_type)
          or pa.types.is_fixed_size_list(field_type)):
        for item in value:
            validate_shape(item, field_type.value_type)

def read_windows(task, cursor, text_column, prefix, read_rows, window_bytes,
                 max_record_bytes, stream_state=None):
    stream, _, _ = open_stream(task['source'])
    if stream.num_shards != task['source']['num_shards']:
        raise AnnotationError('Quantidade de shards Hub mudou.')
    stream = stream.shard(num_shards=stream.num_shards, index=task['shard_index'])
    if stream_state is not None:
        stream.load_state_dict(copy.deepcopy(stream_state))
    elif cursor:
        raise AnnotationError('Checkpoint Hub sem estado do iterador.')
    schema = schema_from_source(task['source'])
    iterator = iter(stream)
    while True:
        rows, raw_rows, texts, size = [], [], [], 0
        for _ in range(read_rows):
            try:
                row = next(iterator)
            except StopIteration:
                break
            except Exception as exc:
                raise AnnotationError(f"Falha de leitura Hub ({type(exc).__name__}); retome com --resume após resolver a origem.") from None
            check_columns(list(row), prefix)
            if set(row) != set(schema.names):
                raise AnnotationError('Colunas do streaming incompatíveis com o schema preparado.')
            for field in schema:
                validate_shape(row[field.name], field.type)
            table = pa.Table.from_pylist([row], schema=schema)
            if task['format'] == 'jsonl':
                try:
                    raw = json.dumps(row, ensure_ascii=True, allow_nan=False).encode('utf-8')
                except (TypeError, ValueError) as exc:
                    raise AnnotationError('Valor Hub não representável em JSON; use saída Parquet.') from exc
                raw_rows.append(raw)
                record_size = len(raw)
            else:
                record_size = table.nbytes
            if record_size > max_record_bytes:
                raise AnnotationError('Registro Hub excede --max-record-bytes.')
            rows.append(row)
            texts.append({text_column: row[text_column]} if text_column in row else {})
            size += record_size
            if size >= window_bytes:
                break
        if not rows:
            return
        end = cursor + len(rows)
        payload = raw_rows if task['format'] == 'jsonl' else pa.Table.from_pylist(rows, schema=schema)
        # Snapshot before prefetch advances the shared iterator. JSON is the on-disk contract.
        snapshot = json.loads(json.dumps(copy.deepcopy(stream.state_dict()), allow_nan=False))
        yield Window(payload, texts, cursor, end, snapshot)
        cursor = end
