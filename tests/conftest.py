import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import ModernBertConfig, ModernBertForSequenceClassification, PreTrainedTokenizerFast

from educability_annotator.cli import parser
from educability_annotator.data import iter_output
from educability_annotator.planning import load_manifest, prepare
from educability_annotator.state import metadata_path, part_path


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("model")
    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3,
                  "a": 4, "b": 5, "c": 6, "educação": 7, "ciência": 8}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        bos_token="[CLS]", eos_token="[SEP]", model_max_length=1024,
    )
    tokenizer.save_pretrained(path)
    config = ModernBertConfig(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=1024,
        local_attention=8, global_attn_every_n_layers=1, pad_token_id=1,
        cls_token_id=2, sep_token_id=3, reference_compile=False, num_labels=4,
        id2label={i: f"EDUCABILITY_{i+1}" for i in range(4)},
        label2id={f"EDUCABILITY_{i+1}": i for i in range(4)},
    )
    config.educability_max_length = 1024
    config.temperature_calibration = 1.138848624315927
    torch.manual_seed(123)
    ModernBertForSequenceClassification(config).save_pretrained(path)
    return path


@pytest.fixture
def make_job(tmp_path, tiny_model):
    count = 0

    def make(rows=None, **kwargs):
        nonlocal count
        count += 1
        source = tmp_path / f"input-{count}.jsonl"
        if rows is None:
            rows = [{"id": i, "text": "a b ciência " * (1 + i % 4)} for i in range(15)]
        source.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
        )
        output = tmp_path / f"output-{count}"
        manifest = prepare(
            [str(source)], output, "text", str(tiny_model), offline=True,
            jsonl_task_bytes=kwargs.pop("jsonl_task_bytes", 200), **kwargs,
        )
        return source, output, manifest

    return make


def run_args(output: Path, *extra):
    return parser().parse_args([
        "run", "--output", str(output), "--device", "cpu", "--offline",
        "--read-rows", "2", "--checkpoint-rows", "4", "--batch-size", "2", *extra,
    ])


def output_rows(output: Path):
    manifest = load_manifest(output)
    rows = []
    for task in manifest["tasks"]:
        sequence = 0
        while metadata_path(output, task, sequence).exists():
            rows.extend(iter_output(part_path(output, task, sequence), task["format"]))
            sequence += 1
    return rows
