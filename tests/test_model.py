import math

import pytest
import torch
from transformers import AutoConfig

from educability_annotator.common import AnnotationError
from educability_annotator.model import Classifier, inspect_model, validate_config


@pytest.fixture
def classifier(tiny_model):
    return Classifier(inspect_model(str(tiny_model)), 5000, "cpu", "float32", batch_size=3)


@pytest.mark.parametrize("length", [1, 1021, 1022, 1023, 2044, 5000, 5200])
def test_chunks_preserve_tokens_and_specials(classifier, length):
    text = " ".join(["a", "b", "c"][i % 3] for i in range(length))
    prepared = classifier.prepare([{"text": text}], "text")
    chunks = prepared.chunks
    content = [token for chunk in chunks for token in chunk.inputs["input_ids"][1:-1]]
    expected = classifier.backend.encode(text, add_special_tokens=False).ids[:5000]
    assert content == expected
    assert sum(chunk.weight for chunk in chunks) == min(length, 5000)
    assert len(chunks) == math.ceil(min(length, 5000) / 1022)
    assert prepared.results[0]["truncated"] == (length > 5000)
    assert all(chunk.inputs["input_ids"][0] == 2 and chunk.inputs["input_ids"][-1] == 3 for chunk in chunks)
    assert all(len(chunk.inputs["input_ids"]) <= 1024 for chunk in chunks)


def test_default_chunk_matches_native_formula(tiny_model):
    spec = inspect_model(str(tiny_model))
    classifier = Classifier(spec, 1022, "cpu", "float32")
    texts = ["a ciência", "b educação " * 700]
    actual, _ = classifier.predict(classifier.prepare([{"text": text} for text in texts], "text"))
    inputs = classifier.tokenizer(
        texts, padding=True, truncation=True, max_length=1024,
        return_tensors="pt", return_token_type_ids=False,
    )
    with torch.inference_mode():
        logits = classifier.model(**inputs).logits.float()
        expected = torch.softmax(logits / spec["temperature"], dim=-1)
    torch.testing.assert_close(
        torch.tensor([[r[f"p{i}"] for i in range(1, 5)] for r in actual]), expected, atol=1e-6, rtol=1e-6,
    )


def test_weighted_probabilities_and_tie_rule(classifier, monkeypatch):
    prepared = classifier.prepare([{"text": "a " * 1500}, {"text": "b"}], "text")
    def known(chunks):
        return [
            ([0.8, 0.1, 0.1, 0.0] if c.index == 0 else [0.0, 0.0, 0.0, 1.0])
            if c.document == 0 else [0.5, 0.5, 0.0, 0.0]
            for c in chunks
        ]
    monkeypatch.setattr(classifier, "_forward", known)
    rows, _ = classifier.predict(prepared)
    expected = [(1022 * x + 478 * y) / 1500 for x, y in zip([.8, .1, .1, 0], [0, 0, 0, 1])]
    assert [rows[0][f"p{i}"] for i in range(1, 5)] == pytest.approx(expected)
    assert rows[0]["expected_score"] == pytest.approx(sum((i + 1) * p for i, p in enumerate(expected)))
    assert rows[1]["class"] == 1


def test_invalid_texts(classifier):
    rows = [{}, {"text": None}, {"text": ""}, {"text": " \n"}, {"text": 2}, {"text": []}]
    prepared = classifier.prepare(rows, "text")
    actual, _ = classifier.predict(prepared)
    assert not prepared.chunks
    assert [row["status"] for row in actual] == [
        "missing_text", "null_text", "empty_text", "empty_text", "non_string_text", "non_string_text",
    ]
    assert all(row["class"] is None for row in actual)


@pytest.mark.parametrize("field,value", [
    ("temperature_calibration", 0), ("temperature_calibration", float("nan")),
    ("educability_max_length", 8192), ("model_type", "bert"),
    ("label2id", {"EDUCABILITY_1": 3}),
])
def test_invalid_config(tiny_model, field, value):
    config = AutoConfig.from_pretrained(tiny_model)
    setattr(config, field, value)
    with pytest.raises(AnnotationError):
        validate_config(config)


def test_batch_limits_and_association(classifier, monkeypatch):
    classifier.batch_size = 2
    classifier.max_batch_tokens = 1024
    prepared = classifier.prepare(
        [{"text": "a " * length} for length in [1000, 1, 500, 2, 1022]], "text",
    )
    original = classifier._forward
    batches = []
    def forward(chunks):
        assert len(chunks) <= 2
        assert len(chunks) * max(len(c.inputs["input_ids"]) for c in chunks) <= 1024
        batches.append([c.document for c in chunks])
        return original(chunks)
    monkeypatch.setattr(classifier, "_forward", forward)
    actual, _ = classifier.predict(prepared)
    assert [row["tokens_used"] for row in actual] == [1000, 1, 500, 2, 1022]
    assert batches[0] == [1, 3]


def test_probability_validation(classifier, monkeypatch):
    def bad(**_):
        return type("Output", (), {"logits": torch.full((1, 4), float("nan"))})()
    monkeypatch.setattr(classifier.model, "forward", bad)
    with pytest.raises(AnnotationError, match="não finitas"):
        classifier.predict(classifier.prepare([{"text": "a"}], "text"))
