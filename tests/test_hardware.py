import os

import pytest
import torch

from educability_annotator.common import MODEL_ID, MODEL_REVISION, AnnotationError
from educability_annotator.model import Classifier, inspect_model, resolve_device


def test_cuda_oom_retries_smaller_batches_without_losing_chunks(tiny_model, monkeypatch):
    classifier = Classifier(inspect_model(str(tiny_model)), 1022, "cpu", "float32", batch_size=8)
    prepared = classifier.prepare([{"text": "a"} for _ in range(7)], "text")
    classifier.device = torch.device("cuda:0")
    calls = []
    def forward(chunks):
        calls.append(len(chunks))
        if len(chunks) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return [[0, 0, 1, 0] for _ in chunks]
    monkeypatch.setattr(classifier, "_forward_once", forward)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    rows, _ = classifier.predict(prepared)
    assert len(rows) == 7
    assert all(row["class"] == 3 for row in rows)
    assert calls[0] == 7 and max(calls[1:]) < 7
    assert classifier.batch_size <= 2


def test_single_chunk_oom_is_explicit(tiny_model, monkeypatch):
    classifier = Classifier(inspect_model(str(tiny_model)), 1022, "cpu", "float32")
    classifier.device = torch.device("cuda:0")
    def forward(_):
        raise torch.cuda.OutOfMemoryError("simulated")
    monkeypatch.setattr(classifier, "_forward_once", forward)
    with pytest.raises(AnnotationError, match="único chunk"):
        classifier.predict(classifier.prepare([{"text": "a"}], "text"))


@pytest.mark.hub
@pytest.mark.skipif(os.environ.get("EDUCABILITY_TEST_HUB") != "1", reason="ative EDUCABILITY_TEST_HUB=1")
def test_published_model_against_reference():
    torch.set_num_threads(2)
    model_id = os.environ.get("EDUCABILITY_TEST_MODEL", MODEL_ID)
    spec = inspect_model(model_id, MODEL_REVISION)
    classifier = Classifier(spec, 1022, "cpu", "float32", batch_size=2)
    texts = [
        "A fotossíntese converte energia luminosa em energia química nas plantas.",
        "Clique aqui para acessar o menu de navegação.",
        "Este tutorial explica como resolver uma equação do segundo grau.",
        "educação e ciência " * 600,
    ]
    actual, _ = classifier.predict(classifier.prepare([{"text": text} for text in texts], "text"))
    inputs = classifier.tokenizer(
        texts, padding=True, truncation=True, max_length=1024,
        return_tensors="pt", return_token_type_ids=False,
    )
    with torch.inference_mode():
        expected = torch.softmax(classifier.model(**inputs).logits.float() / spec["temperature"], dim=-1)
    actual_tensor = torch.tensor([[row[f"p{i}"] for i in range(1, 5)] for row in actual])
    torch.testing.assert_close(actual_tensor, expected, atol=1e-6, rtol=1e-5)
    print(f"published_reference_max_abs_difference={(actual_tensor - expected).abs().max().item():.9g}")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU NVIDIA indisponível")
def test_real_cuda_matches_cpu(tiny_model):
    spec = inspect_model(str(tiny_model))
    inputs = [{"text": "a b ciência"}, {"text": "a " * 1300}]
    cpu = Classifier(spec, 5000, "cpu", "float32")
    gpu = Classifier(spec, 5000, "cuda:0", "float32")
    expected, _ = cpu.predict(cpu.prepare(inputs, "text"))
    actual, _ = gpu.predict(gpu.prepare(inputs, "text"))
    for a, b in zip(actual, expected, strict=True):
        assert [a[f"p{i}"] for i in range(1, 5)] == pytest.approx(
            [b[f"p{i}"] for i in range(1, 5)], abs=1e-4,
        )
