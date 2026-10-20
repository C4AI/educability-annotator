"""Native ModernBERT inference, token-preserving chunks, and calibrated pooling."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .common import (
    MODEL_ID, MODEL_REVISION, AnnotationError, digest, invalid_result, sha256,
)


def local_model_digest(path: Path) -> str:
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors"})
    if not files:
        raise AnnotationError(f"Modelo local sem arquivos JSON/safetensors: {path}")
    return digest({str(p.relative_to(path)): sha256(p) for p in files})


def validate_config(config) -> dict:
    try:
        temperature = float(config.temperature_calibration)
        limit = int(config.educability_max_length)
        labels = {int(i): label for i, label in config.id2label.items()}
        expected = {f"EDUCABILITY_{i}" for i in range(1, 5)}
        if config.model_type != "modernbert" or config.num_labels != 4:
            raise ValueError("é necessário ModernBERT com quatro classes")
        if set(labels) != set(range(4)) or set(labels.values()) != expected:
            raise ValueError("id2label incompatível com as classes 1–4")
        if any(config.label2id[label] != index for index, label in labels.items()):
            raise ValueError("label2id e id2label divergem")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature_calibration deve ser finita e positiva")
        if limit < 3 or limit > 1024 or limit > config.max_position_embeddings:
            raise ValueError("educability_max_length fora do limite suportado de 1.024")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise AnnotationError(f"Configuração do modelo inválida: {exc}") from exc
    return {
        "temperature": temperature,
        "chunk_limit": limit,
        "score_indices": [config.label2id[f"EDUCABILITY_{i}"] for i in range(1, 5)],
    }


def inspect_model(model_id: str = MODEL_ID, revision: str | None = None, offline: bool = False) -> dict:
    local = Path(model_id).is_dir()
    if local:
        model_id = str(Path(model_id).resolve())
        revision = None
    else:
        revision = revision or (MODEL_REVISION if model_id == MODEL_ID else "main")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            if offline:
                raise AnnotationError("Em modo offline, forneça uma revisão SHA completa.")
            revision = HfApi().model_info(model_id, revision=revision).sha
    options = {"revision": revision, "local_files_only": offline, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(model_id, **options)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    settings = validate_config(config)
    if not tokenizer.is_fast:
        raise AnnotationError("O modelo precisa de um tokenizer fast com pós-processador.")
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    if specials < 1 or specials >= settings["chunk_limit"] or tokenizer.pad_token_id is None:
        raise AnnotationError("Tokens especiais/padding incompatíveis.")
    if tokenizer.model_max_length < settings["chunk_limit"]:
        raise AnnotationError("Limite do tokenizer menor que educability_max_length.")
    return {
        "id": model_id, "revision": revision,
        "local_digest": local_model_digest(Path(model_id)) if local else None,
        "tokenizer_digest": digest(tokenizer.backend_tokenizer.to_str()),
        "special_tokens": specials, **settings,
    }


def check_local_model(spec: dict) -> None:
    if spec["local_digest"] is not None and local_model_digest(Path(spec["id"])) != spec["local_digest"]:
        raise AnnotationError("Modelo/tokenizer local alterado; prepare outra saída.")


def resolve_device(device: str, dtype: str, local_rank: int = 0) -> tuple[str, str]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise AnnotationError(f"GPU CUDA local {local_rank} indisponível.")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else "float32"
    if dtype == "bfloat16" and (device == "cpu" or not torch.cuda.is_bf16_supported()):
        raise AnnotationError("BF16 requer GPU CUDA compatível; use float32.")
    return device, dtype


@dataclass
class Chunk:
    document: int
    index: int
    weight: int
    inputs: dict


@dataclass
class Prepared:
    results: list[dict]
    chunks: list[Chunk]
    seconds: float


class Classifier:
    def __init__(
        self, spec: dict, max_document_tokens: int, device: str, dtype: str,
        batch_size: int = 32, max_batch_tokens: int = 8192, offline: bool = False,
        attention: str = "sdpa",
    ):
        self.spec = spec
        self.max_document_tokens = max_document_tokens
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.device = torch.device(device)
        options = {
            "revision": spec["revision"], "local_files_only": offline,
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(spec["id"], **options)
        if digest(self.tokenizer.backend_tokenizer.to_str()) != spec["tokenizer_digest"]:
            raise AnnotationError("Tokenizer diverge do manifesto.")
        self.backend = Tokenizer.from_str(self.tokenizer.backend_tokenizer.to_str())
        self.backend.no_padding()
        self.backend.no_truncation()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            spec["id"], torch_dtype=getattr(torch, dtype), use_safetensors=True,
            attn_implementation=attention, **options,
        ).to(self.device).eval()
        current = validate_config(self.model.config)
        if any(current[key] != spec[key] for key in current):
            raise AnnotationError("Configuração carregada diverge do manifesto.")
        self.capacity = spec["chunk_limit"] - spec["special_tokens"]
        if max_batch_tokens < spec["chunk_limit"]:
            raise AnnotationError("--max-batch-tokens deve comportar um chunk completo.")

    def prepare(self, rows: list[dict], text_column: str) -> Prepared:
        start = time.monotonic()
        results, chunks = [], []
        for document, row in enumerate(rows):
            text = row.get(text_column)
            status = (
                "missing_text" if text_column not in row else
                "null_text" if text is None else
                "non_string_text" if not isinstance(text, str) else
                "empty_text" if not text.strip() else None
            )
            if status:
                results.append(invalid_result(status))
                continue
            encoding = self.backend.encode(text, add_special_tokens=False)
            total = len(encoding.ids)
            if total == 0:
                results.append(invalid_result("empty_text"))
                continue
            used = min(total, self.max_document_tokens)
            encoding.truncate(self.capacity, stride=0)
            pieces = [encoding, *encoding.overflowing]
            remaining = used
            count = 0
            for piece in pieces:
                if remaining == 0:
                    break
                weight = min(len(piece.ids), remaining)
                if weight < len(piece.ids):
                    piece.truncate(weight, stride=0)
                processed = self.backend.post_process(piece, add_special_tokens=True)
                inputs = {"input_ids": processed.ids, "attention_mask": processed.attention_mask}
                if len(processed.ids) != weight + self.spec["special_tokens"]:
                    raise AnnotationError("Pós-processamento alterou a quantidade esperada de tokens.")
                chunks.append(Chunk(document, count, weight, inputs))
                remaining -= weight
                count += 1
            results.append({
                "tokens_used": used, "chunk_count": count, "truncated": total > used, "status": "ok",
            })
        return Prepared(results, chunks, time.monotonic() - start)

    def _forward_once(self, chunks: list[Chunk]) -> list[list[float]]:
        inputs = self.tokenizer.pad(
            [chunk.inputs for chunk in chunks], padding=True, return_tensors="pt",
        )
        if self.device.type == "cuda":
            inputs = {key: value.pin_memory().to(self.device, non_blocking=True)
                      for key, value in inputs.items()}
        else:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits.float()
            probs = torch.softmax(logits / self.spec["temperature"], dim=-1)
            probs = probs[:, self.spec["score_indices"]]
            if not torch.isfinite(probs).all().item():
                raise AnnotationError("Inferência produziu probabilidades não finitas.")
            return probs.cpu().tolist()

    def _forward(self, chunks: list[Chunk]) -> list[list[float]]:
        try:
            return self._forward_once(chunks)
        except torch.cuda.OutOfMemoryError as exc:
            if self.device.type != "cuda":
                raise
            if len(chunks) == 1:
                raise AnnotationError("Um único chunk excedeu a memória CUDA disponível.") from exc
            # Release batch tensors before recursively retrying smaller batches.
            exc.__traceback__ = None
            torch.cuda.empty_cache()
        middle = len(chunks) // 2
        self.batch_size = min(self.batch_size, middle)
        return self._forward(chunks[:middle]) + self._forward(chunks[middle:])

    def predict(self, prepared: Prepared) -> tuple[list[dict], float]:
        start = time.monotonic()
        ordered = sorted(prepared.chunks, key=lambda chunk: len(chunk.inputs["input_ids"]))
        predictions = {}
        position = 0
        while position < len(ordered):
            end = position
            while end < len(ordered) and end - position < self.batch_size:
                length = len(ordered[end].inputs["input_ids"])
                if (end - position + 1) * length > self.max_batch_tokens:
                    break
                end += 1
            batch = ordered[position:end]
            if not batch:
                raise AnnotationError("Limite de tokens do lote insuficiente.")
            for chunk, probs in zip(batch, self._forward(batch), strict=True):
                predictions[(chunk.document, chunk.index)] = probs
            position = end
        # Fixed chunk order and double-precision accumulation avoid scheduling-dependent pooling.
        sums = [[0.0] * 4 for _ in prepared.results]
        for chunk in prepared.chunks:
            for i, probability in enumerate(predictions[(chunk.document, chunk.index)]):
                sums[chunk.document][i] += chunk.weight * probability
        for document, result in enumerate(prepared.results):
            if result["status"] != "ok":
                continue
            probs = [value / result["tokens_used"] for value in sums[document]]
            normalizer = sum(probs)
            probs = [p / normalizer for p in probs]
            score = max(range(4), key=probs.__getitem__) + 1
            result.update({
                "class": score, "label": f"EDUCABILITY_{score}",
                **{f"p{i + 1}": p for i, p in enumerate(probs)},
                "expected_score": sum((i + 1) * p for i, p in enumerate(probs)),
            })
        return prepared.results, time.monotonic() - start
