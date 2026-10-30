"""Micro-benchmark e paridade em GPU (PLANO_EDUCABILIDADE.md, 2.5 e 2.6), sobre a amostra do piloto.

1. Sorteia até N documentos por fonte da amostra, tokeniza com o Chunker (até 5 chunks) e empacota
   em lotes. Isso é feito uma vez, em CPU.
2. Calcula a referência: probabilidades por chunk em FP32 pelo caminho com padding (SDPA).
3. Mede cada configuração (atenção × precisão × orçamento de tokens): tempo de GPU depois do
   aquecimento, tokens/s, memória de pico e diferença contra a referência. A diferença inclui max |Δp|,
   trocas de classe por documento e |Δ score esperado| (p50, p99, máx.).
4. Mede a linha de base, o Classifier original com o lote padrão (SDPA, BF16, 32 chunks/8.192 tokens),
   num subconjunto:
   - ponta a ponta (prepare + predict numa thread, como cada GPU roda hoje);
   - só a GPU;
   - e a mesma conta em FP32, que tem de bater com a referência do motor.

Uso: python tools/bench_gpu.py --sample /pilot/sample --model <pasta> --out bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from educability_annotator.model import Classifier, inspect_model
from educability_annotator.sidecar.chunking import OK, Chunker
from educability_annotator.sidecar.encoder import Encoder
from educability_annotator.sidecar.packing import pack

CONFIGS = [
    ("sdpa", "bfloat16", 65536),
    ("flash_attention_2", "bfloat16", 65536),
    ("flash_attention_2", "float16", 65536),
    ("flash_attention_2", "float32", 65536),
    ("flash_attention_2", "bfloat16", 131072),
    ("flash_attention_2", "float16", 131072),
]


def load_documents(sample: Path, per_source: int, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    texts, sources = [], []
    for path in sorted(sample.glob("source=*/part-*.parquet")):
        table = pq.read_table(path, columns=["text"])
        rows = rng.choice(table.num_rows, size=min(per_source, table.num_rows), replace=False)
        texts += table.column(0).take(np.sort(rows)).to_pylist()
        sources += [path.parent.name.split("=", 1)[1]] * len(rows)
    return texts, sources


def batches_for(chunker: Chunker, tokenized, budget: int) -> list[dict]:
    return list(pack(chunker, tokenized, np.arange(len(tokenized.status)), budget))


def run_config(encoder: Encoder, batches: list[dict], device: str, warmup: int = 3):
    gpu = []
    for batch in batches:
        gpu.append({key: torch.as_tensor(batch[key]).to(device) for key in
                    ("input_ids", "cu_seqlens", "chunk_doc", "chunk_weight", "chunk_first")} |
                   {"max_seqlen": batch["max_seqlen"], "documents": len(batch["doc_id"])})
    usable = [batch for batch in gpu if len(batch["cu_seqlens"]) > 1]
    for batch in usable[:warmup]:
        encoder.chunk_probabilities(batch["input_ids"], batch["cu_seqlens"], batch["max_seqlen"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    chunk_probs, documents = [], []
    start = time.perf_counter()
    for batch in gpu:
        if len(batch["cu_seqlens"]) == 1:  # só documentos inválidos: zeros mantêm o alinhamento
            count = batch["documents"]
            documents.append({"probs": torch.zeros(count, 4, dtype=torch.float64, device=device),
                              "class": torch.ones(count, dtype=torch.long, device=device),
                              "expected_score": torch.zeros(count, dtype=torch.float64, device=device)})
            continue
        probs = encoder.chunk_probabilities(batch["input_ids"], batch["cu_seqlens"], batch["max_seqlen"])
        documents.append(encoder.aggregate(probs, batch["chunk_doc"], batch["chunk_weight"], batch["chunk_first"],
                                           batch["documents"]))
        chunk_probs.append(probs)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    tokens = sum(len(batch["input_ids"]) for batch in batches)
    merged = {key: torch.cat([doc[key] for doc in documents]).cpu().numpy()
              for key in ("probs", "class", "expected_score")}
    return {"seconds": seconds, "tokens": tokens, "tokens_per_second": tokens / seconds,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30, "documents": merged}


def compare(result: dict, reference: dict, valid: np.ndarray) -> dict:
    got, ref = result["documents"], reference["documents"]
    delta = np.abs(got["expected_score"] - ref["expected_score"])[valid]
    return {"max_abs_prob": float(np.abs(got["probs"] - ref["probs"])[valid].max()),
            "class_changes": int((got["class"] != ref["class"])[valid].sum()),
            "class_change_rate": float((got["class"] != ref["class"])[valid].mean()),
            "expected_score_abs_p50": float(np.percentile(delta, 50)),
            "expected_score_abs_p99": float(np.percentile(delta, 99)),
            "expected_score_abs_max": float(delta.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-source", type=int, default=2000)
    parser.add_argument("--baseline-docs", type=int, default=5000)
    parser.add_argument("--max-document-tokens", type=int, default=5110)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = "cuda:0"
    spec = inspect_model(args.model, offline=True)
    capacity = spec["chunk_limit"] - spec["special_tokens"]
    report: dict = {"gpu": torch.cuda.get_device_name(0), "max_document_tokens": args.max_document_tokens}

    texts, sources = load_documents(Path(args.sample), args.per_source, args.seed)
    tokenizer = Classifier(spec, args.max_document_tokens, "cpu", "float32").backend.to_str()
    chunker = Chunker(tokenizer, capacity, args.max_document_tokens, 5)
    start = time.perf_counter()
    tokenized = chunker.tokenize(texts, full_text=lambda index: texts[index])
    report["tokenize"] = {"documents": len(texts), "seconds": round(time.perf_counter() - start, 2),
                          "retried": tokenized.retried, "tokens": int(tokenized.tokens_used.sum())}
    valid = tokenized.status == OK
    print(json.dumps(report), flush=True)

    reference_batches = batches_for(chunker, tokenized, 16384)
    reference = run_config(Encoder(spec, device, "float32", "sdpa"), reference_batches, device)
    report["reference"] = {key: reference[key] for key in ("seconds", "tokens_per_second", "peak_memory_gib")}
    report["configs"] = []
    for attention, dtype, budget in CONFIGS:
        torch.cuda.empty_cache()
        result = run_config(Encoder(spec, device, dtype, attention), batches_for(chunker, tokenized, budget), device)
        entry = {"attention": attention, "dtype": dtype, "batch_tokens": budget,
                 **{key: result[key] for key in ("seconds", "tokens_per_second", "peak_memory_gib")},
                 **compare(result, reference, valid)}
        report["configs"].append(entry)
        print(json.dumps(entry), flush=True)

    # Linha de base: a ferramenta original, como roda hoje (uma thread prepara, a principal infere).
    subset = texts[:args.baseline_docs]
    for dtype in ("bfloat16", "float32"):
        classifier = Classifier(spec, args.max_document_tokens, device, dtype, batch_size=32, max_batch_tokens=8192)
        start = time.perf_counter()
        prepared = classifier.prepare([{"text": text} for text in subset], "text")
        prepare_seconds = time.perf_counter() - start
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows, _ = classifier.predict(prepared)
        torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - start
        tokens = sum(len(chunk.inputs["input_ids"]) for chunk in prepared.chunks)
        probs = np.array([[row[f"p{i}"] for i in range(1, 5)] for row in rows if row["status"] == "ok"])
        ok = valid[:len(subset)]
        engine = reference["documents"]["probs"][:len(subset)][ok]
        report[f"original_{dtype}"] = {
            "documents": len(subset), "tokens": tokens, "prepare_seconds": prepare_seconds,
            "predict_seconds": predict_seconds, "gpu_tokens_per_second": tokens / predict_seconds,
            "end_to_end_documents_per_second": len(subset) / (prepare_seconds + predict_seconds),
            "max_abs_prob_vs_engine_fp32": float(np.abs(probs - engine).max()),
        }
        print(json.dumps({f"original_{dtype}": report[f"original_{dtype}"]}), flush=True)

    by_source = {}
    for source in sorted(set(sources)):
        mask = np.array([item == source for item in sources]) & valid
        by_source[source] = {"documents": int(mask.sum()), "mean_tokens": float(tokenized.tokens_used[mask].mean())}
    report["sources"] = by_source
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
