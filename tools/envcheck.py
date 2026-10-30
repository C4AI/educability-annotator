"""Verificação de ambiente da imagem na DGX (PLANO_EDUCABILIDADE.md, FASE 1).

Grava versões e GPUs. Com GPU, compara o modelo publicado em FlashAttention-2 (BF16 e FP16) e em
SDPA BF16 com a referência em SDPA FP32, num lote com padding (textos de tamanhos diferentes). Mede a
maior diferença nas probabilidades calibradas e se a classe muda. A troca de classe aqui só informa;
o critério de precisão é da FASE 2 (2.5), com 10 mil documentos reais.

Uso: python tools/envcheck.py --model <pasta do modelo> --out envcheck.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PACKAGES = ("torch", "transformers", "tokenizers", "huggingface-hub", "flash_attn", "pyarrow", "numpy", "datasets")
TEXTS = [
    "A fotossíntese converte energia luminosa em energia química nas plantas.",
    "Clique aqui para acessar o menu de navegação e conferir as ofertas da semana.",
    "Para resolver uma equação do segundo grau, calcula-se primeiro o discriminante, igual a b² − 4ac. "
    "Se ele for positivo, há duas raízes reais distintas; se for zero, uma raiz dupla; se for negativo, "
    "não há raízes reais. As raízes saem da fórmula de Bhaskara.",
    "O Supremo Tribunal Federal julgou o recurso extraordinário e fixou a tese de repercussão geral. " * 40,
    "A história do Brasil colonial começa com a chegada dos portugueses em 1500. " * 150,
]


def versions() -> dict:
    found = {}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), **found,
            "torch_cuda": torch.version.cuda, "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI)}


def gpus() -> list[dict]:
    return [{"index": i, "name": torch.cuda.get_device_name(i),
             "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
             "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
            for i in range(torch.cuda.device_count())]


def probabilities(model_dir: str, attention: str, dtype: torch.dtype) -> tuple[torch.Tensor, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=dtype, attn_implementation=attention, use_safetensors=True,
    ).to("cuda").eval()
    inputs = tokenizer(TEXTS, padding=True, truncation=True, max_length=1024, return_tensors="pt",
                       return_token_type_ids=False).to("cuda")
    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits.float()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return torch.softmax(logits / model.config.temperature_calibration, dim=-1).cpu(), seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report: dict = {"versions": versions(), "cuda_available": torch.cuda.is_available()}
    failed = False
    if torch.cuda.is_available():
        report["gpus"] = gpus()
        try:
            import flash_attn  # noqa: F401  (a extensão CUDA só carrega com driver)
            report["flash_attn_import"] = "ok"
        except Exception as exc:  # o relatório registra a falha em vez de derrubar o check
            report["flash_attn_import"] = f"{type(exc).__name__}: {exc}"
            failed = True
        reference, _ = probabilities(args.model, "sdpa", torch.float32)
        report["parity"] = []
        variants = [("sdpa", torch.bfloat16)]
        if report["flash_attn_import"] == "ok":
            variants = [("flash_attention_2", torch.bfloat16), ("flash_attention_2", torch.float16)] + variants
        for attention, dtype in variants:
            try:
                actual, seconds = probabilities(args.model, attention, dtype)
                report["parity"].append({
                    "attention": attention, "dtype": str(dtype).removeprefix("torch."),
                    "max_abs_diff": round((actual - reference).abs().max().item(), 6),
                    "class_changes": int((actual.argmax(-1) != reference.argmax(-1)).sum()),
                    "seconds": round(seconds, 4),
                })
            except Exception as exc:
                report["parity"].append({"attention": attention, "dtype": str(dtype), "error": f"{type(exc).__name__}: {exc}"})
                failed = True
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
