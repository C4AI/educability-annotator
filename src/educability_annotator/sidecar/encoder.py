"""Inferência dos lotes empacotados e agregação por documento.

Dois caminhos dão as mesmas probabilidades, dentro da precisão escolhida:
- flash_attention_2: chama embeddings → camadas → norma final do ModernBERT direto sobre os tokens
  concatenados, com cu_seqlens. É o caminho desempacotado do próprio Transformers, com janela local
  real. Nenhum token de padding é calculado.
- sdpa/eager: monta um lote com padding e chama o modelo do Transformers; é o caminho de CPU e a
  alternativa se o FlashAttention-2 faltar.

Pooling (médio, o do modelo publicado, ou CLS), cabeça e softmax/T rodam em FP32 nos dois, como a
calibração pede. A agregação é em FP64, como as somas em float do Python no Classifier original.
"""

from __future__ import annotations

import copy

import torch
from transformers import AutoModelForSequenceClassification

from ..common import AnnotationError
from ..model import validate_config


class Encoder:
    def __init__(self, spec: dict, device: str, dtype: str, attention: str, offline: bool = True):
        if attention == "flash_attention_2" and not device.startswith("cuda"):
            raise AnnotationError("flash_attention_2 exige GPU CUDA.")
        self.spec = spec
        self.device = torch.device(device)
        self.attention = attention
        model = AutoModelForSequenceClassification.from_pretrained(
            spec["id"], revision=spec["revision"], local_files_only=offline, trust_remote_code=False,
            torch_dtype=getattr(torch, dtype), use_safetensors=True, attn_implementation=attention,
        )
        current = validate_config(model.config)
        if any(current[key] != spec[key] for key in current):
            raise AnnotationError("Configuração carregada diverge do manifesto.")
        if model.config.classifier_pooling not in ("mean", "cls"):
            raise AnnotationError(f"Pooling {model.config.classifier_pooling!r} não suportado.")
        self.pooling = model.config.classifier_pooling
        self.model = model.to(self.device).eval()
        self.head = copy.deepcopy(model.head).float().to(self.device).eval()
        self.classifier = copy.deepcopy(model.classifier).float().to(self.device).eval()
        self.temperature = float(spec["temperature"])
        self.score_indices = torch.tensor(spec["score_indices"], device=self.device)
        self.scores = torch.arange(1, 5, device=self.device, dtype=torch.float64)

    @torch.inference_mode()
    def chunk_probabilities(self, input_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        """Probabilidades calibradas por chunk, na ordem das classes 1–4, em FP32."""
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if self.attention == "flash_attention_2":
            pooled = self._pooled_unpadded(input_ids, cu_seqlens, lengths, max_seqlen)
        else:
            pooled = self._pooled_padded(input_ids, cu_seqlens, lengths, max_seqlen)
        logits = self.classifier(self.head(pooled)).float()
        probs = torch.softmax(logits / self.temperature, dim=-1)[:, self.score_indices]
        if not torch.isfinite(probs).all():
            raise AnnotationError("Inferência produziu probabilidades não finitas.")
        return probs

    def _pooled_unpadded(self, input_ids, cu_seqlens, lengths, max_seqlen) -> torch.Tensor:
        backbone = self.model.model
        hidden = backbone.embeddings(input_ids=input_ids)
        for layer in backbone.layers:
            hidden = layer(hidden, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)[0]
        hidden = backbone.final_norm(hidden)
        if self.pooling == "cls":
            return hidden[cu_seqlens[:-1].long()].float()
        segment = torch.repeat_interleave(torch.arange(len(lengths), device=hidden.device), lengths)
        sums = torch.zeros(len(lengths), hidden.shape[-1], device=hidden.device, dtype=torch.float32)
        sums.index_add_(0, segment, hidden.float())
        return sums / lengths.unsqueeze(-1).float()

    def _pooled_padded(self, input_ids, cu_seqlens, lengths, max_seqlen, padded_budget: int = 65536) -> torch.Tensor:
        """Sublotes ordenados por tamanho, cada um com padding só até o próprio maior chunk."""
        pooled = torch.empty(len(lengths), self.model.config.hidden_size, device=input_ids.device)
        if not len(lengths):
            return pooled
        order = torch.argsort(lengths).tolist()
        sizes = lengths.tolist()
        starts = cu_seqlens.tolist()
        group: list[int] = []
        for index in order + [None]:
            if index is not None and (not group or (len(group) + 1) * sizes[index] <= padded_budget):
                group.append(index)
                continue
            longest = sizes[group[-1]]
            batch = torch.full((len(group), longest), int(self.model.config.pad_token_id),
                               device=input_ids.device, dtype=torch.long)
            mask = torch.zeros((len(group), longest), device=input_ids.device, dtype=torch.long)
            for row, chunk in enumerate(group):
                batch[row, :sizes[chunk]] = input_ids[starts[chunk]:starts[chunk + 1]]
                mask[row, :sizes[chunk]] = 1
            hidden = self.model.model(input_ids=batch, attention_mask=mask).last_hidden_state.float()
            if self.pooling == "cls":
                values = hidden[:, 0]
            else:
                values = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            pooled[torch.tensor(group, device=input_ids.device)] = values
            group = [index] if index is not None else []
        return pooled

    @torch.inference_mode()
    def aggregate(self, probs: torch.Tensor, chunk_doc: torch.Tensor, chunk_weight: torch.Tensor,
                  chunk_first: torch.Tensor, documents: int) -> dict[str, torch.Tensor]:
        """Por documento: probabilidades médias ponderadas por tokens (renormalizadas), classe (empate →
        menor), score esperado, maior score esperado entre os chunks e o do 1º chunk."""
        probs64 = probs.double()
        weight = chunk_weight.double()
        sums = torch.zeros(documents, 4, device=probs.device, dtype=torch.float64)
        sums.index_add_(0, chunk_doc, probs64 * weight.unsqueeze(-1))
        totals = torch.zeros(documents, device=probs.device, dtype=torch.float64).index_add_(0, chunk_doc, weight)
        mean = sums / totals.clamp(min=1).unsqueeze(-1)
        mean = mean / mean.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float64).tiny)
        chunk_score = probs64 @ self.scores
        maximum = torch.full((documents,), -torch.inf, device=probs.device, dtype=torch.float64)
        maximum.scatter_reduce_(0, chunk_doc, chunk_score, reduce="amax")
        first = torch.zeros(documents, device=probs.device, dtype=torch.float64)
        first.index_add_(0, chunk_doc[chunk_first], chunk_score[chunk_first])
        return {"probs": mean, "class": mean.argmax(dim=-1) + 1, "expected_score": mean @ self.scores,
                "max_chunk_score": maximum, "first_chunk_score": first}
