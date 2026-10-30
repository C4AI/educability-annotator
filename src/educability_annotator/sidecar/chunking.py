"""Tokens de conteúdo por documento, tokenizando só o prefixo que o modelo vai ver.

O Classifier.prepare original tokeniza o documento inteiro e depois corta. Aqui o texto é cortado antes,
no último espaço em branco ASCII até `max_document_tokens × chars_per_token` caracteres. Os IDs saem
iguais aos do documento inteiro porque o pré-tokenizador byte-level separa as palavras nesse espaço e o
NFC não compõe caracteres através dele. Todo pré-token do prefixo é, portanto, um pré-token do texto
inteiro. Se o prefixo não render mais tokens do que o necessário, o documento é refeito com o texto
inteiro: só assim se sabe o total exato e o `truncated`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from tokenizers import Tokenizer

STATUSES = ("ok", "missing_text", "null_text", "empty_text", "non_string_text")
OK, MISSING_TEXT, NULL_TEXT, EMPTY_TEXT, NON_STRING_TEXT = range(len(STATUSES))
CUT_CHARACTERS = (" ", "\n", "\t", "\r")


@dataclass
class Tokenized:
    status: np.ndarray       # int8, um por documento, índice em STATUSES
    tokens_used: np.ndarray  # int32, tokens de conteúdo classificados (0 se inválido)
    truncated: np.ndarray    # bool, havia tokens além de max_document_tokens
    ids: list                # por documento: np.ndarray int32 com os tokens usados (vazio se inválido)
    retried: int             # documentos refeitos com o texto inteiro


class Chunker:
    def __init__(self, tokenizer_json: str, capacity: int, max_document_tokens: int, chars_per_token: int = 5):
        if capacity < 1 or max_document_tokens < 1 or chars_per_token < 1:
            raise ValueError("capacity, max_document_tokens e chars_per_token precisam ser positivos")
        self.tokenizer = Tokenizer.from_str(tokenizer_json)
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.capacity = capacity
        self.max_tokens = max_document_tokens
        self.prefix_chars = max_document_tokens * chars_per_token
        self.prefix_ids, self.suffix_ids = self._specials()
        encode = getattr(self.tokenizer, "encode_batch_fast", None) or self.tokenizer.encode_batch
        self._encode_batch = encode

    def _specials(self) -> tuple[list[int], list[int]]:
        """Tokens especiais em volta de um chunk, descobertos pelo pós-processador (como no model.py)."""
        content = self.tokenizer.encode("a", add_special_tokens=False)
        if len(content.ids) != 1:
            content.truncate(1)
        wrapped = self.tokenizer.post_process(content, add_special_tokens=True).ids
        position = wrapped.index(content.ids[0])
        return wrapped[:position], wrapped[position + 1:]

    @property
    def special_count(self) -> int:
        return len(self.prefix_ids) + len(self.suffix_ids)

    def cut(self, text: str, longer: bool) -> tuple[str, bool]:
        """Prefixo e se houve corte. `text` pode já vir fatiado em prefix_chars; `longer` diz se o original é maior."""
        if not longer and len(text) <= self.prefix_chars:
            return text, False
        end = max(text.rfind(character, 0, self.prefix_chars) for character in CUT_CHARACTERS)
        if end <= 0:
            return text, None  # sem espaço para cortar: o chamador refaz com o texto inteiro
        return text[:end], True

    def tokenize(self, texts: Sequence, longer: Sequence[bool] | None = None,
                 full_text: Callable[[int], str] | None = None) -> Tokenized:
        """`texts[i]` é o texto (ou o prefixo já fatiado em prefix_chars, com longer[i] verdadeiro se o
        original passava disso); `full_text(i)` devolve o texto inteiro para as refações."""
        count = len(texts)
        longer = longer if longer is not None else [False] * count
        status = np.zeros(count, dtype=np.int8)
        tokens_used = np.zeros(count, dtype=np.int32)
        truncated = np.zeros(count, dtype=bool)
        ids: list = [np.empty(0, dtype=np.int32)] * count
        batch_texts, batch_index, batch_cut = [], [], []
        retry: list[int] = []
        for index, text in enumerate(texts):
            if text is None:
                status[index] = NULL_TEXT
                continue
            if not isinstance(text, str):
                status[index] = NON_STRING_TEXT
                continue
            if not text.strip():
                if longer[index] and full_text is not None:
                    retry.append(index)  # prefixo só de espaços: decide no texto inteiro
                else:
                    status[index] = EMPTY_TEXT
                continue
            prefix, was_cut = self.cut(text, bool(longer[index]))
            if was_cut is None:
                retry.append(index)
                continue
            batch_texts.append(prefix)
            batch_index.append(index)
            batch_cut.append(was_cut)
        if batch_texts:
            for encoding, index, was_cut in zip(self._encode_batch(batch_texts, add_special_tokens=False),
                                                batch_index, batch_cut):
                available = encoding.ids
                if was_cut and len(available) <= self.max_tokens:
                    retry.append(index)  # prefixo curto demais para saber o total
                    continue
                self._store(index, available, status, tokens_used, truncated, ids)
        if retry:
            if full_text is None:
                raise ValueError("documentos precisam do texto inteiro, mas full_text não foi passado")
            whole = [full_text(index) for index in retry]
            valid = [(index, text) for index, text in zip(retry, whole) if text.strip()]
            for index in set(retry) - {index for index, _ in valid}:
                status[index] = EMPTY_TEXT
            if valid:
                for (index, _), encoding in zip(valid, self._encode_batch([text for _, text in valid],
                                                                          add_special_tokens=False)):
                    self._store(index, encoding.ids, status, tokens_used, truncated, ids)
        return Tokenized(status, tokens_used, truncated, ids, len(retry))

    def _store(self, index, available, status, tokens_used, truncated, ids) -> None:
        total = len(available)
        if total == 0:
            status[index] = EMPTY_TEXT
            return
        used = min(total, self.max_tokens)
        tokens_used[index] = used
        truncated[index] = total > used
        ids[index] = np.asarray(available[:used], dtype=np.int32)

    def chunk_count(self, tokens_used: np.ndarray) -> np.ndarray:
        return ((tokens_used.astype(np.int64) + self.capacity - 1) // self.capacity).astype(np.int16)
