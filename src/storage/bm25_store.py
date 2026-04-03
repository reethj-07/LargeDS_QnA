"""BM25 keyword index using rank_bm25."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class BM25Store:
    def __init__(self, path: Path | None = None) -> None:
        from src.config import BM25_PATH

        self.path = path or BM25_PATH
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []
        self._id_map: list[int] = []

    def build(self, doc_texts: list[str], ids: list[int]) -> None:
        if len(doc_texts) != len(ids):
            raise ValueError("doc_texts and ids length mismatch")
        self._tokenized_corpus = [tokenize(t) for t in doc_texts]
        self._id_map = list(ids)
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def save(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("BM25 not built.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "id_map": self._id_map,
            "corpus": self._tokenized_corpus,
        }
        with self.path.open("wb") as f:
            pickle.dump(payload, f)

    def load(self) -> None:
        with self.path.open("rb") as f:
            payload = pickle.load(f)
        self._id_map = [int(x) for x in payload["id_map"]]
        self._tokenized_corpus = payload["corpus"]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("BM25 not loaded.")
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        scores = np.asarray(scores, dtype=np.float64)
        n = min(top_k, int(scores.size))
        if n <= 0:
            return []
        idxs = np_argsort_topk(scores, n)
        return [(self._id_map[i], float(scores[i])) for i in idxs]


def np_argsort_topk(scores: Any, k: int) -> list[int]:
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0 or k <= 0:
        return []
    k = min(k, len(s))
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [int(i) for i in idx.tolist()]
