"""FAISS inner-product index for normalized embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.config import FAISS_INDEX_PATH, FAISS_META_PATH


class VectorStore:
    def __init__(self, index_path: Path | None = None, meta_path: Path | None = None) -> None:
        self.index_path = index_path or FAISS_INDEX_PATH
        self.meta_path = meta_path or FAISS_META_PATH
        self._index: faiss.Index | None = None
        self._id_map: list[int] = []

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() or build_from_embeddings().")
        return self._index

    def build_from_embeddings(self, embeddings: np.ndarray, ids: list[int]) -> None:
        """embeddings: (n, dim) float32, L2-normalized. ids parallel to rows."""
        if len(ids) != len(embeddings):
            raise ValueError("ids length must match embeddings rows")
        dim = embeddings.shape[1]
        x = np.ascontiguousarray(embeddings.astype("float32"))
        # Inner product on normalized vectors == cosine similarity
        idx = faiss.IndexFlatIP(dim)
        idx.add(x)
        self._index = idx
        self._id_map = list(ids)

    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("No index to save.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        meta = {"id_map": self._id_map, "dim": self._index.d}
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")

    def load(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self._id_map = [int(x) for x in meta["id_map"]]

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """query_embedding: (dim,) or (1, dim) float32 normalized."""
        q = np.ascontiguousarray(query_embedding.reshape(1, -1).astype("float32"))
        scores, idxs = self.index.search(q, min(top_k, len(self._id_map)))
        out: list[tuple[int, float]] = []
        for j, col in enumerate(idxs[0]):
            if col < 0:
                continue
            internal = int(col)
            if internal >= len(self._id_map):
                continue
            doc_id = self._id_map[internal]
            score = float(scores[0][j])
            out.append((doc_id, score))
        return out

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self._index is not None else 0
