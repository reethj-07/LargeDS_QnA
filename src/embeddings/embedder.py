"""Sentence-transformers embedding wrapper with batching."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBED_BATCH_SIZE, EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str | None = None, batch_size: int | None = None) -> None:
        self.model_name = model_name or EMBEDDING_MODEL
        self.batch_size = batch_size or EMBED_BATCH_SIZE
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """Return float32 array shape (n, dim)."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_batches(
        self,
        texts: list[str],
        chunk_size: int = 5000,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode in chunks to limit peak RAM."""
        parts: list[np.ndarray] = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            parts.append(self.encode(chunk, show_progress=show_progress))
            gc.collect()
        if not parts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        return np.vstack(parts)
