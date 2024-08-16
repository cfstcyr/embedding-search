import numpy as np
from sentence_transformers import SentenceTransformer

from src.providers import HuggingFaceProvider

from .embedder import Embedder


class HuggingFaceEmbedder(Embedder[HuggingFaceProvider]):
    def _embed(self, input: list[str]) -> np.ndarray:
        return self.provider.client.encode(input)

    def _init_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._config.model)
