import numpy as np

from src.providers import OpenAIProvider

from .embedder import Embedder


class OpenAIEmbedder(Embedder[OpenAIProvider]):
    def _embed(self, input: list[str]) -> np.ndarray:
        result = self.provider.client.embeddings.create(
            input=input,
            model=self.provider.config.model,
        )

        return np.array([item.embedding for item in result.data])
