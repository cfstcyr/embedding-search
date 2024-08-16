import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from codetiming import Timer

from src.providers import Provider

EmbedInput = str | list | pd.Series
ProviderType = TypeVar("ProviderType", bound=Provider)

logger = logging.getLogger(__name__)


class Embedder(ABC, Generic[ProviderType]):
    provider: ProviderType

    def __init__(self, provider: ProviderType):
        self.provider = provider

    @Timer(
        text="Embedding time: {seconds:.2f} seconds",
        initial_text="Starting embedding...",
        logger=logger.info,
    )
    def embed(self, input: EmbedInput) -> np.ndarray:
        return self._embed(self._to_list(input))

    def embed_one(self, input: str) -> np.ndarray:
        return self.embed(input)[0]

    @abstractmethod
    def _embed(self, input: list[str]) -> np.ndarray: ...

    def _to_list(self, data: EmbedInput) -> list:
        if isinstance(data, str):
            return [data]
        if isinstance(data, pd.Series):
            return data.tolist()
        return data
