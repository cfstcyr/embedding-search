import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from codetiming import Timer

from src.providers import Provider

ProviderType = TypeVar("ProviderType", bound=Provider)

logger = logging.getLogger(__name__)


class LLMModel(ABC, Generic[ProviderType]):
    provider: ProviderType

    def __init__(self, provider: ProviderType):
        self.provider = provider

    @Timer(
        text="LLM Model time: {seconds:.2f} seconds",
        initial_text="Starting LLM Model...",
        logger=logger.info,
    )
    def ask(self, question: str) -> str:
        return self._ask(question)

    @abstractmethod
    def _ask(self, question: str) -> str: ...
