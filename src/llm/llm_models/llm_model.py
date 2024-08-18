import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from codetiming import Timer
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from src.providers import Provider

ProviderType = TypeVar("ProviderType", bound=Provider)

logger = logging.getLogger(__name__)


class LLMModel(ABC, Generic[ProviderType]):
    provider: ProviderType

    def __init__(self, provider: ProviderType):
        self.provider = provider

    @Timer(
        text="Ask time: {seconds:.2f} seconds",
        initial_text="Starting ask...",
        logger=logger.info,
    )
    def ask(
        self,
        question: str | ChatCompletionUserMessageParam,
        *,
        previous_messages: list[ChatCompletionMessageParam] = [],
    ) -> str:
        return self._ask(question, previous_messages=previous_messages)

    @abstractmethod
    def _ask(
        self,
        question: str | ChatCompletionUserMessageParam,
        *,
        previous_messages: list[ChatCompletionMessageParam] = [],
    ) -> str: ...
