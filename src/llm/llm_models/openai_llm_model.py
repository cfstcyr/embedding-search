from typing import Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from src.providers import OpenAIProvider

from .llm_model import LLMModel


class OpenAILLMModel(LLMModel[OpenAIProvider]):
    def _ask(
        self,
        question: str,
        *,
        previous_messages: list[ChatCompletionMessageParam] = [],
        context: Optional[str] = None,
    ) -> str:
        messages: list[ChatCompletionMessageParam] = previous_messages

        if context is not None:
            messages.append(
                ChatCompletionSystemMessageParam(
                    content=context,
                    role="system",
                    name="system",
                ),
            )

        messages.append(
            ChatCompletionUserMessageParam(
                content=question,
                role="user",
                name="user",
            ),
        )

        return (
            self.provider.client.chat.completions.create(
                model=self.provider.config.model,
                messages=messages,
            )
            .choices[0]
            .message.content
        )

    def _init_client(self) -> OpenAI:
        return OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
        )
