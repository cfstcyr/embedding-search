from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from .provider import Provider


class OpenAIProviderConfig(BaseModel):
    base_url: Optional[str] = Field(None)
    api_key: str
    model: str


class OpenAIProvider(Provider[OpenAIProviderConfig, OpenAI]):
    _configType = OpenAIProviderConfig

    def _init_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
