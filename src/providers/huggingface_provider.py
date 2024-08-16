from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .provider import Provider


class HuggingFaceProviderConfig(BaseModel):
    model: str


class HuggingFaceProvider(Provider[HuggingFaceProviderConfig, SentenceTransformer]):
    _configType = HuggingFaceProviderConfig

    def _init_client(self) -> SentenceTransformer:
        return SentenceTransformer(self.config.model)
