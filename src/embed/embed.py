from src.config import load_config
from src.providers import HuggingFaceProvider, OpenAIProvider

from .embedder import Embedder, HuggingFaceEmbedder, OpenAIEmbedder


def embedder_factory(type: str, config: dict) -> Embedder:
    if type == "openai":
        return OpenAIEmbedder(OpenAIProvider(config))
    if type == "huggingface":
        return HuggingFaceEmbedder(HuggingFaceProvider(config))
    else:
        raise ValueError(f"Unsupported embedder type: {type}")


def embedder_factory_env(prefix: str = "EMBEDDER_") -> Embedder:
    config = load_config("env", prefix)
    return embedder_factory(config["type"], config)
