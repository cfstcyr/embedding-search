from src.config import load_config
from src.providers.openai_provider import OpenAIProvider

from .llm_models import LLMModel, OpenAILLMModel


def llm_model_factory(type: str, config: dict) -> LLMModel:
    if type == "openai":
        return OpenAILLMModel(OpenAIProvider(config))
    else:
        raise ValueError(f"Unsupported LLM model type: {type}")


def llm_model_factory_env(prefix: str = "LLM_") -> LLMModel:
    config = load_config("env", prefix)
    return llm_model_factory(config["type"], config)
