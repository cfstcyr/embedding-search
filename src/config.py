import os
from typing import Literal


def _load_config_env(prefix: str) -> dict:
    return {
        key[len(prefix) :].lower(): value
        for key, value in os.environ.items()
        if key.startswith(prefix)
    }


def load_config(src: Literal["env"], *kwargs) -> dict:
    if src == "env":
        default_configs = _load_config_env("DEFAULT_")
        configs = _load_config_env(*kwargs)
        return {**default_configs, **configs}
    else:
        raise ValueError(f"Unsupported config source: {src}")
