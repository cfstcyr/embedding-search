import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

ConfigType = TypeVar("ConfigType", bound=BaseModel)
ClientType = TypeVar("ClientType")

logger = logging.getLogger(__name__)


class Provider(ABC, Generic[ConfigType, ClientType]):
    _configType: type[ConfigType]
    config: ConfigType
    client: ClientType

    def __init__(self, config: ConfigType | dict):
        self.config = type(self)._configType.model_validate(config)
        self.client = self._init_client()
        logger.info(f"Initialized {self}")

    @abstractmethod
    def _init_client(self) -> ClientType: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config})"

    def __str__(self) -> str:
        return self.__repr__()
