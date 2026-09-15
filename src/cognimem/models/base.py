from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> Any:
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, path: str):
        pass