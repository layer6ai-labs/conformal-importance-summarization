import re
from abc import ABC, abstractmethod
from typing import Type, Any, Optional

import numpy as np

PASCAL_PATTERN = re.compile(r'([a-z])([A-Z])')


class ConfidenceMethod(ABC):

    @abstractmethod
    def compute(self, text: str, claims: list[str], summary: Optional[str] = None) -> np.ndarray:
        """
        Abstract method to compute confidence for a given instance.
        Must be implemented by subclasses.
        """
        ...


class MethodFactory:
    _registry: dict[str, Type[ConfidenceMethod]] = {}
    _legends: dict[str, str] = {}
    _cache: dict[Any, ConfidenceMethod] = {}

    @classmethod
    def register(cls, name: str, legend: str = None):
        def decorator(method_cls: Type[ConfidenceMethod]):
            cls._registry[name] = method_cls
            cls._legends[name] = legend or ' '.join(map(lambda x: x.capitalize(), name.split(' ')))
            return method_cls

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> ConfidenceMethod:
        key = (name, tuple(args), tuple(sorted(kwargs.items())))
        if key in cls._cache:
            return cls._cache[key]

        if name not in cls._registry:
            raise ValueError(f"Unknown confidence method: {name}")

        # noinspection PyArgumentList
        result = cls._cache[key] = cls._registry[name](*args, **kwargs)
        return result

    @classmethod
    def get_legend(cls, name: str) -> str:
        return cls._legends.get(name, "Unknown")

    @classmethod
    def available_methods(cls) -> list[str]:
        return list(cls._registry.keys())
