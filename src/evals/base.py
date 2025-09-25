from abc import ABC, abstractmethod
from typing import Type, Any, Union, Optional

from src.data.dataset import ImportanceDataset, FactualDataset
from src.scoring.method import ConfidenceMethod


class EvalMetric(ABC):
    def __init__(
            self,
            method: ConfidenceMethod | str,
            dataset: Union[FactualDataset, ImportanceDataset],
            cal_dataset: Union[FactualDataset, ImportanceDataset],
            alpha: float,
            beta: float,
            rank: bool = False
    ):
        self.method = method
        self.dataset = dataset
        self.cal_dataset = cal_dataset
        self.alpha = alpha
        self.beta = beta
        self.rank = rank

    @abstractmethod
    def compute(self, ) -> dict[str, float]:
        """
        Abstract method to compute an evaluation metric for a given dataset using a 
        specified conformal scoring method
        """
        ...


class EvalMethodFactory:
    _registry: dict[str, Type[EvalMetric]] = {}
    _legends: dict[str, str] = {}
    _cache: dict[Any, EvalMetric] = {}

    @classmethod
    def register(cls, name: str, legend: str = None):
        def decorator(method_cls: Type[EvalMetric]):
            cls._registry[name] = method_cls
            cls._legends[name] = legend or ' '.join(map(lambda x: x.capitalize(), name.split(' ')))
            return method_cls

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> EvalMetric:
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
