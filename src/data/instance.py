from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from src.scoring import *

MAX_CALLS_PER_MINUTE = 100
FREQUENCY_SCORER_N_SAMPLES = 5


class Instance(BaseModel, ABC):
    DEFAULT_METHODS: ClassVar[set[str]] = {'gpt', 'optimal'}

    model_config = ConfigDict(
        json_encoders={np.ndarray: lambda v: v.tolist(), pl.Series: lambda v: v.to_list()},
        arbitrary_types_allowed=True,
    )

    original_output: str = Field(alias='original-output', default=None)

    @property
    @abstractmethod
    def noise(self) -> np.ndarray:
        """
        Noise for each subclaim. It is used to add randomness to the confidence scores to break ties.

        :return: Array of noise values.
        """
        ...

    @property
    @abstractmethod
    def labels(self) -> np.ndarray:
        """
        Correctness of each subclaim. It is used to compute the conformal scores.

        :return: Array of correctness values.
        """
        ...

    @abstractmethod
    def get_confidence_score(
            self,
            method: Union[ConfidenceMethod, str],
            ranking: bool,
            noise: bool = True
    ) -> np.ndarray:
        """
        Computes the confidence score R(c) for each subclaim using the specified method.
        - For factual claims, R(c) is the factuality score.
        - For importance claims, R(c) is the importance score.

        :param method: ConfidenceMethod | str to use as the subclaim scoring function.
        :param ranking: Whether to use ranking or not.
        :param noise: Whether to add noise to the scores.
        :return: Array of confidence scores.
        """
        ...

    @abstractmethod
    def get_accepted_subclaims(
            self, method: ConfidenceMethod, ranking: bool, threshold: float
    ) -> list[str]:
        """
        Get the claims with a confidence score R(c) above the threshold.

        :param method: ConfidenceMethod to use as the subclaim scoring function.
        :param ranking: Whether to use ranking or not.
        :param threshold: Threshold for the confidence score.
        :return: List of accepted subclaims.
        """
        ...

    @abstractmethod
    def get_conformal_score(self, method: ConfidenceMethod, ranking: bool, beta: float) -> float:
        """
        Compute the conformal score for a dataset instance when method is used as the subclaim scoring function.

        :param method: ConfidenceMethod to use as the subclaim scoring function.
        :param ranking: Whether to use ranking or not.
        :param beta: Fraction of correct subclaims required.
        :return: Conformal score for the dataset instance.
        """
        ...

    def compute(self, methods: Optional[list[Union[ConfidenceMethod, str]]] = None) -> Instance:
        """
        Computes all methods, which populate the fields of the instance.

        :return: Instance with all fields populated.
        """
        if methods is None:
            methods = MethodFactory.available_methods()

        for method in methods:
            self.get_confidence_score(method, ranking=False)

        return self
