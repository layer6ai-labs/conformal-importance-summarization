from __future__ import annotations

import json
import logging
import math
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Union, Optional

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.instance import Instance
from src.scoring import ConfidenceMethod
from src.tasks import FactualInstance, ImportanceInstance
from src.utils import ParallelTqdm


class Task(str, Enum):
    FACTUAL = 'factual'
    IMPORTANCE = 'importance'


logger = logging.getLogger(__name__)


class Dataset(ABC):
    instances: list[Instance]
    task: Task
    name: str

    @abstractmethod
    def __init__(self, instances: list[Union[str, dict, Instance]], name: str):
        ...

    def compute(self, methods: Optional[list[Union[ConfidenceMethod, str]]] = None) -> Dataset:
        for instance in self.instances:
            _ = instance.compute(methods)
        return self

    def compute_confidence(self, method: ConfidenceMethod | str, ranking: bool) -> np.ndarray:
        """
        Compute the confidence scores for the dataset.
        :param method: ConfidenceMethod
        :param ranking: Whether to use ranking or not.
        :return: Confidence scores for the dataset.
        """
        confidences = Parallel(n_jobs=-1, prefer='threads')(
            delayed(instance.get_confidence_score)(method=method, ranking=ranking)
            for instance in self.instances
        )

        return np.asarray(confidences)

    def compute_conformal_scores(
        self,
        method: ConfidenceMethod | str,
        ranking: bool,
        beta: float,
        parallel: bool = True
    ) -> np.ndarray:
        """
        Compute the conformal scores for the dataset.
        :param method: ConfidenceMethod to use as the subclaim scoring function.
        :param ranking: Whether to use ranking or not.
        :param beta: Fraction of correct subclaims required.
        :param parallel: Whether to use parallel processing.
        :return: Conformal scores for the dataset.
        """
        if parallel:
            conformal_scores = Parallel(n_jobs=-1, prefer='threads')(
                delayed(instance.get_conformal_score)(method=method, ranking=ranking, beta=beta)
                for instance in self.instances
                if method in instance.get_available_methods() or not isinstance(method, str)
            )
        else:
            conformal_scores = []
            for instance in self.instances:
                if isinstance(method, str) and method not in instance.get_available_methods():
                    continue
                else:
                    conformal_scores.append(instance.get_conformal_score(method=method, ranking=ranking, beta=beta))
        return np.asarray(conformal_scores)

    def compute_threshold(self, method: ConfidenceMethod | str, ranking: bool, alpha: float, beta: float) -> float:
        """
        Computes the quantile/threshold from conformal prediction.
        :param method: ConfidenceMethod
        :param ranking: Whether to use ranking or not.
        :param alpha: float in (0, 1)
        :param beta: Fraction of correct subclaims required.
        :return: Threshold for conformal precision.
        """
        ...

    def save(self, file: Union[str, Path]):
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)

        with open(file, 'w') as f:
            for instance in self.instances:
                f.write(instance.model_dump_json(by_alias=True) + '\n')

    @classmethod
    def _load(cls, file: Union[str, Path], task: Task):
        file = Path(file)

        clazz = FactualInstance if task == Task.FACTUAL else ImportanceInstance

        if not file.exists():
            raise FileNotFoundError(f'File {file} does not exist.')

        if file.suffix == '.json':
            with open(file, 'r') as f:
                instances = [clazz(**instance) for instance in json.load(f)['data']]
        elif file.suffix == '.jsonl':
            with jsonlines.open(file, 'r') as f:
                instances = [clazz(**instance) for instance in f]
        elif file.suffix == '.txt':
            with open(file, 'r') as f:
                instances = [line.strip() for line in f]
        else:
            raise ValueError(f'Unsupported file format: {file.suffix}')

        return cls(instances=instances, name=file.stem)

    @classmethod
    def load(cls, file: Union[str, Path]) -> Dataset:
        ...

    def plot_histogram(
            self,
            method: ConfidenceMethod | str,
            ranking: bool,
            alpha: float,
            beta: float,
            figsize: tuple[int, int] = (6, 3.5),
    ):
        logger.info(f'Plotting histogram for {self.name} with method {method} and alpha {alpha}')

        plt.figure()
        conformal_scores = self.compute_conformal_scores(method, ranking, beta)
        order = np.argsort(conformal_scores)
        sorted_conformal_scores = np.asarray(conformal_scores[order])

        # rank[i] gives the position of conformal_scores[i] in sorted_conformal_scores
        rank = np.empty(len(self.instances), dtype=int)
        rank[order] = np.arange(len(self.instances))

        target_index = int(math.ceil(len(self.instances) * (1 - alpha)))
        target_index = max(1, min(target_index, len(self.instances))) - 1

        thresholds = np.where(
            rank <= target_index, sorted_conformal_scores[target_index + 1], sorted_conformal_scores[target_index]
        )

        if len(thresholds) != len(self.instances):
            raise ValueError(f"Thresholds length {len(thresholds)} != instances length {len(self.instances)}")

        fraction_removed_results = np.empty(len(self.instances), dtype=float)
        for i, threshold in enumerate(list(thresholds)):
            instance = self.instances[i]
            accepted_subclaims = instance.get_accepted_subclaims(method, ranking=ranking, threshold=threshold)
            fraction_removed_results[i] = 1 - len(accepted_subclaims) / len(instance.claims)

        fig, ax = plt.subplots(figsize=figsize)
        plt.xlabel("Percent removed")
        plt.ylabel("Fraction of outputs")

        title = self.name.capitalize() if self.name is not None else 'Dataset'
        plt.title(f"{title}, {chr(945)}={alpha}")
        weights = np.ones_like(fraction_removed_results) / float(len(fraction_removed_results))
        plt.hist(fraction_removed_results, weights=weights)
        plt.tight_layout()
        plt.show()

    @abstractmethod
    def plot_calibration(
            self,
            method: ConfidenceMethod | str,
            ranking: bool,
            alphas: list[float],
            beta: float,
            n_samples: int,
            figsize: tuple[int, int] = (6, 4),
            fontsize=16
    ):
        ...

    @abstractmethod
    def plot_conformal(
            self,
            methods: list[Union[ConfidenceMethod, str]],
            ranking: bool,
            alphas: list[float],
            beta: float,
            fontsize=16
    ):
        ...


class FactualDataset(Dataset):
    def __init__(
            self,
            instances: list[Union[str, dict, Instance]],
            name: str = 'Dataset'
    ):
        instance_tasks: list[Union[str, dict]] = []
        instance_ids: list[int] = []

        for i, instance in enumerate(instances):
            if isinstance(instance, str):
                instance_tasks.append(delayed(FactualInstance.from_text)(instance))
            elif isinstance(instance, dict):
                instance_tasks.append(delayed(FactualInstance.from_text)(**instance))
            else:
                continue
            instance_ids.append(i)

        instance_tasks = ParallelTqdm(n_jobs=-1, parallel_kwargs={'prefer': 'threads'})(instance_tasks)

        self.instances = instances
        for i, instance in zip(instance_ids, instance_tasks):
            # noinspection PyTypeChecker
            self.instances[i] = instance

        self.name = name
        self.task = Task.FACTUAL

    def compute_threshold(self, method, ranking, alpha, beta):
        conformal_scores = self.compute_conformal_scores(method, ranking, beta)

        # The quantile is floor(alpha*(n+1))/n, and we map this to the index by
        # dropping the division by n and subtracting one (for zero-index).
        quantile_target_index = int(math.ceil((len(conformal_scores) + 1) * (1 - alpha)))
        quantile_target_index = max(1, min(quantile_target_index, len(conformal_scores))) - 1

        # np.partition finds the kth smallest element without fully sorting the array.
        threshold = np.partition(conformal_scores, quantile_target_index)[quantile_target_index]

        return float(threshold)

    @classmethod
    def load(cls, file: Union[str, Path]) -> FactualDataset:
        return cls._load(file, Task.FACTUAL)

    def plot_calibration(
            self,
            method: ConfidenceMethod,
            ranking: bool,
            alphas: list[float],
            beta: float,
            n_samples: int,
            figsize: tuple[int, int] = (6, 4),
            fontsize=16
    ):
        """
        Creates a calibration plot comparing target factuality (1 - alpha) to empirical factuality.

        :param method: ConfidenceMethod to use.
        :param ranking: Whether to use ranking or not.
        :param alphas: List of alpha values.
        :param beta: Fraction of correct subclaims required.
        :param n_samples: Number of samples to use for each alpha.
        :param figsize: Figure size.
        :param fontsize: Font size.
        """
        fig, ax = plt.subplots(figsize=figsize)
        alphas = np.sort(alphas)

        conformal_scores = self.compute_conformal_scores(method, ranking, beta)
        fraction_correct = np.zeros(len(alphas), dtype=float)
        split_index = len(self.instances) // 2

        # Cache for storing sorted adjusted scores and cumulative mean
        score_cache: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(self.instances)

        count = 0

        for i, instance in enumerate(self.instances):
            confidence_scores = instance.get_confidence_score(method, ranking=ranking)
            sort_idx = np.argsort(-confidence_scores)  # sort in descending order
            sorted_confidence_scores = confidence_scores[sort_idx]

            correctness = instance.labels[sort_idx]
            cumulative_mean = np.cumsum(correctness) / np.arange(1, len(correctness) + 1)
            score_cache[i] = (sorted_confidence_scores, cumulative_mean)

        for _ in range(n_samples):  # Perform n_samples trials
            permutation = np.random.permutation(len(self.instances))
            calibration_indices, test_indices = permutation[:split_index], permutation[split_index:]
            count += len(test_indices)

            calibration_conformal_scores = np.sort(conformal_scores[calibration_indices])

            target_indices = np.floor((len(calibration_indices) + 1) * alphas)
            target_indices = target_indices.astype(int).clip(1, len(calibration_indices)) - 1

            thresholds = calibration_conformal_scores[target_indices]

            for test_index in test_indices:
                sorted_confidence_scores, cumulative_mean = score_cache[test_index]

                # We want to use threshold that is as small as possible (side='right') (descending)
                indices = np.searchsorted(sorted_confidence_scores, thresholds, side='right')
                entailed_fractions = np.where(indices > 0, cumulative_mean[indices - 1], 1.0)
                fraction_correct += (entailed_fractions >= beta)  # size of alpha

        x_values = 1 - alphas  # x-values for theoretical bounds
        plt.plot(x_values, x_values, "--", color="gray", linewidth=2, label="Theoretical Bounds")
        plt.plot(x_values, x_values + 1 / (split_index + 1), "--", color="gray", linewidth=2)
        plt.plot(1 - alphas, fraction_correct / count, label=self.name or 'Dataset', linewidth=2)  # empirical

        plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=fontsize)
        plt.ylabel("Empirical factuality", fontsize=fontsize)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_conformal(
            self,
            methods: list[Union[ConfidenceMethod, str]],
            ranking: bool,
            alphas: list[float],
            beta: float,
            fontsize=16
    ):
        """
        Creates leave-one-out conformal plots.
        """
        plt.figure()
        alphas = np.sort(alphas)

        target_indices = np.ceil((len(self.instances) + 1) * (1 - alphas))
        target_indices = target_indices.astype(int).clip(1, len(self.instances)) - 1

        if not methods:
            raise ValueError("No methods provided.")

        x, y = None, None
        for method in tqdm(methods):
            r_scores = self.compute_conformal_scores(method, ranking, beta)

            order = np.argsort(r_scores)
            sorted_r_scores = r_scores[order]

            rank = np.empty(len(self.instances), dtype=int)
            rank[order] = np.arange(len(self.instances))  # rank[i] gives the sorted position of instance i

            score_cache: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(self.instances)
            for i, instance in enumerate(self.instances):
                scores = instance.get_confidence_score(method, ranking=ranking)
                sort_idx = np.argsort(-scores)
                sorted_scores = scores[sort_idx]

                correctness = instance.labels[sort_idx]
                cumulative_mean = np.cumsum(correctness) / np.arange(1, len(correctness) + 1)
                score_cache[i] = (sorted_scores, cumulative_mean)

            correctness_results = np.empty((len(self.instances), len(alphas)), dtype=bool)
            fraction_removed_results = np.empty((len(self.instances), len(alphas)), dtype=float)

            thresholds = np.where(
                rank[:, None] <= target_indices, sorted_r_scores[target_indices + 1], sorted_r_scores[target_indices]
            )  # [n, len(alphas)]

            for i, threshold in enumerate(thresholds):
                threshold: np.ndarray

                sorted_scores, cumulative_mean = score_cache[i]
                indices = np.searchsorted(-sorted_scores, -threshold, side='right')
                entailed_fraction = np.where(indices > 0, cumulative_mean[indices - 1], 1.0)
                correctness_results[i] = entailed_fraction >= beta
                fraction_removed_results[i] = 1 - indices / len(sorted_scores)

            # Compute mean correctness and beta removed per alpha
            x = correctness_results.mean(axis=0)
            y = fraction_removed_results.mean(axis=0)

            # Compute error bars
            yerr = fraction_removed_results.std(axis=0) * 1.96 / np.sqrt(len(self.instances))
            plt.errorbar(x, y, yerr=yerr, label=method.get_legend(), linewidth=2)

        # Plot base factuality point
        plt.scatter(x[-1], y[-1], color="black", marker="*", s=235, label="Base factuality", zorder=1000)

        plt.title(f"{self.name}, beta={beta}" if beta != 1 else self.name, fontsize=fontsize + 4)
        plt.xlabel(f"Fraction achieving avg factuality >= {beta}"
                   if beta != 1 else "Fraction of factual outputs", fontsize=fontsize)
        plt.ylabel("Average percent removed", fontsize=fontsize)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.show()


class ImportanceDataset(Dataset):
    def __init__(
            self,
            instances: list[Union[str, dict, ImportanceInstance]],
            name: str = 'Dataset'
    ):
        instance_tasks: list[Union[str, dict]] = []
        instance_ids: list[int] = []

        for i, instance in enumerate(instances):
            if isinstance(instance, str):
                instance_tasks.append(delayed(ImportanceInstance.from_text)(instance))
            elif isinstance(instance, dict):
                instance_tasks.append(delayed(ImportanceInstance.from_text)(**instance))
            else:
                continue
            instance_ids.append(i)

        instance_tasks = ParallelTqdm(n_jobs=-1, parallel_kwargs={'prefer': 'threads'})(instance_tasks)

        self.instances = instances
        for i, instance in zip(instance_ids, instance_tasks):
            # noinspection PyTypeChecker
            self.instances[i] = instance

        self.name = name
        self.task = Task.IMPORTANCE

    def compute_threshold(self, method, ranking, alpha, beta):
        conformal_scores = self.compute_conformal_scores(method, ranking, beta)

        # The quantile is floor(alpha*(n+1))/n, and we map this to the index by
        # dropping the division by n and subtracting one (for zero-index).
        quantile_target_index = int(math.floor((len(conformal_scores) + 1) * alpha))
        quantile_target_index = max(1, min(quantile_target_index, len(conformal_scores))) - 1

        # np.partition finds the kth smallest element without fully sorting the array.
        threshold = np.partition(conformal_scores, quantile_target_index)[quantile_target_index]

        return float(threshold)

    @classmethod
    def load(cls, file: Union[str, Path]) -> ImportanceDataset:
        return cls._load(file, Task.IMPORTANCE)

    def plot_calibration(
            self,
            method: ConfidenceMethod | str,
            ranking: bool,
            alphas: list[float],
            betas: list[float] | float,
            n_samples: int,
            cal_size: float = 0.1,
            figsize: tuple[int, int] = (20, 6),  # Adjusted for 1x2 layout
            fontsize=24, 
            name: str = "Dataset"
    ):
        """
        Creates a calibration plot comparing target importance (1 - alpha) to importance factuality.
        Can plot multiple methods on the same axes for comparison. Also plots the difference between
        actual and empirical importance on a separate axis.

        :param methods: List of ConfidenceMethods to use or single method.
        :param ranking: Whether to use ranking or not.
        :param alphas: List of alpha values.
        :param beta: Fraction of important subclaims required.
        :param n_samples: Number of samples to use for each alpha.
        :param figsize: Figure size.
        :param fontsize: Font size.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)  # Create 1x2 layout
        alphas = np.sort(alphas)

        # Convert single method to list for uniform processing
        if not isinstance(betas, list):
            betas = [betas]

        split_index = int(np.round(len(self.instances) * cal_size))
        x_values = 1 - alphas  # x-values for theoretical bounds

        for beta in betas:
            conformal_scores = self.compute_conformal_scores(method, ranking, beta)
            fraction_important = np.zeros(len(alphas), dtype=float)

            # Cache for storing sorted adjusted scores and cumulative mean
            score_cache: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(self.instances)
            count = 0

            for i, instance in enumerate(self.instances):
                confidence_scores = instance.get_confidence_score(method, ranking=ranking)
                sort_idx = np.argsort(-confidence_scores)  # sort in descending order
                sorted_confidence_scores = confidence_scores[sort_idx]

                labels = instance.labels[sort_idx]
                total_importance = np.sum(labels)

                if total_importance == 0:
                    cumulative_mean = np.ones_like(confidence_scores)  # No important claims
                else:
                    cumulative_mean = np.cumsum(labels) / total_importance
                score_cache[i] = (sorted_confidence_scores, cumulative_mean)

            for _ in range(n_samples):  # Perform n_samples trials
                permutation = np.random.permutation(len(self.instances))
                calibration_indices, test_indices = permutation[:split_index], permutation[split_index:]
                count += len(test_indices)

                calibration_conformal_scores = np.sort(conformal_scores[calibration_indices])

                target_indices = np.floor((len(calibration_indices) + 1) * (alphas))
                target_indices = target_indices.astype(int).clip(1, len(calibration_indices)) - 1

                thresholds = calibration_conformal_scores[target_indices]

                for test_index in test_indices:
                    sorted_confidence_scores, cumulative_mean = score_cache[test_index]

                    # We want to use threshold that is as large as possible (side='right') (descending)
                    indices = np.searchsorted(-sorted_confidence_scores, -thresholds, side='right')
                    important_fractions = np.where(indices > 0, cumulative_mean[indices - 1], 0.0)
                    fraction_important += (important_fractions >= beta)  # size of alpha

            # Plot empirical coverage
            axes[0].plot(1 - alphas, fraction_important / count, label=f"{chr(946)} = {beta}", linewidth=2)

            # Plot difference between theoretical and empirical coverage
            difference = (fraction_important / count) - (1 - alphas)
            axes[1].plot(1 - alphas, difference, label=f"{chr(946)} = {beta}", linewidth=2)

        # Plot theoretical bounds on the first axis
        axes[0].plot(x_values, x_values, "--", color="black", linewidth=1, label="Theoretical Bounds")
        axes[0].plot(x_values, x_values + 1 / (split_index + 1), "--", color="black", linewidth=2)
        axes[0].set_xlabel(f"Target Coverage (1 - {chr(945)})", fontsize=fontsize)
        axes[0].set_ylabel("Empirical Coverage", fontsize=fontsize)
        axes[0].set_title(f"Empirical vs Theoretical Coverage ({name})", fontsize=fontsize)
        axes[0].legend(fontsize=fontsize)

        # Configure the second axis
        axes[1].axhline(0, color="black", linestyle="--", linewidth=3.5)  # Add a horizontal line at y=0
        axes[1].axhline(1 / (split_index + 1), color="black", linestyle="--", linewidth=2)  # Add upper bound
        axes[1].set_xlabel(f"Target Coverage (1 - {chr(945)})", fontsize=fontsize)
        axes[1].set_ylabel("Empirical - Theoretical", fontsize=fontsize)
        axes[1].set_title(f"Difference in Coverage ({name})", fontsize=fontsize)
        axes[1].legend(fontsize=fontsize)

        plt.tight_layout()
        return fig, axes  # Return figure and axes for further customization

    def plot_conformal(
            self,
            methods: list[ConfidenceMethod],
            ranking: bool,
            alphas: list[float],
            beta: float,
            fontsize=16
    ):
        """
        Creates leave-one-out conformal plots.
        """
        fig, ax = plt.subplots()  # Create figure and axis objects
        alphas = np.sort(alphas)

        target_indices = np.floor((len(self.instances) + 1) * (alphas))
        target_indices = target_indices.astype(int).clip(1, len(self.instances)) - 1

        if not methods:
            raise ValueError("No methods provided.")

        x, y = None, None
        for method in tqdm(methods):
            r_scores = self.compute_conformal_scores(method, ranking, beta)

            order = np.argsort(r_scores)
            sorted_r_scores = r_scores[order]

            rank = np.empty(len(self.instances), dtype=int)
            rank[order] = np.arange(len(self.instances))  # rank[i] gives the sorted position of instance i

            score_cache: list[Optional[tuple[np.ndarray, np.ndarray]]] = [None] * len(self.instances)
            for i, instance in enumerate(self.instances):
                confidence_scores = instance.get_confidence_score(method, ranking=ranking)
                sort_idx = np.argsort(-confidence_scores)
                sorted_confidence_scores = confidence_scores[sort_idx]

                importance = instance.labels[sort_idx]
                total_importance = np.sum(importance)

                if total_importance == 0:
                    cumulative_mean = np.ones_like(importance)  # No important claims
                else:
                    cumulative_mean = np.cumsum(importance) / total_importance
                score_cache[i] = (sorted_confidence_scores, cumulative_mean)

            importance_results = np.empty((len(self.instances), len(alphas)), dtype=bool)
            fraction_removed_results = np.empty((len(self.instances), len(alphas)), dtype=float)

            shifted_target_indices = (target_indices + 1).clip(0, len(self.instances) - 1)
            thresholds = np.where(
                rank[:, None] <= target_indices, sorted_r_scores[shifted_target_indices],
                sorted_r_scores[target_indices]
            )  # [n, len(alphas)]

            for i, threshold in enumerate(thresholds):
                threshold: np.ndarray

                sorted_confidence_scores, cumulative_mean = score_cache[i]

                indices = np.searchsorted(-sorted_confidence_scores, -threshold, side='right')
                importance_fraction = np.where(indices > 0, cumulative_mean[indices - 1], 1.0)
                importance_results[i] = importance_fraction >= beta
                fraction_removed_results[i] = 1 - indices / len(sorted_confidence_scores)

            # Compute mean correctness and beta removed per alpha
            x = importance_results.mean(axis=0)
            y = fraction_removed_results.mean(axis=0)

            # Compute error bars
            yerr = fraction_removed_results.std(axis=0) * 1.96 / np.sqrt(len(self.instances))
            ax.errorbar(x, y, yerr=yerr, label=method, linewidth=2)

        # Plot base factuality point
        ax.scatter(x[0], y[0], color="black", marker="*", s=235, label="Base factuality", zorder=1000)

        ax.set_title(f"{self.name}, beta={beta}" if beta != 1 else self.name, fontsize=fontsize + 4)
        ax.set_xlabel(f"Fraction achieving importance recall >= {beta}"
                      if beta != 1 else "Fraction of important outputs", fontsize=fontsize)
        ax.set_ylabel("Average percent removed", fontsize=fontsize)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.tight_layout()

        return fig, ax  # Return the figure and axis objects

    @classmethod
    def from_dataframe(cls, df: pl.DataFrame, name="Dataset", label_col="input_sentences_labels") -> ImportanceDataset:
        """
        Create an ImportanceDataset from a Polars DataFrame.
        :param df: Polars DataFrame containing the dataset.
        :param name: Name of the dataset.
        :return: An instance of ImportanceDataset.
        """
        instances = [ImportanceInstance.from_row(row, label_col) for row in df.iter_rows(named=True)]
        return cls(instances=instances, name=name)

    def write_to_cache(self, dataset_name):
        with open(f"../../data/cache/{dataset_name}_cal.pkl", "wb") as f:
            pickle.dump(self, f)
