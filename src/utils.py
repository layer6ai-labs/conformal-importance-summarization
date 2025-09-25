from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
import tqdm
from joblib import Parallel
from scipy.spatial.distance import pdist, squareform


class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Additional parameters:
    ----------------------
    n_jobs: int, default: 0
        The maximum number of concurrently running jobs.
        Will pass `n_jobs if int(n_jobs)>0 else n_jobs-1` to joblib Parallel.
        i.e.: 1 means no parallel computing, n means n CPUs, 0 means all CPUs,
        and -n means (num_cpu-n) CPUs.
    total_tasks: int, default: None
        The number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    parallel_kwargs: dict, optional
        kwargs to pass to `joblib.Parralel`

    tqdm_kwargs: dict, optional
        kwargs to pass to `tqdm.tqdm`, like 'desc', 'ncols', 'disable'.


    Usage:
    ------
    >>> from joblib_parallel_with_tqdm import ParallelTqdm, delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(i) for i in range(10)])
    80%|████████  | 8/10 [00:07<00:01,  1.08tasks/s]

    """

    def __init__(
            self,
            n_jobs: int = 0,
            *,
            total_tasks: int | None = None,
            show_joblib_header: bool = False,
            parallel_kwargs: dict | None = None,
            tqdm_kwargs: dict | None = None,
    ):
        # set the Parallel class
        if parallel_kwargs is None:
            parallel_kwargs = {}
        parallel_kwargs["verbose"] = 1 if show_joblib_header else 0
        parallel_kwargs["n_jobs"] = n_jobs if int(n_jobs) > 0 else n_jobs - 1
        super().__init__(**parallel_kwargs)
        # prepare the tqdm kwargs
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        if "iterable" in tqdm_kwargs:
            raise TypeError(
                "keyword argument 'iterable' is not supported in 'tqdm_kwargs'."
            )
        if "total" in tqdm_kwargs:
            total_from_tqdm = tqdm_kwargs.pop("total")
            if total_tasks is None:
                total_tasks = total_from_tqdm
            elif total_tasks != total_from_tqdm:
                raise ValueError(
                    "keyword argument 'total' for tqdm_kwargs is specified and different from 'total_tasks'"
                )
        self.tqdm_kwargs = dict(unit="tasks") | tqdm_kwargs
        self.total_tasks = total_tasks
        self.progress_bar: tqdm.tqdm | None = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm.tqdm(
                total=self.total_tasks,
                **self.tqdm_kwargs,
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.progress_bar.total = self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def sim2diss(s, method='reciprocal'):
    EPS = np.finfo(float).eps
    s = np.array(s)

    if method == 'corr':
        if np.any(s < -1) or np.any(s > 1):
            raise ValueError("Correlations expected for correlation transformation.")
        dissmat = np.sqrt(1 - s)

    elif method == 'reverse':
        dissmat = np.max(s) + np.min(s) - s

    elif method == 'reciprocal':
        s[s == 0] = np.nan
        dissmat = 1 / s

    elif method == 'ranks':
        dissmat = np.argsort(-s, axis=None).reshape(s.shape)

    elif method == 'exp':
        dissmat = -np.log((EPS + s) / (EPS + np.max(s)))

    elif method == 'gaussian':
        dissmat = np.sqrt(-np.log((EPS + s) / (EPS + np.max(s))))

    elif method == 'cooccurrence':
        rsum = np.sum(s, axis=1, keepdims=True)
        csum = np.sum(s, axis=0, keepdims=True)
        tsum = np.sum(s)
        s = (tsum * s) / (rsum @ csum)
        dissmat = (1 / (1 + s))

    elif method == 'gravity':
        s[s == 0] = np.nan
        rsum = np.sum(s, axis=1, keepdims=True)
        csum = np.sum(s, axis=0, keepdims=True)
        tsum = np.sum(s)
        s = (rsum @ csum) / (tsum * s)
        dissmat = np.sqrt(s)

    elif method == 'confusion':
        if np.any(s < 0) or np.any(s > 1):
            raise ValueError("Proportions expected for confusion transformation.")
        dissmat = 1 - s

    elif method == 'transition':
        if np.any(s < 0):
            raise ValueError("Frequencies expected for transition transformation.")
        s[s == 0] = np.nan
        dissmat = 1 / np.sqrt(s)

    elif method == 'membership':
        dissmat = 1 - s

    elif method == 'probability':
        if np.any(s < 0) or np.any(s > 1):
            raise ValueError("Probabilities expected for probability transformation.")
        s[s == 0] = np.nan
        dissmat = 1 / np.sqrt(np.arcsin(s))

    elif method == "one_minus":
        if np.any(s < 0) or np.any(s > 1):
            raise ValueError("Probabilities expected for one_minus transformation.")
        dissmat = 1 - s

    else:
        raise ValueError("Invalid method specified.")

    return dissmat


# Plotting
def plot_distances(sentence_emb, target_emb, plot_emb, method='cosine'):
    sentence_adj = 1 - squareform(pdist(sentence_emb, method))
    target_adj = 1 - squareform(pdist(target_emb, method))
    emb_adj = 1 - squareform(pdist(plot_emb, method))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(sentence_adj, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Sentence ({sentence_emb.shape[1]}D) Pairwise {method.capitalize()} Similarity')

    axes[1].imshow(target_adj, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Target ({target_emb.shape[1]}D) Pairwise {method.capitalize()} Similarity')

    axes[2].imshow(emb_adj, cmap='viridis', aspect='auto')
    axes[2].set_title(f'Plot (2D) Pairwise {method.capitalize()} Similarity')

    plt.tight_layout()
    plt.show()


def plot_dendrogram(linkage_matrix, sentences):
    fig, ax = plt.subplots(figsize=(8, 6))

    sch.dendrogram(linkage_matrix, labels=sentences, leaf_rotation=0, leaf_font_size=10, orientation="right")

    plt.title("Hierarchical Clustering Dendrogram of Sentences")
    plt.xlabel("Distance")
    plt.ylabel("Sentence")
    plt.gca().invert_yaxis()

    plt.show()


def combine_conversations(conversation):
    """
    Used to process the MTS dialogue dataset
    Combines a list of conversation sentences where each sentence starts with "Doctor:" or "Patient:".
    It concatenates a doctor's sentence with all immediately following patient sentences,

    Returns a list of combined conversation strings.
    """
    sentences = [sentence.strip() for sentence in conversation.replace('\r', '').split('\n') if sentence.strip()]
    combined = []  # List to hold combined conversation segments.
    current = ""  # Holds the current conversation segment.

    for sentence in sentences:
        sentence = sentence.strip()
        # If it's a doctor's sentence, start a new conversation segment.
        if sentence.startswith("Doctor:"):
            if current:
                combined.append(current)
            current = sentence
        # If it's a patient's sentence, append it to the current segment.
        elif sentence.startswith("Patient:"):
            # Append with a space so the sentences join naturally.
            if current:
                current += " " + sentence
            else:
                current = sentence  # In case conversation starts with Patient (unlikely but handled).
        else:
            # If there is no role-indicator, simply add it to the current segment.
            current += " " + sentence
    if current:
        combined.append(current)

    # Finally, remove the role-indicators.
    final_combined = [
        s.strip() for s in combined
        # s.replace("Doctor:", "").replace("Patient:", "").strip() for s in combined
    ]
    return final_combined
