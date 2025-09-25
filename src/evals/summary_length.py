import numpy as np
from nltk import word_tokenize

from .base import EvalMethodFactory, EvalMetric


@EvalMethodFactory.register('sentence_reduction')
class SentenceReductionEval(EvalMetric):
    """
    Evaluate the summary length in terms of the number of sentences.
    """

    def compute(self, agg="mean"):
        lengths = []
        q = self.cal_dataset.compute_threshold(self.method, ranking=self.rank, alpha=self.alpha, beta=self.beta)
        for instance in self.dataset.instances:
            if isinstance(self.method, str) and self.method not in instance.get_available_methods():
                continue
            text_length = len(instance.claims)
            summary_length = len(instance.get_accepted_subclaims(self.method, ranking=self.rank, threshold=q))
            lengths.append((text_length, summary_length))

        if agg == "mean":
            average_func = np.mean
        elif agg == "median":
            average_func = np.median
        return {
            'text_length_sentences': average_func([length[0] for length in lengths]),
            'summary_length_sentences': average_func([length[1] for length in lengths]),
            'reduction_ratio_sentences': average_func([length[1] / length[0] for length in lengths]),
            "fraction_removed_sentences": average_func([(length[0] - length[1]) / length[0] for length in lengths]),
        }


@EvalMethodFactory.register('word_reduction')
class WordReductionEval(EvalMetric):
    """
    Evaluate the summary length in terms of the number of words.
    """

    def compute(self):
        lengths = []
        q = self.cal_dataset.compute_threshold(self.method, ranking=self.rank, alpha=self.alpha, beta=self.beta)
        for instance in self.dataset.instances:
            text_length = len(word_tokenize(instance.text))
            summary_length = sum([len(word_tokenize(claim.subclaim))
                                  for claim in
                                  instance.get_accepted_subclaims(self.method, ranking=self.rank, threshold=q)])
            lengths.append((text_length, summary_length))

        return {
            'text_length_words': np.mean([length[0] for length in lengths]),
            'summary_length_words': np.mean([length[1] for length in lengths]),
            'reduction_ratio_words': np.mean([length[1] / length[0] for length in lengths]),
            'reduction_average_words': np.mean([length[0] - length[1] for length in lengths]),
        }
