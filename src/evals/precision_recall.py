import numpy as np

from .base import EvalMethodFactory, EvalMetric


@EvalMethodFactory.register('precision-recall')
class PrecisionRecallEval(EvalMetric):
    """
    Evaluate the precision of the summary.
    """

    def compute(self):
        precisions = []
        recalls = []
        q = self.cal_dataset.compute_threshold(self.method, ranking=self.rank, alpha=self.alpha, beta=self.beta)
        for instance in self.dataset.instances:
            if self.method not in instance.get_available_methods():
                continue
            summary_claims = instance.get_accepted_subclaims(self.method, ranking=self.rank, threshold=q)
            instance_precision = np.sum([claim.label.is_important() for claim in summary_claims]) / len(
                summary_claims) if summary_claims else 0
            precisions.append(instance_precision)

            instance_recall = np.sum([claim.label.is_important() for claim in summary_claims]) / np.sum(
                instance.labels) if np.sum(instance.labels) > 0 else 1
            recalls.append(instance_recall)
        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls)
        }
