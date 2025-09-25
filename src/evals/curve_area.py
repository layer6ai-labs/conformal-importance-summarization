import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from .base import EvalMethodFactory, EvalMetric


@EvalMethodFactory.register('auc')
class AUCEval(EvalMetric):

    def __init__(self, method, dataset):
        super().__init__(method, dataset, None, None, None, None)

    def compute(self):
        aucs = []
        for instance in self.dataset.instances:
            scores = instance.get_confidence_score(self.method, ranking=False)
            if not instance.labels.any() or instance.labels.all():
                continue
            auc = roc_auc_score(instance.labels, scores)
            aucs.append(auc)
        return {'auc': np.mean(aucs)}


@EvalMethodFactory.register('prauc')
class PRAUCEval(EvalMetric):

    def __init__(self, method, dataset):
        super().__init__(method, dataset, None, None, None, None)

    def compute(self, agg="mean"):
        pr_aucs = []
        for instance in self.dataset.instances:
            if self.method not in instance.get_available_methods():
                continue
            scores = instance.get_confidence_score(self.method, ranking=False)
            if not instance.labels.any():
                continue
            pr_auc = average_precision_score(instance.labels, scores)
            pr_aucs.append(pr_auc)
        if agg == "median":
            return {'prauc': np.median(pr_aucs)}
        elif agg == "mean":
            return {'prauc': np.mean(pr_aucs)}
        else:
            raise ValueError(f"Aggregation method {agg} not supported, choose from ['mean', 'median'].")


@EvalMethodFactory.register('auc_pooled')
class AUCEvalPooled(EvalMetric):

    def __init__(self, method, dataset):
        super().__init__(method, dataset, None, None, None, None)

    def compute(self):
        all_labels = []
        all_scores = []
        for instance in self.dataset.instances:
            if self.method not in instance.get_available_methods():
                continue
            all_labels.extend(instance.labels)
            all_scores.extend(instance.get_confidence_score(self.method, ranking=False))
        auc = roc_auc_score(all_labels, all_scores)
        return {'auc_pooled': auc}


@EvalMethodFactory.register('prauc_pooled')
class PRAUCEvalPooled(EvalMetric):

    def __init__(self, method, dataset):
        super().__init__(method, dataset, None, None, None, None)

    def compute(self):
        all_labels = []
        all_scores = []
        for instance in self.dataset.instances:
            if self.method not in instance.get_available_methods():
                continue
            all_labels.extend(instance.labels)
            all_scores.extend(instance.get_confidence_score(self.method, ranking=False))
        pr_auc = average_precision_score(all_labels, all_scores)
        return {'prauc_pooled': pr_auc}
