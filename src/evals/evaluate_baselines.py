import numpy as np
from sklearn.metrics import average_precision_score

from src.data.loaddata import load_dataset


def evaluate_baseline(dataset_name, score_column="gpt_scores_binary"):
    _, d_test = load_dataset(dataset_name, as_dataframe=True)
    recalls = []
    reduction_lengths = []
    auprc = []
    for row in d_test.iter_rows(named=True):
        labels = np.array(row["input_sentences_labels"]).astype(float)
        baseline_scores = np.array(row[score_column]).astype(float)

        if not labels.any() or np.isnan(baseline_scores).any():
            continue
        else:
            auprc.append(average_precision_score(labels, baseline_scores))
            recalls.append(np.sum((labels == 1.0) & (baseline_scores == 1.0)) / np.sum(labels))

        reduction_lengths.append(np.sum(baseline_scores) / len(labels))

    return {
        "recall": np.mean(recalls),
        "reduction_ratio_sentences": np.mean(reduction_lengths),
        "auprc": np.mean(auprc)
    }
