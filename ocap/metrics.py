from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Sequence

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class Prediction:
    user_id: str
    label: int
    predicted: int
    probability: float


def paper_metrics(history: Sequence[Prediction]) -> dict[str, float | None]:
    """Score the full stream and each eligible user's first 5 and 10 predictions."""
    labels = [row.label for row in history]
    predicted = [row.predicted for row in history]
    scores = [row.probability for row in history]
    metrics = {
        "F1": float(f1_score(labels, predicted, zero_division=0)) if labels else None,
        "Accuracy": float(accuracy_score(labels, predicted)) if labels else None,
        "Recall": float(recall_score(labels, predicted, zero_division=0)) if labels else None,
        "AUC": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
    }

    first_ten = defaultdict(list)
    for row in history:
        if len(first_ten[row.user_id]) < 10:
            first_ten[row.user_id].append(row)

    for k in (5, 10):
        f1_values = []
        recall_values = []
        for rows in first_ten.values():
            if len(rows) < k:
                continue
            prefix = rows[:k]
            y = [row.label for row in prefix]
            p = [row.predicted for row in prefix]
            f1_values.append(float(f1_score(y, p, zero_division=0)))
            recall_values.append(float(recall_score(y, p, zero_division=0)))
        metrics[f"F1@{k}"] = mean(f1_values) if f1_values else None
        metrics[f"R@{k}"] = mean(recall_values) if recall_values else None
    return metrics
