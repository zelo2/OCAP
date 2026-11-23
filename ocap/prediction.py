"""Expert mixing and decision thresholds for online predictions."""

from __future__ import annotations

from typing import Deque, Mapping, Set

import numpy as np
import torch

from .config import GroupingConfig, TrainingConfig


def expert_weights(
    experts: Mapping[str, torch.Tensor], previous: Mapping[str, float], grouping: GroupingConfig,
) -> dict[str, float]:
    priors = {
        "global": grouping.low_evidence_prior_global,
        "personal": grouping.low_evidence_prior_personal,
    }
    for key in experts:
        if key.startswith("group::"):
            priors[key] = grouping.low_evidence_prior_group
    weights = {key: float(previous.get(key, priors.get(key, 0.0))) for key in experts}
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        weights = {key: 1.0 / len(experts) for key in experts}
    else:
        weights = {key: max(0.0, value) / total for key, value in weights.items()}
    return weights


def _f1_at_threshold(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    predictions = probabilities >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum((~predictions) & (labels == 1)))
    return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def stream_threshold(
    history: Deque[tuple[float, int]], prediction_count: int, cached_threshold: float,
    train: TrainingConfig,
) -> float:
    if prediction_count % max(1, train.threshold_refresh_every) == 0:
        probabilities = np.asarray([row[0] for row in history], dtype=np.float64)
        labels = np.asarray([row[1] for row in history], dtype=np.int64)
        best_score = -1.0
        best_threshold = float(train.threshold_default)
        for threshold in np.linspace(0.0, 1.0, max(2, train.threshold_grid_size)):
            score = _f1_at_threshold(probabilities, labels, float(threshold))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        return best_threshold
    return cached_threshold


def user_threshold(
    uid: str, base: float, counts: Mapping[str, int], positives_by_user: Mapping[str, int],
    assignments: Mapping[str, str], groups: Mapping[str, Set[str]], train: TrainingConfig,
) -> float:
    if not train.user_threshold_posrate_bias or counts[uid] < train.user_threshold_min_samples:
        return base
    user_rate = positives_by_user[uid] / max(1, counts[uid])
    gid = assignments.get(uid)
    members = groups.get(gid, {uid}) if gid is not None else {uid}
    positives = sum(positives_by_user[u] for u in members)
    observations = sum(counts[u] for u in members)
    group_rate = positives / observations if observations else user_rate
    adjusted = base - train.user_threshold_posrate_bias_alpha * (user_rate - group_rate)
    return float(np.clip(adjusted, 0.05, 0.95))
