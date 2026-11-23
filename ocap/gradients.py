"""Track user gradients, project their histories, and update the shared backbone."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from .config import TrainingConfig


def update_user_history(
    uid: str, grad: torch.Tensor | None, decay: float,
    histories: dict[str, np.ndarray], interval_histories: dict[str, np.ndarray],
) -> np.ndarray | None:
    if grad is None:
        return None
    direction = grad.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=True)
    if np.all(np.isfinite(direction)) and np.linalg.norm(direction) > 1e-12:
        for history in (histories, interval_histories):
            previous = history.get(uid)
            if previous is None:
                history[uid] = direction.copy()
            else:
                history[uid] = decay * previous + (1.0 - decay) * direction
    return direction


def project_user_representations(
    users: Sequence[str], representations: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    ordered = [uid for uid in sorted(users) if uid in representations]
    if not ordered:
        return {}
    rows = np.stack([
        np.asarray(representations[uid], dtype=np.float32).reshape(-1)
        for uid in ordered
    ])
    matrix = torch.as_tensor(rows, dtype=torch.float32)
    standardized = (matrix - matrix.mean(dim=0)) / matrix.std(
        dim=0, unbiased=False,
    ).clamp_min(1e-6)
    rank = min(8, max(1, len(ordered) - 1), int(standardized.shape[1]))
    try:
        # X @ V_k from the historical SVD equals U_k @ Sigma_k. Computing
        # the latter from the user-by-user Gram matrix preserves all row
        # inner products in the selected principal subspace and avoids a
        # repeated SVD over the much wider gradient-projection dimension.
        eigenvalues, eigenvectors = torch.linalg.eigh(standardized @ standardized.T)
        order = torch.argsort(eigenvalues, descending=True)[:rank]
        scales = torch.sqrt(torch.clamp(eigenvalues[order], min=0.0))
        coordinates = eigenvectors[:, order] * scales.unsqueeze(0)
    except RuntimeError:
        coordinates = standardized[:, :rank]
    values = coordinates.detach().cpu().numpy().astype(np.float64, copy=False)
    return {uid: values[index].copy() for index, uid in enumerate(ordered)}


def apply_group_gradient_update(
    projection: torch.Tensor, gradients_by_group: dict[str, list[np.ndarray]], train: TrainingConfig,
) -> None:
    group_means = [np.mean(np.stack(rows), axis=0) for rows in gradients_by_group.values() if rows]
    gradients_by_group.clear()
    if not group_means:
        return
    update = np.mean(np.stack(group_means), axis=0)
    norm = float(np.linalg.norm(update))
    clip = train.gradient_direction_clip_norm
    if norm > clip > 0.0:
        update = update * (clip / norm)
    tensor = torch.as_tensor(
        update.reshape(projection.shape),
        dtype=projection.dtype,
        device=projection.device,
    )
    with torch.no_grad():
        projection.add_(
            tensor, alpha=-train.backbone_lr * train.backbone_lr_scale
        )
