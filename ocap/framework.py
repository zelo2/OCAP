from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, MutableMapping, Sequence, Set

import numpy as np
import torch

from .config import ExperimentConfig
from .data import DataTriplet
from .grouping import agglomerative_groups, cosine, match_clusters, normalize
from .metrics import binary_metrics, first_k_metrics
from .model import OCAPModel, weighted_focal_loss


@dataclass
class RecentDataPair:
    features: np.ndarray
    label: int


class OCAP:
    """Overall online framework of OCAP."""

    def __init__(
        self,
        input_dim: int,
        config: ExperimentConfig,
    ) -> None:
        self.cfg = copy.deepcopy(config)
        self.device = torch.device(self.cfg.device)
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.model = OCAPModel(input_dim, self.cfg.model, self.cfg.device)
        self.backbone_optimizer = torch.optim.AdamW(
            self.model.backbone_parameters(), lr=self.cfg.training.backbone_lr,
            weight_decay=self.cfg.training.weight_decay,
        )
        self.adapter_optimizers: Dict[str, torch.optim.Adam] = {}
        self.user_record_counts: MutableMapping[str, int] = defaultdict(int)
        self.positive_record_counts: MutableMapping[str, int] = defaultdict(int)
        self.user_bias: Dict[str, float] = {}
        self.gradient_history: Dict[str, np.ndarray] = {}
        self.interval_gradient_history: Dict[str, np.ndarray] = {}
        self.recent_data_buffers: Dict[str, Deque[RecentDataPair]] = defaultdict(lambda: deque(maxlen=64))
        self.group_assignment: Dict[str, str] = {}
        self.group_set: Dict[str, Set[str]] = {}
        self.next_group_index = 0
        self.group_update_count = 0
        self.prediction_history: List[Dict[str, object]] = []
        self.group_trace: List[Dict[str, object]] = []
        self._last_reassignment_diagnostics: Dict[str, int] = {}
        self.group_gradient_buffer: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.smooth_probabilities: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, self.cfg.training.smoothing_window))
        )
        self.threshold_history: Deque[tuple[float, int]] = deque(
            maxlen=max(1, self.cfg.training.threshold_window)
        )
        self.cached_threshold = float(self.cfg.training.threshold_default)
        self.threshold_updates = 0
        self.mix_weights: Dict[str, Dict[str, float]] = {}
        self.group_initialized_personal_adapters: Set[str] = set()
        self.personal_adapter_update_count: MutableMapping[str, int] = defaultdict(int)
        self._ensure_adapter("global")

    def _ensure_adapter(self, key: str, *, reset_optimizer: bool = False) -> None:
        adapter = self.model.ensure_adapter(key)
        if reset_optimizer or key not in self.adapter_optimizers:
            self.adapter_optimizers[key] = torch.optim.Adam(
                adapter.parameters(), lr=self.cfg.training.adapter_lr,
            )

    def _personal_adapter_key(self, uid: str) -> str:
        key = f"personal::{uid}"
        self._ensure_adapter(key)
        return key

    def _routed_adapter_key(self, uid: str) -> str:
        if uid in self.group_assignment:
            key = f"group::{self.group_assignment[uid]}"
            self._ensure_adapter(key)
            return key
        return self._personal_adapter_key(uid)

    def _tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)

    def _task_loss(self, logits: torch.Tensor, label: int) -> torch.Tensor:
        target = torch.tensor([label], dtype=torch.long, device=self.device)
        train = self.cfg.training
        return weighted_focal_loss(
            logits, target, train.focal_alpha, train.focal_gamma, train.positive_class_weight
        )

    def _group_mean_user_bias(self, uid: str) -> float:
        gid = self.group_assignment.get(uid)
        if gid is None:
            return 0.0
        values = [
            float(self.user_bias[member])
            for member in self.group_set.get(gid, set())
            if member != uid and member in self.user_bias
        ]
        return float(np.mean(values)) if values else 0.0

    def _ensure_user_bias(self, uid: str) -> float:
        if uid in self.user_bias:
            return float(self.user_bias[uid])
        ref = self.cfg.refinements
        value = ref.per_user_bias_init_group_weight * self._group_mean_user_bias(uid)
        value = float(np.clip(value, -ref.per_user_bias_clip, ref.per_user_bias_clip))
        self.user_bias[uid] = value
        return value

    def _apply_bias(
        self, logits: torch.Tensor, uid: str, *, warmup: bool = False,
    ) -> torch.Tensor:
        if warmup:
            return logits
        offset = torch.zeros_like(logits)
        offset[:, 1] = self._ensure_user_bias(uid)
        return logits + offset

    def _group_representations(self) -> Dict[str, np.ndarray]:
        output = {}
        for gid, members in self.group_set.items():
            rows = [self.gradient_history[u] for u in members if u in self.gradient_history]
            if rows:
                output[gid] = normalize(np.mean(np.stack(rows), axis=0))
        return output

    def _nearest_group(self, uid: str) -> str | None:
        if uid in self.group_assignment:
            return self.group_assignment[uid]
        if uid not in self.gradient_history:
            return None
        prototypes = self._group_representations()
        if not prototypes:
            return None
        return max(prototypes, key=lambda gid: cosine(self.gradient_history[uid], prototypes[gid]))

    def _expert_logits(self, features: np.ndarray, uid: str) -> Dict[str, torch.Tensor]:
        x = self._tensor(features)
        hidden = self.model.hidden(x)
        base = self.model.base_logits_from_hidden(hidden)
        experts: Dict[str, torch.Tensor] = {"global": base}
        personal_key = self._personal_adapter_key(uid)
        experts["personal"] = base + self.model.adapter_logits_from_hidden(hidden, personal_key)
        gid = self._nearest_group(uid)
        if gid is not None:
            group_key = f"group::{gid}"
            self._ensure_adapter(group_key)
            experts[group_key] = base + self.model.adapter_logits_from_hidden(hidden, group_key)
        return experts

    def _mix_logits(self, uid: str, experts: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        grouping = self.cfg.grouping
        priors = {
            "global": grouping.low_evidence_prior_global,
            "personal": grouping.low_evidence_prior_personal,
        }
        for key in experts:
            if key.startswith("group::"):
                priors[key] = grouping.low_evidence_prior_group
        previous = self.mix_weights.get(uid, {})
        weights = {key: float(previous.get(key, priors.get(key, 0.0))) for key in experts}
        total = sum(max(0.0, value) for value in weights.values())
        if total <= 0.0:
            weights = {key: 1.0 / len(experts) for key in experts}
        else:
            weights = {key: max(0.0, value) / total for key, value in weights.items()}
        self.mix_weights[uid] = weights
        mixed = sum(float(weights[key]) * value for key, value in experts.items())
        return mixed, weights

    def _prediction_logits(
        self, triplet: DataTriplet,
    ) -> tuple[torch.Tensor, str, Dict[str, torch.Tensor], Dict[str, float]]:
        uid = triplet.user_id
        active = self._routed_adapter_key(uid)
        grouping = self.cfg.grouping
        can_mix = (
            triplet.index >= grouping.backbone_warmup_steps
            and self.user_record_counts[uid] < grouping.low_evidence_mixing_max_records
        )
        if can_mix:
            experts = self._expert_logits(triplet.features, uid)
            mixed, weights = self._mix_logits(uid, experts)
            return self._apply_bias(mixed, uid, warmup=False), "safe_mix", experts, weights
        logits = self.model(self._tensor(triplet.features), active)
        return self._apply_bias(
            logits, uid, warmup=triplet.index < grouping.backbone_warmup_steps,
        ), active, {}, {}

    @staticmethod
    def _positive_probability(logits: torch.Tensor) -> float:
        return float(torch.sigmoid(logits[:, 1] - logits[:, 0])[0].item())

    @staticmethod
    def _f1_at_threshold(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1)))
        fp = int(np.sum(predictions & (labels == 0)))
        fn = int(np.sum((~predictions) & (labels == 1)))
        return 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    def _current_threshold(self) -> float:
        train = self.cfg.training
        if len(self.threshold_history) < train.threshold_min_history:
            return float(train.threshold_default)
        if self.threshold_updates % max(1, train.threshold_refresh_every) == 0:
            probabilities = np.asarray([row[0] for row in self.threshold_history], dtype=np.float64)
            labels = np.asarray([row[1] for row in self.threshold_history], dtype=np.int64)
            best_score = -1.0
            best_threshold = float(train.threshold_default)
            for threshold in np.linspace(0.0, 1.0, max(2, train.threshold_grid_size)):
                score = self._f1_at_threshold(probabilities, labels, float(threshold))
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
            self.cached_threshold = best_threshold
        return self.cached_threshold

    def _biased_threshold(self, uid: str, base: float) -> float:
        train = self.cfg.training
        if not train.user_threshold_posrate_bias or self.user_record_counts[uid] < train.user_threshold_min_samples:
            return base
        user_rate = self.positive_record_counts[uid] / max(1, self.user_record_counts[uid])
        gid = self.group_assignment.get(uid)
        members = self.group_set.get(gid, {uid}) if gid is not None else {uid}
        positives = sum(self.positive_record_counts[u] for u in members)
        observations = sum(self.user_record_counts[u] for u in members)
        group_rate = positives / observations if observations else user_rate
        adjusted = base - train.user_threshold_posrate_bias_alpha * (user_rate - group_rate)
        return float(np.clip(adjusted, 0.05, 0.95))

    def _predict(self, triplet: DataTriplet) -> tuple[float, float, str, Dict[str, torch.Tensor], Dict[str, float]]:
        self.model.eval()
        with torch.no_grad():
            logits, adapter_source, experts, weights = self._prediction_logits(triplet)
            raw_probability = self._positive_probability(logits)
        history = self.smooth_probabilities[triplet.user_id]
        history.append(raw_probability)
        probability = float(np.mean(history))
        return probability, raw_probability, adapter_source, experts, weights

    def _update_mix_weights(
        self, uid: str, experts: Mapping[str, torch.Tensor], weights: Mapping[str, float], label: int,
    ) -> None:
        if not experts:
            return
        eta = self.cfg.grouping.low_evidence_mixing_eta
        updated = {}
        with torch.no_grad():
            for key, logits in experts.items():
                loss = float(self._task_loss(self._apply_bias(logits, uid), label).item())
                updated[key] = float(weights[key]) * math.exp(-eta * loss)
        total = sum(updated.values())
        if total > 0.0:
            self.mix_weights[uid] = {key: value / total for key, value in updated.items()}

    def _update_gradient_history(self, uid: str) -> np.ndarray | None:
        grad = self.model.backbone.gradient_projection.grad
        gradient_direction = None
        if grad is not None:
            gradient_direction = grad.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=True)
            if np.all(np.isfinite(gradient_direction)) and np.linalg.norm(gradient_direction) > 1e-12:
                decay = self.cfg.training.gradient_history_decay
                old = self.gradient_history.get(uid)
                updated = gradient_direction if old is None else decay * old + (1.0 - decay) * gradient_direction
                self.gradient_history[uid] = updated.copy()
                if uid not in self.interval_gradient_history:
                    self.interval_gradient_history[uid] = gradient_direction.copy()
                else:
                    self.interval_gradient_history[uid] = (
                        decay * self.interval_gradient_history[uid]
                        + (1.0 - decay) * gradient_direction
                    )
        return gradient_direction

    def _learn_from_triplet(self, triplet: DataTriplet, raw_probability_before_update: float) -> None:
        uid = triplet.user_id
        adapter_key = self._routed_adapter_key(uid)
        optimizer = self.adapter_optimizers[adapter_key]
        self.model.train()
        self.backbone_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        is_warmup = triplet.index < self.cfg.grouping.backbone_warmup_steps
        hard_logits = self.model(self._tensor(triplet.features), adapter_key)
        logits = self._apply_bias(hard_logits, uid, warmup=is_warmup)
        loss = self._task_loss(logits, triplet.label)
        loss.backward()
        gradient_direction = self._update_gradient_history(uid)
        if is_warmup:
            self.backbone_optimizer.step()
        optimizer.step()
        if adapter_key == self._personal_adapter_key(uid):
            self.personal_adapter_update_count[uid] += 1
        if not is_warmup and gradient_direction is not None and uid in self.group_assignment:
            self.group_gradient_buffer[self.group_assignment[uid]].append(gradient_direction)
        self.backbone_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        if not is_warmup:
            ref = self.cfg.refinements
            current = self._ensure_user_bias(uid)
            gradient = raw_probability_before_update - float(triplet.label)
            updated = (1.0 - ref.per_user_bias_lr * ref.per_user_bias_l2) * current
            updated -= ref.per_user_bias_lr * gradient
            self.user_bias[uid] = float(np.clip(updated, -ref.per_user_bias_clip, ref.per_user_bias_clip))

    def _apply_group_gradient_update(self) -> int:
        group_means = [np.mean(np.stack(rows), axis=0) for rows in self.group_gradient_buffer.values() if rows]
        self.group_gradient_buffer.clear()
        if not group_means:
            return 0
        update = np.mean(np.stack(group_means), axis=0)
        norm = float(np.linalg.norm(update))
        clip = self.cfg.training.gradient_direction_clip_norm
        if norm > clip > 0.0:
            update = update * (clip / norm)
        tensor = torch.as_tensor(
            update.reshape(self.model.backbone.gradient_projection.shape),
            dtype=self.model.backbone.gradient_projection.dtype,
            device=self.device,
        )
        with torch.no_grad():
            self.model.backbone.gradient_projection.add_(
                tensor, alpha=-self.cfg.training.backbone_lr * self.cfg.training.backbone_lr_scale
            )
        return len(group_means)

    def _users_ready_for_grouping(self, initial: bool = False) -> List[str]:
        grouping = self.cfg.grouping
        minimum = grouping.warmup_initial_min_samples if initial else grouping.min_observations
        users = [
            uid for uid in sorted(self.interval_gradient_history)
            if self.user_record_counts[uid] >= minimum
        ]
        if not initial or len(users) >= grouping.warmup_min_grouping_users:
            return users
        fallback = [
            uid for uid in sorted(self.interval_gradient_history)
            if self.user_record_counts[uid] >= grouping.warmup_initial_min_samples_fallback
        ]
        return (
            fallback
            if len(fallback) >= grouping.warmup_initial_min_grouping_users_fallback
            else []
        )

    @staticmethod
    def _project_user_representations(
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

    def _recent_loss(self, uid: str, adapter_key: str) -> float:
        observations = self.recent_data_buffers[uid]
        losses = []
        self.model.eval()
        with torch.no_grad():
            for data_pair in observations:
                logits = self.model(self._tensor(data_pair.features), adapter_key)
                losses.append(float(self._task_loss(logits, data_pair.label).item()))
        return float(np.mean(losses))

    def _apply_dynamic_user_reassignment(
        self,
        clusters: Sequence[Set[str]],
        users_for_grouping: Sequence[str],
        personal_adapter_sources: Mapping[str, str],
        step: int,
    ) -> tuple[List[Set[str]], int]:
        ref = self.cfg.refinements
        result = [set(cluster) for cluster in clusters]
        diagnostics = {
            "skipped_cadence": 0,
            "skipped_group_count": 0,
            "users_for_grouping": len(users_for_grouping),
            "skipped_observations": 0,
            "skipped_buffer": 0,
            "skipped_unassigned": 0,
            "skipped_singleton": 0,
            "evaluated_users": 0,
            "moves": 0,
        }
        if self.group_update_count % max(1, ref.reassignment_every) != 0:
            diagnostics["skipped_cadence"] = 1
            self._last_reassignment_diagnostics = diagnostics
            return result, 0
        if len(result) < 2:
            diagnostics["skipped_group_count"] = 1
            self._last_reassignment_diagnostics = diagnostics
            return result, 0
        candidate_group_adapter_keys: Dict[int, str] = {}
        for index, cluster in enumerate(result):
            key = f"diagnostic::reassignment::{step}::{index}"
            self.model.set_adapter_mean(key, [personal_adapter_sources[uid] for uid in sorted(cluster)])
            candidate_group_adapter_keys[index] = key
        cluster_by_user = {
            uid: index for index, cluster in enumerate(result) for uid in cluster
        }
        moves = 0
        for uid in sorted(users_for_grouping):
            if self.user_record_counts[uid] < ref.reassignment_min_observations:
                diagnostics["skipped_observations"] += 1
                continue
            if len(self.recent_data_buffers[uid]) < ref.reassignment_min_buffer:
                diagnostics["skipped_buffer"] += 1
                continue
            if uid not in cluster_by_user:
                diagnostics["skipped_unassigned"] += 1
                continue
            current = cluster_by_user[uid]
            if len(result[current]) <= 1:
                diagnostics["skipped_singleton"] += 1
                continue
            diagnostics["evaluated_users"] += 1
            losses = {
                index: self._recent_loss(uid, key)
                for index, key in candidate_group_adapter_keys.items()
            }
            best = min(losses, key=losses.get)
            if (
                best != current
                and np.isfinite(losses[current])
                and np.isfinite(losses[best])
                and losses[current] - losses[best] > ref.reassignment_loss_margin
            ):
                result[current].discard(uid)
                result[best].add(uid)
                cluster_by_user[uid] = best
                moves += 1
        diagnostics["moves"] = moves
        self._last_reassignment_diagnostics = diagnostics
        return [cluster for cluster in result if cluster], moves

    def _initialize_group_adapter(
        self, gid: str, members: Set[str], personal_adapter_sources: Mapping[str, str], created: bool,
    ) -> None:
        key = f"group::{gid}"
        if created or key not in self.model._external_to_internal:
            sources = [personal_adapter_sources[uid] for uid in sorted(members)]
            self.model.set_adapter_mean(key, sources)
            self._ensure_adapter(key, reset_optimizer=True)
        else:
            self._ensure_adapter(key)

    def _initialize_personal_adapter_from_group(self, uid: str, gid: str) -> None:
        if uid in self.group_initialized_personal_adapters:
            return
        grouping = self.cfg.grouping
        if self.user_record_counts[uid] < grouping.personal_adapter_group_init_min_records:
            return
        personal_key = self._personal_adapter_key(uid)
        self.model.blend_adapter(
            personal_key, f"group::{gid}", grouping.personal_adapter_group_init_weight
        )
        self._ensure_adapter(personal_key, reset_optimizer=True)
        self.group_initialized_personal_adapters.add(uid)

    def _update_groups(self, step: int, initial: bool = False) -> None:
        users_for_grouping = self._users_ready_for_grouping(initial)
        if not users_for_grouping:
            return
        sources = {uid: self._personal_adapter_key(uid) for uid in users_for_grouping}
        projected_user_representations = self._project_user_representations(
            users_for_grouping, self.interval_gradient_history,
        )
        grouping = self.cfg.grouping
        target = grouping.minimum_group_count
        min_cosine = grouping.min_pair_cosine
        if initial:
            target = grouping.warmup_initial_target_groups
            if target <= 0:
                target = math.ceil(math.sqrt(len(users_for_grouping)))
            target = max(grouping.warmup_initial_min_groups, min(grouping.warmup_initial_max_groups, target))
            if grouping.warmup_initial_force_accept:
                min_cosine = -1.0
        result = agglomerative_groups(users_for_grouping, projected_user_representations, target, min_cosine)
        clusters, reassignment_moves = self._apply_dynamic_user_reassignment(
            result.clusters, users_for_grouping, sources, step,
        )
        previous = self.group_set
        old_assignment = self.group_assignment
        new_groups, self.next_group_index, created = match_clusters(previous, clusters, self.next_group_index)
        for gid, members in new_groups.items():
            self._initialize_group_adapter(gid, members, sources, gid in created)
        self.group_set = new_groups
        self.group_assignment = {uid: gid for gid, users in self.group_set.items() for uid in users}
        for uid, gid in self.group_assignment.items():
            self._initialize_personal_adapter_from_group(uid, gid)
        self.group_update_count += 1
        common = set(old_assignment) & set(self.group_assignment)
        churn = float(np.mean([old_assignment[u] != self.group_assignment[u] for u in common])) if common else 0.0
        self.group_trace.append({
            "step": step,
            "initial": initial,
            "users_for_grouping": len(users_for_grouping),
            "n_groups": len(self.group_set),
            "n_merges": result.merges,
            "n_reassignment_moves": reassignment_moves,
            "churn": churn,
            "stop_reason": "warmup_forced_target" if initial else result.stop_reason,
            "last_similarity": result.last_similarity,
            **{
                f"reassignment_{key}": value
                for key, value in self._last_reassignment_diagnostics.items()
            },
        })

    def _boundary(self, step: int, initial: bool) -> None:
        gradient_groups = self._apply_group_gradient_update()
        self._update_groups(step, initial=initial)
        if self.group_trace and self.group_trace[-1]["step"] == step:
            self.group_trace[-1]["gradient_update_groups"] = gradient_groups
        if initial:
            self.interval_gradient_history.clear()

    def process(self, triplet: DataTriplet) -> Dict[str, object]:
        uid = triplet.user_id
        self._personal_adapter_key(uid)
        probability, raw_probability, adapter_source, experts, weights = self._predict(triplet)
        threshold = self._biased_threshold(uid, self._current_threshold())
        is_warmup = triplet.index < self.cfg.grouping.backbone_warmup_steps
        row: Dict[str, object] = {
            "time_index": triplet.index,
            "timestamp": triplet.timestamp,
            "user_id": uid,
            "user_record_index": self.user_record_counts[uid] + 1,
            "label": triplet.label,
            "prediction": int(probability >= threshold),
            "probability": probability,
            "raw_probability": raw_probability,
            "threshold": threshold,
            "routed_adapter": adapter_source,
            "group_id": self.group_assignment.get(uid, ""),
            "learning_phase": "backbone_warmup" if is_warmup else "adapter_learning",
            "adapter_routing_state": (
                "backbone_warmup"
                if is_warmup
                else ("group" if uid in self.group_assignment else "personal")
            ),
            "during_backbone_warmup": is_warmup,
        }
        self.prediction_history.append(row)
        self._update_mix_weights(uid, experts, weights, triplet.label)
        self._learn_from_triplet(triplet, raw_probability)
        data_pair = RecentDataPair(triplet.features.copy(), triplet.label)
        self.recent_data_buffers[uid].append(data_pair)
        self.threshold_history.append((probability, triplet.label))
        self.threshold_updates += 1
        self.user_record_counts[uid] += 1
        self.positive_record_counts[uid] += int(triplet.label == 1)
        step = triplet.index + 1
        warmup = self.cfg.grouping.backbone_warmup_steps
        interval = self.cfg.grouping.group_update_interval
        if step == warmup:
            self._boundary(step, initial=True)
        elif step > warmup and (step - warmup) % max(1, interval) == 0:
            self._boundary(step, initial=False)
        return row

    def summarize(self) -> Dict[str, object]:
        y = [int(row["label"]) for row in self.prediction_history]
        predictions = [int(row["prediction"]) for row in self.prediction_history]
        post_rows = [
            row for row in self.prediction_history
            if not bool(row["during_backbone_warmup"])
        ]
        post_y = [int(row["label"]) for row in post_rows]
        post_predictions = [int(row["prediction"]) for row in post_rows]
        low_evidence = first_k_metrics(
            self.prediction_history, self.cfg.evaluation.first_k, self.cfg.evaluation.require_complete_first_k
        )
        return {
            "overall": binary_metrics(y, predictions),
            "post_warmup": binary_metrics(post_y, post_predictions),
            "low_evidence": low_evidence,
            "mean_first_k_f1": float(np.mean([row["f1_at_k"] for row in low_evidence])) if low_evidence else float("nan"),
            "n_groups_final": len(self.group_set),
            "mean_groups": float(np.mean([row["n_groups"] for row in self.group_trace])) if self.group_trace else 0.0,
            "n_data_triplets": len(self.prediction_history),
            "n_users": len(self.user_record_counts),
            "configuration": "main",
            "group_adapter_construction": "merged_trainable",
            "personal_adapter_updates": int(sum(self.personal_adapter_update_count.values())),
        }

    def run(self, stream: Sequence[DataTriplet]) -> Dict[str, object]:
        for triplet in stream:
            self.process(triplet)
        return self.summarize()
