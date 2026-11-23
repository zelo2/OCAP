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
from .gradients import apply_group_gradient_update, project_user_representations, update_user_history
from .grouping import agglomerative_groups, eligible_users, match_clusters, nearest_group
from .metrics import Prediction, paper_metrics
from .model import OCAPModel, weighted_focal_loss
from .prediction import expert_weights, stream_threshold, user_threshold


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
        self.prediction_history: List[Prediction] = []
        self.group_gradient_buffer: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.smooth_probabilities: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, self.cfg.training.smoothing_window))
        )
        self.threshold_history: Deque[tuple[float, int]] = deque(
            maxlen=max(1, self.cfg.training.threshold_window)
        )
        self.cached_threshold = float(self.cfg.training.threshold_default)
        self.mix_weights: Dict[str, Dict[str, float]] = {}
        self.group_initialized_personal_adapters: Set[str] = set()
        # Keep adapter initialization in the same random sequence as the paper runs.
        self.model.ensure_adapter("global")

    def run(self, stream: Sequence[DataTriplet]) -> dict[str, float | None]:
        for triplet in stream:
            self.process(triplet)
        return paper_metrics(self.prediction_history)

    def process(self, triplet: DataTriplet) -> Prediction:
        """Predict from earlier observations, learn from this label, then regroup if due."""
        uid = triplet.user_id
        self._personal_adapter_key(uid)
        probability, raw_probability, experts, weights = self._predict(triplet)
        threshold = self._decision_threshold(uid)
        row = Prediction(uid, triplet.label, int(probability >= threshold), probability)
        self.prediction_history.append(row)

        self._update_mix_weights(uid, experts, weights, triplet.label)
        self._learn_from_triplet(triplet, raw_probability)
        data_pair = RecentDataPair(triplet.features.copy(), triplet.label)
        self.recent_data_buffers[uid].append(data_pair)
        self.threshold_history.append((probability, triplet.label))
        self.user_record_counts[uid] += 1
        self.positive_record_counts[uid] += int(triplet.label == 1)

        step = triplet.index + 1
        warmup = self.cfg.grouping.backbone_warmup_steps
        interval = self.cfg.grouping.group_update_interval
        if step == warmup:
            self._update_at_group_boundary(step, initial=True)
        elif step > warmup and (step - warmup) % max(1, interval) == 0:
            self._update_at_group_boundary(step, initial=False)
        return row

    def _predict(self, triplet: DataTriplet) -> tuple[float, float, Dict[str, torch.Tensor], Dict[str, float]]:
        self.model.eval()
        with torch.no_grad():
            logits, experts, weights = self._prediction_logits(triplet)
            raw_probability = float(torch.sigmoid(logits[:, 1] - logits[:, 0])[0].item())
        history = self.smooth_probabilities[triplet.user_id]
        history.append(raw_probability)
        probability = float(np.mean(history))
        return probability, raw_probability, experts, weights

    def _prediction_logits(
        self, triplet: DataTriplet,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        uid = triplet.user_id
        active = self._routed_adapter_key(uid)
        grouping = self.cfg.grouping
        can_mix = (
            triplet.index >= grouping.backbone_warmup_steps
            and self.user_record_counts[uid] < grouping.low_evidence_mixing_max_records
        )
        if can_mix:
            experts = self._expert_logits(triplet.features, uid)
            weights = expert_weights(experts, self.mix_weights.get(uid, {}), grouping)
            self.mix_weights[uid] = weights
            mixed = sum(float(weights[key]) * value for key, value in experts.items())
            return self._apply_bias(mixed, uid, warmup=False), experts, weights
        logits = self.model(self._tensor(triplet.features), active)
        return self._apply_bias(
            logits, uid, warmup=triplet.index < grouping.backbone_warmup_steps,
        ), {}, {}

    def _expert_logits(self, features: np.ndarray, uid: str) -> Dict[str, torch.Tensor]:
        x = self._tensor(features)
        hidden = self.model.backbone(x)
        base = self.model.head(hidden)
        experts: Dict[str, torch.Tensor] = {"global": base}
        personal_key = self._personal_adapter_key(uid)
        experts["personal"] = base + self.model.adapter_logits_from_hidden(hidden, personal_key)
        gid = nearest_group(uid, self.group_assignment, self.group_set, self.gradient_history)
        if gid is not None:
            group_key = f"group::{gid}"
            self._ensure_adapter(group_key)
            experts[group_key] = base + self.model.adapter_logits_from_hidden(hidden, group_key)
        return experts

    def _decision_threshold(self, uid: str) -> float:
        train = self.cfg.training
        if len(self.threshold_history) < train.threshold_min_history:
            base = float(train.threshold_default)
        else:
            self.cached_threshold = stream_threshold(
                self.threshold_history, len(self.prediction_history), self.cached_threshold, train,
            )
            base = self.cached_threshold
        return user_threshold(
            uid, base, self.user_record_counts, self.positive_record_counts,
            self.group_assignment, self.group_set, train,
        )

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
        gradient_direction = update_user_history(
            uid, self.model.backbone.gradient_projection.grad, self.cfg.training.gradient_history_decay,
            self.gradient_history, self.interval_gradient_history,
        )
        if is_warmup:
            self.backbone_optimizer.step()
        optimizer.step()
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

    def _update_at_group_boundary(self, step: int, initial: bool) -> None:
        apply_group_gradient_update(
            self.model.backbone.gradient_projection, self.group_gradient_buffer, self.cfg.training,
        )
        self._update_groups(step, initial=initial)
        if initial:
            self.interval_gradient_history.clear()

    def _update_groups(self, step: int, initial: bool = False) -> None:
        users_for_grouping = eligible_users(
            self.interval_gradient_history, self.user_record_counts, self.cfg.grouping, initial=initial,
        )
        if not users_for_grouping:
            return
        sources = {uid: self._personal_adapter_key(uid) for uid in users_for_grouping}
        projected_user_representations = project_user_representations(
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
        clusters = agglomerative_groups(users_for_grouping, projected_user_representations, target, min_cosine)
        clusters = self._apply_dynamic_user_reassignment(
            clusters, users_for_grouping, sources, step,
        )
        previous = self.group_set
        new_groups, self.next_group_index, created = match_clusters(previous, clusters, self.next_group_index)
        for gid, members in new_groups.items():
            self._initialize_group_adapter(gid, members, sources, gid in created)
        self.group_set = new_groups
        self.group_assignment = {uid: gid for gid, users in self.group_set.items() for uid in users}
        for uid, gid in self.group_assignment.items():
            self._initialize_personal_adapter_from_group(uid, gid)
        self.group_update_count += 1

    def _apply_dynamic_user_reassignment(
        self,
        clusters: Sequence[Set[str]],
        users_for_grouping: Sequence[str],
        personal_adapter_sources: Mapping[str, str],
        step: int,
    ) -> List[Set[str]]:
        ref = self.cfg.refinements
        result = [set(cluster) for cluster in clusters]
        if len(result) < 2 or self.group_update_count % max(1, ref.reassignment_every) != 0:
            return result
        candidate_group_adapter_keys: Dict[int, str] = {}
        for index, cluster in enumerate(result):
            key = f"diagnostic::reassignment::{step}::{index}"
            self.model.set_adapter_mean(key, [personal_adapter_sources[uid] for uid in sorted(cluster)])
            candidate_group_adapter_keys[index] = key
        cluster_by_user = {
            uid: index for index, cluster in enumerate(result) for uid in cluster
        }
        for uid in sorted(users_for_grouping):
            if self.user_record_counts[uid] < ref.reassignment_min_observations:
                continue
            if len(self.recent_data_buffers[uid]) < ref.reassignment_min_buffer:
                continue
            if uid not in cluster_by_user:
                continue
            current = cluster_by_user[uid]
            if len(result[current]) <= 1:
                continue
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
        for key in candidate_group_adapter_keys.values():
            self.model.remove_adapter(key)
        return [cluster for cluster in result if cluster]

    def _recent_loss(self, uid: str, adapter_key: str) -> float:
        observations = self.recent_data_buffers[uid]
        losses = []
        self.model.eval()
        with torch.no_grad():
            for data_pair in observations:
                logits = self.model(self._tensor(data_pair.features), adapter_key)
                losses.append(float(self._task_loss(logits, data_pair.label).item()))
        return float(np.mean(losses))

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

    def _routed_adapter_key(self, uid: str) -> str:
        if uid in self.group_assignment:
            key = f"group::{self.group_assignment[uid]}"
            self._ensure_adapter(key)
            return key
        return self._personal_adapter_key(uid)

    def _personal_adapter_key(self, uid: str) -> str:
        key = f"personal::{uid}"
        self._ensure_adapter(key)
        return key

    def _ensure_adapter(self, key: str, *, reset_optimizer: bool = False) -> None:
        adapter = self.model.ensure_adapter(key)
        if reset_optimizer or key not in self.adapter_optimizers:
            self.adapter_optimizers[key] = torch.optim.Adam(
                adapter.parameters(), lr=self.cfg.training.adapter_lr,
            )

    def _apply_bias(
        self, logits: torch.Tensor, uid: str, *, warmup: bool = False,
    ) -> torch.Tensor:
        if warmup:
            return logits
        offset = torch.zeros_like(logits)
        offset[:, 1] = self._ensure_user_bias(uid)
        return logits + offset

    def _ensure_user_bias(self, uid: str) -> float:
        if uid in self.user_bias:
            return float(self.user_bias[uid])
        ref = self.cfg.refinements
        value = ref.per_user_bias_init_group_weight * self._group_mean_user_bias(uid)
        value = float(np.clip(value, -ref.per_user_bias_clip, ref.per_user_bias_clip))
        self.user_bias[uid] = value
        return value

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

    def _task_loss(self, logits: torch.Tensor, label: int) -> torch.Tensor:
        target = torch.tensor([label], dtype=torch.long, device=self.device)
        train = self.cfg.training
        return weighted_focal_loss(
            logits, target, train.focal_alpha, train.focal_gamma, train.positive_class_weight
        )

    def _tensor(self, features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, dtype=torch.float32, device=self.device).reshape(1, -1)
