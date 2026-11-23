from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    depth: int = 2
    num_heads: int = 4
    ff_mult: int = 4
    lora_rank: int = 4
    lora_alpha: float = 8.0
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    backbone_lr: float = 2e-4
    adapter_lr: float = 7e-3
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    positive_class_weight: float = 5.0
    weight_decay: float = 1e-4
    gradient_history_decay: float = 0.95
    backbone_lr_scale: float = 0.3
    gradient_direction_clip_norm: float = 1.0
    smoothing_window: int = 5
    threshold_window: int = 1000
    threshold_grid_size: int = 101
    threshold_min_history: int = 200
    threshold_refresh_every: int = 100
    threshold_default: float = 0.5
    user_threshold_posrate_bias: bool = False
    user_threshold_posrate_bias_alpha: float = 0.0
    user_threshold_min_samples: int = 3


@dataclass
class GroupingConfig:
    backbone_warmup_steps: int = 300
    min_observations: int = 20
    group_update_interval: int = 100
    minimum_group_count: int = 4
    min_pair_cosine: float = 0.0
    warmup_initial_min_samples: int = 5
    warmup_initial_min_samples_fallback: int = 1
    warmup_initial_min_grouping_users_fallback: int = 2
    warmup_min_grouping_users: int = 10
    warmup_initial_force_accept: bool = True
    warmup_initial_target_groups: int = 0
    warmup_initial_min_groups: int = 2
    warmup_initial_max_groups: int = 12
    personal_adapter_group_init_min_records: int = 3
    personal_adapter_group_init_weight: float = 0.35
    low_evidence_mixing_max_records: int = 20
    low_evidence_mixing_eta: float = 0.5
    low_evidence_prior_global: float = 0.30
    low_evidence_prior_personal: float = 0.50
    low_evidence_prior_group: float = 0.20


@dataclass
class RefinementConfig:
    per_user_bias_lr: float = 0.005
    per_user_bias_l2: float = 1e-3
    per_user_bias_clip: float = 2.0
    per_user_bias_init_group_weight: float = 0.5
    reassignment_min_observations: int = 20
    reassignment_min_buffer: int = 5
    reassignment_loss_margin: float = 0.01
    reassignment_every: int = 10


@dataclass
class EvaluationConfig:
    first_k: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    require_complete_first_k: bool = True


@dataclass
class ExperimentConfig:
    dataset: str = "custom"
    seed: int = 42
    device: str = "cpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    refinements: RefinementConfig = field(default_factory=RefinementConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _construct(data: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=str(data.get("dataset", "custom")),
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "cpu")),
        model=ModelConfig(**data.get("model", {})),
        training=TrainingConfig(**data.get("training", {})),
        grouping=GroupingConfig(**data.get("grouping", {})),
        refinements=RefinementConfig(**data.get("refinements", {})),
        evaluation=EvaluationConfig(**data.get("evaluation", {})),
    )


def load_config(path: str | Path | None = None, dataset: str | None = None) -> ExperimentConfig:
    if path is None:
        if dataset not in {"ces", "globem"}:
            raise ValueError("Specify dataset='ces' or dataset='globem', or supply a configuration file.")
        root = Path(__file__).resolve().parents[1]
        path = root / "configs" / f"{dataset}.yaml"
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _construct(data)
