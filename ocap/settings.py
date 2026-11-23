from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict

from .config import ExperimentConfig


@dataclass(frozen=True)
class MainConfiguration:
    dataset: str
    name: str
    gradient_history_alpha: float
    grouping_min_observations: int
    minimum_group_count: int
    reassignment_min_observations: int = 20
    reassignment_loss_margin: float = 0.01

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


MAIN_CONFIGURATIONS: Dict[tuple[str, str], MainConfiguration] = {
    ("ces", "main"): MainConfiguration(
        "ces", "main", 0.20, 10, 5,
    ),
    ("globem", "main"): MainConfiguration(
        "globem", "main", 0.10, 2, 5,
    ),
}


def get_main_configuration(dataset: str, name: str) -> MainConfiguration:
    key = (dataset.strip().lower(), name.strip().lower())
    if key not in MAIN_CONFIGURATIONS:
        available = ", ".join(f"{d}/{n}" for d, n in sorted(MAIN_CONFIGURATIONS))
        raise ValueError(f"unknown main_configuration {dataset}/{name}; available: {available}")
    return MAIN_CONFIGURATIONS[key]


def apply_main_configuration(
    config: ExperimentConfig, name: str,
) -> tuple[ExperimentConfig, MainConfiguration]:
    main_configuration = get_main_configuration(config.dataset, name)
    resolved = deepcopy(config)
    resolved.training.gradient_history_decay = 1.0 - main_configuration.gradient_history_alpha
    resolved.grouping.min_observations = main_configuration.grouping_min_observations
    resolved.grouping.minimum_group_count = main_configuration.minimum_group_count
    resolved.refinements.reassignment_min_observations = (
        main_configuration.reassignment_min_observations
    )
    resolved.refinements.reassignment_loss_margin = main_configuration.reassignment_loss_margin
    return resolved, main_configuration
