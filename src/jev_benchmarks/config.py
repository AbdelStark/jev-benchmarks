from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def experiment_id(self) -> str:
        return str(self.raw["experiment_id"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def output_dir(self) -> Path:
        value = Path(self.raw["output_dir"])
        return value if value.is_absolute() else self.path.parent.parent / value


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    required = {"experiment_id", "seed", "dataset", "models", "metrics", "output_dir"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if not raw["dataset"]["datasets"]:
        raise ValueError("at least one dataset is required")
    if int(raw["dataset"]["samples_per_dataset"]) <= 0:
        raise ValueError("samples_per_dataset must be positive")
    if int(raw["metrics"]["ece_bins"]) <= 0:
        raise ValueError("ece_bins must be positive")
    error_budget = float(raw["metrics"]["error_budget"])
    if not 0 <= error_budget <= 1:
        raise ValueError("error_budget must be in [0, 1]")
    if int(raw["metrics"]["bootstrap_resamples"]) <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not isinstance(raw["models"], dict) or not raw["models"]:
        raise ValueError("at least one model backend is required")
    return BenchmarkConfig(raw=raw, path=config_path)
