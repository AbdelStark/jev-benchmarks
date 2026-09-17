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
    required = {"experiment_id", "seed", "dataset", "models", "metrics", "output_dir"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    if not raw["dataset"]["datasets"]:
        raise ValueError("at least one dataset is required")
    if int(raw["dataset"]["samples_per_dataset"]) <= 0:
        raise ValueError("samples_per_dataset must be positive")
    return BenchmarkConfig(raw=raw, path=config_path)
