from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jev_benchmarks.config import load_config


def write_config(path: Path, **overrides: object) -> Path:
    raw = {
        "experiment_id": "test",
        "seed": 1,
        "dataset": {"samples_per_dataset": 2, "datasets": [{"name": "x"}]},
        "models": {"jev": {"model_id": "jev"}},
        "metrics": {"ece_bins": 10, "error_budget": 0.05, "bootstrap_resamples": 10},
        "output_dir": "results/test",
    }
    raw.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_load_config_resolves_relative_output(tmp_path: Path) -> None:
    path = write_config(tmp_path / "configs" / "test.yaml")
    config = load_config(path)
    assert config.experiment_id == "test"
    assert config.seed == 1
    assert config.output_dir == tmp_path / "results" / "test"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dataset": {"samples_per_dataset": 0, "datasets": [{"name": "x"}]}}, "positive"),
        (
            {"metrics": {"ece_bins": 0, "error_budget": 0.05, "bootstrap_resamples": 10}},
            "ece_bins",
        ),
        (
            {"metrics": {"ece_bins": 10, "error_budget": 1.1, "bootstrap_resamples": 10}},
            "error_budget",
        ),
        (
            {"metrics": {"ece_bins": 10, "error_budget": 0.05, "bootstrap_resamples": 0}},
            "bootstrap_resamples",
        ),
        ({"models": {}}, "model backend"),
    ],
)
def test_load_config_rejects_invalid_contract(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    path = write_config(tmp_path / "test.yaml", **override)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "test.yaml"
    path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root"):
        load_config(path)
