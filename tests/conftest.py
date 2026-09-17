from __future__ import annotations

from pathlib import Path

from jev_benchmarks.config import BenchmarkConfig


def make_config(root: Path, *, resamples: int = 20) -> BenchmarkConfig:
    raw = {
        "experiment_id": "fixture",
        "protocol_revision": "fixture-v1",
        "seed": 7,
        "dataset": {
            "repository": "fixture/repo",
            "revision": "abc123",
            "samples_per_dataset": 2,
            "datasets": [{"name": "fixture", "task": "topic"}],
        },
        "models": {
            "gliner": {"model_id": "fixture/gliner", "revision": "def456"},
            "jev": {"model_id": "fixture-jev", "question": "Choose a label"},
        },
        "metrics": {
            "ece_bins": 5,
            "error_budget": 0.05,
            "bootstrap_resamples": resamples,
        },
        "output_dir": str(root / "runs" / "fixture"),
    }
    config_path = root / "fixture.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    return BenchmarkConfig(raw=raw, path=config_path)
