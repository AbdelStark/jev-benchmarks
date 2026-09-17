import math
from pathlib import Path

import pytest
from conftest import make_config

from jev_benchmarks.io import read_jsonl, write_jsonl
from jev_benchmarks.models import Prediction
from jev_benchmarks.runner import _make_backend, _validate_prediction, run_backend


def test_rounding_tolerance_preserves_raw_sum_and_normalizes() -> None:
    prediction = Prediction(
        experiment_id="test",
        backend="test",
        model_requested="test",
        model_resolved="test",
        dataset="fixture",
        example_id="1",
        target_index=0,
        predicted_index=0,
        labels=("a", "b"),
        probabilities=(0.50, 0.49),
        latency_seconds=0.1,
    )
    normalized = _validate_prediction(prediction)
    assert normalized.probability_sum_raw == 0.99
    assert sum(normalized.probabilities) == 1.0


@pytest.mark.parametrize(
    "probabilities",
    [(), (math.nan, 1.0), (-0.1, 1.1), (0.4, 0.4)],
)
def test_invalid_probability_vectors_are_rejected(probabilities: tuple[float, ...]) -> None:
    prediction = Prediction(
        "test",
        "test",
        "test",
        "test",
        "fixture",
        "1",
        0,
        0,
        ("a", "b"),
        probabilities,
        0.1,
    )
    with pytest.raises(ValueError):
        _validate_prediction(prediction)


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        _make_backend(make_config(tmp_path), "unknown")


def test_run_backend_is_resumable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(tmp_path)
    example_row = {
        "dataset": "fixture",
        "task": "topic",
        "example_id": "fixture:0",
        "text": "hello",
        "text_sha256": "hash",
        "labels": ["a", "b"],
        "target_index": 1,
    }
    write_jsonl(config.output_dir / "manifest.jsonl", [example_row])

    class FakeBackend:
        name = "jev"

        def __init__(self) -> None:
            self.calls = 0
            self.closed = False

        def warmup(self, example) -> None:
            return None

        def predict(self, experiment_id, example) -> Prediction:
            self.calls += 1
            return Prediction(
                experiment_id,
                "jev",
                "jev",
                "jev-resolved",
                example.dataset,
                example.example_id,
                example.target_index,
                1,
                example.labels,
                (0.1, 0.9),
                0.2,
            )

        def close(self) -> None:
            self.closed = True

    backend = FakeBackend()
    monkeypatch.setattr("jev_benchmarks.runner._make_backend", lambda config, name: backend)
    output = run_backend(config, "jev")
    assert len(read_jsonl(output)) == 1
    assert backend.calls == 1
    assert backend.closed
    assert run_backend(config, "jev") == output
    assert backend.calls == 1
