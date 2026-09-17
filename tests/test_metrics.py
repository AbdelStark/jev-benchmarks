from __future__ import annotations

import pytest

from jev_benchmarks.metrics import paired_bootstrap, score_predictions
from jev_benchmarks.models import Prediction


def prediction(target: int, probabilities: tuple[float, ...]) -> Prediction:
    return Prediction(
        experiment_id="test",
        backend="test",
        model_requested="test",
        model_resolved="test",
        dataset="fixture",
        example_id=str(target),
        target_index=target,
        predicted_index=max(range(len(probabilities)), key=probabilities.__getitem__),
        labels=("a", "b"),
        probabilities=probabilities,
        latency_seconds=0.1,
    )


def test_perfect_predictions_have_expected_metrics() -> None:
    scores = score_predictions([prediction(0, (1.0, 0.0)), prediction(1, (0.0, 1.0))])
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert scores["brier"] == 0.0
    assert scores["nll"] == 0.0
    assert scores["ece"] == 0.0
    assert scores["coverage_at_error_budget"] == 1.0


def test_uniform_binary_brier_and_nll() -> None:
    scores = score_predictions([prediction(0, (0.5, 0.5)), prediction(1, (0.5, 0.5))])
    assert scores["brier"] == 0.5
    assert scores["nll"] == pytest.approx(0.693147, rel=1e-5)


def test_failures_are_explicit() -> None:
    failed = Prediction(
        **{
            **prediction(0, (1.0, 0.0)).__dict__,
            "predicted_index": -1,
            "probabilities": (),
            "error": "boom",
        }
    )
    scores = score_predictions([failed])
    assert scores == {"n": 1, "valid": 0, "failures": 1}


def test_paired_bootstrap_reports_right_minus_left() -> None:
    left = [prediction(0, (0.6, 0.4)), prediction(1, (0.6, 0.4))]
    right = [prediction(0, (0.9, 0.1)), prediction(1, (0.1, 0.9))]
    result = paired_bootstrap(left, right, resamples=100, seed=7)
    assert result["accuracy"]["difference"] == 0.5
    assert result["brier"]["difference"] < 0


def test_selective_coverage_never_splits_confidence_ties() -> None:
    rows = [prediction(0, (0.9, 0.1)), prediction(1, (0.9, 0.1))]
    scores = score_predictions(rows, error_budget=0.05)
    assert scores["coverage_at_error_budget"] == 0.0
    assert scores["confidence_threshold_at_error_budget"] is None


def test_paired_bootstrap_rejects_mismatched_contracts() -> None:
    left = [prediction(0, (0.6, 0.4))]
    right = [Prediction(**{**prediction(1, (0.4, 0.6)).__dict__, "example_id": "0"})]
    with pytest.raises(ValueError, match="contract mismatch"):
        paired_bootstrap(left, right, resamples=5, seed=1)
