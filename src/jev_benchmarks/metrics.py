from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .models import Prediction


def _macro_f1(targets: np.ndarray, predictions: np.ndarray, n_classes: int) -> float:
    scores = []
    for class_id in range(n_classes):
        tp = int(np.sum((targets == class_id) & (predictions == class_id)))
        fp = int(np.sum((targets != class_id) & (predictions == class_id)))
        fn = int(np.sum((targets == class_id) & (predictions != class_id)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def score_predictions(
    predictions: Sequence[Prediction], *, ece_bins: int = 10, error_budget: float = 0.05
) -> dict[str, Any]:
    valid = [prediction for prediction in predictions if prediction.error is None]
    if not valid:
        return {"n": len(predictions), "valid": 0, "failures": len(predictions)}
    all_targets = np.array([row.target_index for row in predictions], dtype=int)
    all_predicted = np.array([row.predicted_index for row in predictions], dtype=int)
    targets = np.array([row.target_index for row in valid], dtype=int)
    predicted = np.array([row.predicted_index for row in valid], dtype=int)
    probabilities = np.array([row.probabilities for row in valid], dtype=float)
    confidence = probabilities.max(axis=1)
    correct = predicted == targets
    one_hot = np.eye(probabilities.shape[1])[targets]
    ece = 0.0
    edges = np.linspace(0, 1, ece_bins + 1)
    for bin_index in range(ece_bins):
        lower, upper = edges[bin_index], edges[bin_index + 1]
        mask = (confidence >= lower) & (
            (confidence <= upper) if bin_index == ece_bins - 1 else (confidence < upper)
        )
        if np.any(mask):
            confidence_gap = abs(
                float(np.mean(confidence[mask])) - float(np.mean(correct[mask]))
            )
            ece += float(np.mean(mask)) * confidence_gap
    order = np.argsort(-confidence)
    cumulative_errors = np.cumsum(~correct[order])
    accepted = np.arange(1, len(valid) + 1)
    feasible = accepted[(cumulative_errors / accepted) <= error_budget]
    coverage = 0.0 if len(feasible) == 0 else float(feasible.max() / len(valid))
    latencies = np.array([row.latency_seconds for row in valid])
    return {
        "n": len(predictions),
        "valid": len(valid),
        "failures": len(predictions) - len(valid),
        "accuracy": float(np.mean(all_targets == all_predicted)),
        "macro_f1": _macro_f1(all_targets, all_predicted, probabilities.shape[1]),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(
            -np.mean(
                np.log(np.clip(probabilities[np.arange(len(valid)), targets], 1e-12, 1))
            )
        ),
        "ece": ece,
        "mean_confidence": float(np.mean(confidence)),
        "true_label_zero_rate": float(
            np.mean(probabilities[np.arange(len(valid)), targets] == 0)
        ),
        "renormalized_vectors": sum(
            row.probability_sum_raw is not None
            and not math.isclose(row.probability_sum_raw, 1.0, abs_tol=1e-9)
            for row in valid
        ),
        "coverage_at_error_budget": coverage * len(valid) / len(predictions),
        "latency_p50_seconds": float(np.quantile(latencies, 0.50)),
        "latency_p95_seconds": float(np.quantile(latencies, 0.95)),
        "input_tokens_total": sum(row.input_tokens or 0 for row in valid),
    }


def group_scores(predictions: Sequence[Prediction], **kwargs: Any) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for row in predictions:
        grouped[row.dataset].append(row)
    return {dataset: score_predictions(rows, **kwargs) for dataset, rows in sorted(grouped.items())}


def uniform_predictions(reference: Sequence[Prediction]) -> list[Prediction]:
    output = []
    for row in reference:
        size = len(row.labels)
        probabilities = tuple([1.0 / size] * size)
        output.append(
            Prediction(
                **{
                    **row.__dict__,
                    "backend": "uniform",
                    "model_requested": "uniform",
                    "model_resolved": "uniform",
                    "predicted_index": 0,
                    "probabilities": probabilities,
                    "latency_seconds": 0.0,
                    "input_tokens": None,
                    "probability_sum_raw": 1.0,
                    "error": None,
                }
            )
        )
    return output


def paired_bootstrap(
    left: Sequence[Prediction],
    right: Sequence[Prediction],
    *,
    resamples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return right-minus-left paired intervals on examples valid for both backends."""
    left_by_id = {row.example_id: row for row in left if row.error is None}
    right_by_id = {row.example_id: row for row in right if row.error is None}
    ids = sorted(left_by_id.keys() & right_by_id.keys())
    if not ids:
        return {}
    left_rows = [left_by_id[example_id] for example_id in ids]
    right_rows = [right_by_id[example_id] for example_id in ids]
    metrics = ("accuracy", "macro_f1", "brier", "nll")
    observed_left = score_predictions(left_rows)
    observed_right = score_predictions(right_rows)
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = {metric: [] for metric in metrics}
    strata: dict[int, np.ndarray] = {}
    for target in sorted({row.target_index for row in left_rows}):
        strata[target] = np.array(
            [index for index, row in enumerate(left_rows) if row.target_index == target]
        )
    for _ in range(resamples):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in strata.values()]
        )
        sampled_left = [left_rows[int(index)] for index in indices]
        sampled_right = [right_rows[int(index)] for index in indices]
        left_scores = score_predictions(sampled_left)
        right_scores = score_predictions(sampled_right)
        for metric in metrics:
            differences[metric].append(float(right_scores[metric] - left_scores[metric]))
    return {
        metric: {
            "difference": float(observed_right[metric] - observed_left[metric]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in differences.items()
    }
