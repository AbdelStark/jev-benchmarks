from __future__ import annotations

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
        "accuracy": float(np.mean(correct)),
        "macro_f1": _macro_f1(targets, predicted, probabilities.shape[1]),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(
            -np.mean(
                np.log(np.clip(probabilities[np.arange(len(valid)), targets], 1e-12, 1))
            )
        ),
        "ece": ece,
        "coverage_at_error_budget": coverage,
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
                    "error": None,
                }
            )
        )
    return output
