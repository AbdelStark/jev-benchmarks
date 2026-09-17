from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

from .adapters.base import Backend
from .adapters.gliner import GLiNERBackend
from .adapters.jev import JevBackend
from .config import BenchmarkConfig
from .data import prepare_manifest
from .io import append_jsonl, read_jsonl
from .models import Example, Prediction


def _validate_prediction(prediction: Prediction) -> Prediction:
    if len(prediction.labels) != len(prediction.probabilities):
        raise ValueError("labels and probabilities have different lengths")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in prediction.probabilities):
        raise ValueError("probabilities must be finite and in [0, 1]")
    total = sum(prediction.probabilities)
    if not math.isclose(total, 1.0, abs_tol=0.02):
        raise ValueError(f"probabilities sum to {total:.8f}, outside rounding tolerance")
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return replace(
        prediction,
        probabilities=tuple(value / total for value in prediction.probabilities),
        probability_sum_raw=total,
    )


def _make_backend(config: BenchmarkConfig, name: str) -> Backend:
    model = config.raw["models"][name]
    if name == "gliner":
        return GLiNERBackend(model["model_id"], model["revision"], model.get("device", "cpu"))
    if name == "jev":
        return JevBackend(model["model_id"], model["question"])
    raise ValueError(f"unknown backend: {name}")


def run_backend(config: BenchmarkConfig, backend_name: str) -> Path:
    manifest_path = config.output_dir / "manifest.jsonl"
    if not manifest_path.exists():
        prepare_manifest(config)
    examples = [Example.from_dict(row) for row in read_jsonl(manifest_path)]
    output = config.output_dir / f"predictions-{backend_name}.jsonl"
    completed = {row["example_id"] for row in read_jsonl(output) if row.get("error") is None}
    pending = [example for example in examples if example.example_id not in completed]
    if not pending:
        return output
    backend = _make_backend(config, backend_name)
    try:
        if backend_name == "gliner":
            backend.warmup(pending[0])
        for index, example in enumerate(pending, start=1):
            try:
                prediction = backend.predict(config.experiment_id, example)
                prediction = _validate_prediction(prediction)
            except Exception as exc:
                prediction = Prediction(
                    experiment_id=config.experiment_id,
                    backend=backend_name,
                    model_requested=str(config.raw["models"][backend_name]["model_id"]),
                    model_resolved="unknown",
                    dataset=example.dataset,
                    example_id=example.example_id,
                    target_index=example.target_index,
                    predicted_index=-1,
                    labels=example.labels,
                    probabilities=tuple(),
                    latency_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            append_jsonl(output, prediction.to_dict())
            print(f"[{backend_name}] {index}/{len(pending)} {example.example_id}", flush=True)
    finally:
        backend.close()
    return output
