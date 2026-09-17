from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__
from .config import BenchmarkConfig
from .io import read_jsonl, runtime_metadata, sha256_file
from .metrics import group_scores, paired_bootstrap, uniform_predictions
from .models import Prediction


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _latest_predictions(rows: list[dict[str, Any]]) -> list[Prediction]:
    latest: dict[str, Prediction] = {}
    for row in rows:
        prediction = Prediction.from_dict(row)
        latest[prediction.example_id] = prediction
    return list(latest.values())


def build_report(config: BenchmarkConfig) -> tuple[Path, Path]:
    options = {
        "ece_bins": int(config.raw["metrics"]["ece_bins"]),
        "error_budget": float(config.raw["metrics"]["error_budget"]),
    }
    manifest_path = config.output_dir / "manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    if not manifest_rows:
        raise FileNotFoundError(f"missing or empty manifest: {manifest_path}")
    manifest = {str(row["example_id"]): row for row in manifest_rows}
    predictions: dict[str, list[Prediction]] = {}
    prediction_paths: dict[str, Path] = {}
    for backend in config.raw["models"]:
        path = config.output_dir / f"predictions-{backend}.jsonl"
        prediction_paths[backend] = path
        predictions[backend] = _latest_predictions(read_jsonl(path))
        if not predictions[backend]:
            raise FileNotFoundError(f"missing or empty predictions for {backend}: {path}")
        prediction_ids = {row.example_id for row in predictions[backend]}
        if prediction_ids != set(manifest):
            missing = sorted(set(manifest) - prediction_ids)
            extra = sorted(prediction_ids - set(manifest))
            raise ValueError(
                f"{backend} predictions do not match manifest; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for prediction in predictions[backend]:
            source = manifest[prediction.example_id]
            if prediction.target_index != int(source["target_index"]) or prediction.labels != tuple(
                source["labels"]
            ):
                raise ValueError(f"{backend} prediction contract mismatch: {prediction.example_id}")
    reference = next((rows for rows in predictions.values() if rows), [])
    predictions["uniform"] = uniform_predictions(reference)
    comparisons: dict[str, Any] = {}
    if predictions.get("gliner") and predictions.get("jev"):
        for dataset in sorted({row.dataset for row in predictions["gliner"]}):
            left = [row for row in predictions["gliner"] if row.dataset == dataset]
            right = [row for row in predictions["jev"] if row.dataset == dataset]
            comparisons[dataset] = paired_bootstrap(
                left,
                right,
                resamples=int(config.raw["metrics"]["bootstrap_resamples"]),
                seed=config.seed,
            )
    payload = {
        "schema_version": 1,
        "benchmark_version": __version__,
        "experiment_id": config.experiment_id,
        "protocol_revision": config.raw.get("protocol_revision"),
        "config": config.raw,
        "artifacts": {
            "config_sha256": sha256_file(config.path),
            "manifest_sha256": sha256_file(manifest_path),
            "prediction_sha256": {
                backend: sha256_file(path) for backend, path in prediction_paths.items()
            },
        },
        "resolved_models": {
            backend: sorted({row.model_resolved for row in rows if row.error is None})
            for backend, rows in predictions.items()
            if backend != "uniform"
        },
        "runtime": runtime_metadata(),
        "results": {
            backend: group_scores(rows, **options) for backend, rows in predictions.items()
        },
        "paired_comparison_jev_minus_gliner": comparisons,
    }
    json_path = config.output_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    columns = [
        "model",
        "dataset",
        "n",
        "accuracy",
        "macro_f1",
        "brier",
        "nll",
        "ece",
        "coverage_at_error_budget",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "failures",
    ]
    lines = [
        f"# {config.experiment_id}",
        "",
        "Pilot results. Coverage thresholds are descriptive on this slice and are not deployable "
        "operating points.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for backend, datasets in payload["results"].items():
        for dataset, scores in datasets.items():
            row = {"model": backend, "dataset": dataset, **scores}
            values = " | ".join(_format(row.get(column, "")) for column in columns)
            lines.append(f"| {values} |")
    if comparisons:
        lines.extend(
            [
                "",
                "## Paired comparison: Jev minus GLiNER2.5",
                "",
                "Positive favors Jev for accuracy/F1; negative favors Jev for Brier/NLL.",
                "",
                "| dataset | metric | difference | 95% CI |",
                "| --- | --- | --- | --- |",
            ]
        )
        for dataset, metrics in comparisons.items():
            for metric, interval in metrics.items():
                ci = f"[{interval['ci95_low']:.4f}, {interval['ci95_high']:.4f}]"
                lines.append(f"| {dataset} | {metric} | {interval['difference']:.4f} | {ci} |")
    markdown_path = config.output_dir / "report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
