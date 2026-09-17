from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .io import read_jsonl, runtime_metadata
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
    predictions: dict[str, list[Prediction]] = {}
    for backend in config.raw["models"]:
        path = config.output_dir / f"predictions-{backend}.jsonl"
        predictions[backend] = _latest_predictions(read_jsonl(path))
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
        "experiment_id": config.experiment_id,
        "config": config.raw,
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
        "Pilot results; not approved for public release. Coverage thresholds are descriptive "
        "on this slice.",
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
                lines.append(
                    f"| {dataset} | {metric} | {interval['difference']:.4f} | {ci} |"
                )
    markdown_path = config.output_dir / "report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
