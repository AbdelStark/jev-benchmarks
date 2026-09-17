from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .io import read_jsonl, runtime_metadata
from .metrics import group_scores, uniform_predictions
from .models import Prediction


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_report(config: BenchmarkConfig) -> tuple[Path, Path]:
    options = {
        "ece_bins": int(config.raw["metrics"]["ece_bins"]),
        "error_budget": float(config.raw["metrics"]["error_budget"]),
    }
    predictions: dict[str, list[Prediction]] = {}
    for backend in config.raw["models"]:
        path = config.output_dir / f"predictions-{backend}.jsonl"
        predictions[backend] = [Prediction.from_dict(row) for row in read_jsonl(path)]
    reference = next((rows for rows in predictions.values() if rows), [])
    predictions["uniform"] = uniform_predictions(reference)
    payload = {
        "experiment_id": config.experiment_id,
        "config": config.raw,
        "runtime": runtime_metadata(),
        "results": {
            backend: group_scores(rows, **options) for backend, rows in predictions.items()
        },
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
    markdown_path = config.output_dir / "report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
