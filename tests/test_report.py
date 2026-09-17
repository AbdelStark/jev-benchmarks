from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_config

from jev_benchmarks.io import write_jsonl
from jev_benchmarks.models import Example, Prediction
from jev_benchmarks.report import _latest_predictions, build_report


def _prediction(backend: str, example: Example, probabilities: tuple[float, ...]) -> Prediction:
    return Prediction(
        "fixture",
        backend,
        backend,
        f"{backend}-resolved",
        example.dataset,
        example.example_id,
        example.target_index,
        max(range(len(probabilities)), key=probabilities.__getitem__),
        example.labels,
        probabilities,
        0.1,
    )


def test_report_is_machine_readable_and_checks_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    examples = [
        Example("fixture", "topic", "fixture:0", "zero", "h0", ("a", "b"), 0),
        Example("fixture", "topic", "fixture:1", "one", "h1", ("a", "b"), 1),
    ]
    write_jsonl(config.output_dir / "manifest.jsonl", [row.to_dict() for row in examples])
    for backend, vectors in {
        "gliner": [(0.6, 0.4), (0.6, 0.4)],
        "jev": [(0.9, 0.1), (0.1, 0.9)],
    }.items():
        rows = [
            _prediction(backend, example, vector)
            for example, vector in zip(examples, vectors, strict=True)
        ]
        write_jsonl(
            config.output_dir / f"predictions-{backend}.jsonl",
            [row.to_dict() for row in rows],
        )
    json_path, markdown_path = build_report(config)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["resolved_models"]["jev"] == ["jev-resolved"]
    assert len(payload["artifacts"]["manifest_sha256"]) == 64
    assert "Paired comparison" in markdown_path.read_text(encoding="utf-8")


def test_latest_prediction_wins() -> None:
    rows = [
        _prediction("jev", Example("d", "t", "id", "x", "h", ("a", "b"), 0), vector).to_dict()
        for vector in ((0.4, 0.6), (0.9, 0.1))
    ]
    assert _latest_predictions(rows)[0].probabilities == (0.9, 0.1)


def test_report_rejects_incomplete_predictions(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    example = Example("fixture", "topic", "fixture:0", "zero", "h0", ("a", "b"), 0)
    write_jsonl(config.output_dir / "manifest.jsonl", [example.to_dict()])
    with pytest.raises(FileNotFoundError, match="gliner"):
        build_report(config)
