from __future__ import annotations

from pathlib import Path

from jev_benchmarks.io import append_jsonl, read_jsonl, sha256_file, write_jsonl
from jev_benchmarks.models import Example, Prediction


def test_jsonl_round_trip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rows.jsonl"
    write_jsonl(path, [{"b": 2, "a": 1}])
    append_jsonl(path, {"a": 3})
    assert read_jsonl(path) == [{"a": 1, "b": 2}, {"a": 3}]
    assert len(sha256_file(path)) == 64
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_model_records_round_trip() -> None:
    example = Example("d", "topic", "d:1", "text", "hash", ("a", "b"), 1)
    assert Example.from_dict(example.to_dict()) == example
    prediction = Prediction(
        "e", "m", "requested", "resolved", "d", "d:1", 1, 1, ("a", "b"), (0.2, 0.8), 0.1
    )
    assert Prediction.from_dict(prediction.to_dict()) == prediction
