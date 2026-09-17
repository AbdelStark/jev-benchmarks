from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import make_config

from jev_benchmarks.data import _balanced_indices, _class_count, load_examples, prepare_manifest


def test_class_count_from_repeated_text() -> None:
    assert _class_count(["first", "first", "first", "second", "second", "second"]) == 3


def test_balanced_indices_are_deterministic_and_cover_classes() -> None:
    targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    first = _balanced_indices(targets, 6, 42)
    assert first == _balanced_indices(targets, 6, 42)
    assert {targets[index] for index in first} == {0, 1, 2}


def test_class_count_rejects_single_group() -> None:
    with pytest.raises(ValueError, match="infer"):
        _class_count(["same", "same"])


class FakeRows:
    def __init__(self) -> None:
        self.rows = [
            {"text": "one", "hypothesis": "label a", "labels": 1},
            {"text": "one", "hypothesis": "label b", "labels": 0},
            {"text": "two", "hypothesis": "label a", "labels": 0},
            {"text": "two", "hypothesis": "label b", "labels": 1},
            {"text": "oos", "hypothesis": "label a", "labels": 0},
            {"text": "oos", "hypothesis": "label b", "labels": 0},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: int | str):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def test_load_examples_excludes_rows_without_one_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = SimpleNamespace(load_dataset=lambda *args, **kwargs: FakeRows())
    monkeypatch.setitem(sys.modules, "datasets", module)
    config = make_config(tmp_path)
    examples = load_examples(config)
    assert [example.example_id for example in examples] == ["fixture:0", "fixture:1"]
    assert [example.target_index for example in examples] == [0, 1]

    manifest = prepare_manifest(config)
    assert manifest.exists()
    assert prepare_manifest(config) == manifest
