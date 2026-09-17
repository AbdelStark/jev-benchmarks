from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from jev_benchmarks.adapters import gliner, jev
from jev_benchmarks.models import Example


def example() -> Example:
    return Example("fixture", "topic", "fixture:1", "hello", "hash", ("a", "b"), 1)


def test_gliner_adapter_preserves_label_order_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeClassifier:
        def eval(self):
            return self

        def classify(self, text, schema):
            return SimpleNamespace(probabilities=lambda task: {"a": 0.25, "b": 0.75})

    fake = FakeClassifier()
    monkeypatch.setattr(
        gliner.Classifier,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: fake),
    )
    backend = gliner.GLiNERBackend("model", "revision")
    backend.warmup(example())
    result = backend.predict("experiment", example())
    assert result.probabilities == (0.25, 0.75)
    assert result.predicted_index == 1
    assert result.model_resolved == "model@revision"
    assert backend.close() is None


def test_jev_adapter_maps_stable_option_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, model: str) -> None:
            self.model = model
            self.closed = False

        def system_one(self, **kwargs):
            assert list(kwargs["questions"]["label"].criteria) == ["label_000", "label_001"]
            answer = SimpleNamespace(probabilities={"label_000": 0.2, "label_001": 0.8})
            return SimpleNamespace(
                choices={"label": answer},
                model="jev-resolved",
                usage=SimpleNamespace(input_tokens=42),
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(jev, "TypeSafeClient", FakeClient)
    backend = jev.JevBackend("jev-latest", "Choose")
    assert backend.warmup(example()) is None
    result = backend.predict("experiment", example())
    assert result.probabilities == (0.2, 0.8)
    assert result.input_tokens == 42
    assert result.model_resolved == "jev-resolved"
    backend.close()
    assert cast(Any, backend.client).closed
