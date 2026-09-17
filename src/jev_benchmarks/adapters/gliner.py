from __future__ import annotations

import time
from typing import Any

from gliner2.classification import ClassificationSchema, Classifier

from ..models import Example, Prediction


class GLiNERBackend:
    name = "gliner"

    def __init__(self, model_id: str, revision: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.revision = revision
        self.classifier = Classifier.from_pretrained(
            model_id, revision=revision, device=device
        ).eval()
        self._schemas: dict[tuple[str, ...], Any] = {}

    def _schema(self, labels: tuple[str, ...]) -> Any:
        if labels not in self._schemas:
            self._schemas[labels] = ClassificationSchema().single("label", list(labels))
        return self._schemas[labels]

    def warmup(self, example: Example) -> None:
        self.classifier.classify(example.text, self._schema(example.labels))

    def predict(self, experiment_id: str, example: Example) -> Prediction:
        start = time.perf_counter()
        result = self.classifier.classify(example.text, self._schema(example.labels))
        latency = time.perf_counter() - start
        probabilities = tuple(
            float(result.probabilities("label")[label]) for label in example.labels
        )
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        return Prediction(
            experiment_id=experiment_id,
            backend=self.name,
            model_requested=self.model_id,
            model_resolved=f"{self.model_id}@{self.revision}",
            dataset=example.dataset,
            example_id=example.example_id,
            target_index=example.target_index,
            predicted_index=predicted,
            labels=example.labels,
            probabilities=probabilities,
            latency_seconds=latency,
        )

    def close(self) -> None:
        return None
